"""Sandbox volume operations, namespaced under the sandbox clients."""

import asyncio
import math
import time
from typing import TYPE_CHECKING, Optional
from urllib.parse import quote

from ._polling import (
    CHECKPOINT_POLL_FAST_DELAY,
    CHECKPOINT_POLL_FAST_WINDOW,
    _checkpoint_poll_delay,
    _jittered_poll_delay,
)
from .exceptions import VolumeSyncFailedError, VolumeSyncSupersededError
from .models import SandboxCheckpoint, Volume, VolumeListResponse, VolumeSync, VolumeVisibility

if TYPE_CHECKING:
    from .core import APIClient, AsyncAPIClient
    from .sandbox import AsyncSandboxClient, SandboxClient


def _resolved_team_id(
    client: "APIClient | AsyncAPIClient", team_id: Optional[str]
) -> Optional[str]:
    """Explicit team wins; otherwise use the team from the client config."""
    return team_id if team_id is not None else client.config.team_id


def _quoted_volume_ref(name: str) -> str:
    """Quote a volume name or ``<namespace>/<name>`` ref for a route path.

    A ref has exactly one slash, and no segment may be empty, ``.`` or ``..``:
    the HTTP client normalizes dot segments, so an unchecked ref like
    ``../../sandbox/<id>`` would retarget the request at a different route.
    """
    namespace, slash, volume_name = name.partition("/")
    segments = [namespace, volume_name] if slash else [name]
    if not name or "" in segments or "/" in volume_name:
        raise ValueError(
            f"invalid volume name or ref '{name}': expected <name> or <namespace>/<name>"
        )
    if any(segment in (".", "..") for segment in segments):
        raise ValueError(f"invalid volume name or ref '{name}': '.' and '..' are not allowed")
    return "/".join(quote(segment, safe="") for segment in segments)


def _volume_path(name: str) -> str:
    return f"/sandbox/volumes/{_quoted_volume_ref(name)}"


def _publish_path(sandbox_id: str, name: str) -> str:
    return f"/sandbox/{sandbox_id}/volumes/{_quoted_volume_ref(name)}/publish"


def _sync_path(sandbox_id: str, name: str) -> str:
    return f"/sandbox/{sandbox_id}/volumes/{_quoted_volume_ref(name)}/sync"


# The server fails a sync past its 10-minute deadline, enforced on a ~1-minute
# tick, so the wait must outlast both for DEADLINE and ROLLBACK_FAILED to
# reach the caller instead of a client-side timeout.
SYNC_WAIT_TIMEOUT_SECONDS = 720.0


def _validate_sync_timeout(timeout_seconds: float) -> None:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and greater than zero")


def _require_pinned_sync(
    sandbox_id: str, name: str, started: VolumeSync, polled: VolumeSync
) -> None:
    # A new sync starts only after the previous one is terminal, so a different
    # sync_id means the polled row no longer describes the sync this call started.
    if polled.sync_id != started.sync_id:
        raise VolumeSyncSupersededError(sandbox_id, name, started.sync_id, polled.sync_id)


def _sync_timeout_message(
    sandbox_id: str, name: str, timeout_seconds: float, sync: VolumeSync
) -> str:
    return (
        f"Volume '{name}' in sandbox {sandbox_id} did not finish syncing "
        f"within {timeout_seconds:g}s (state={sync.state})"
    )


def _name_params(
    client: "APIClient | AsyncAPIClient", team_id: Optional[str], name: str
) -> dict[str, object]:
    # A ref carries its own namespace, so a team_id beside it would be ambiguous.
    if "/" in name:
        return {}
    resolved_team_id = _resolved_team_id(client, team_id)
    if resolved_team_id:
        return {"team_id": resolved_team_id}
    return {}


class Volumes:
    """Volume operations for one owning ``SandboxClient``.

    Reaches the platform through the parent client's transport and waits for
    publishes with the parent's checkpoint waiter; it owns no transport,
    waiter, batcher, or lifecycle of its own.

    Name and ref arguments are validated before any request: a ref has at most
    one slash and no empty, ``.`` or ``..`` segment, else ValueError.
    """

    def __init__(self, parent: "SandboxClient"):
        self._parent = parent

    def create(
        self, name: str, *, team_id: Optional[str] = None, size_gb: Optional[int] = None
    ) -> Volume:
        """Create an empty volume in the caller's personal scope or in a team.

        ``name`` is a bare name, unique in that scope; refs name existing
        volumes and are not accepted here. ``size_gb`` is the requested
        capacity in whole GiB; omitted means the base image size (100 GiB).
        """
        payload: dict[str, object] = {"name": name}
        resolved_team_id = _resolved_team_id(self._parent.client, team_id)
        if resolved_team_id:
            payload["team_id"] = resolved_team_id
        if size_gb is not None:
            payload["size_gb"] = size_gb
        response = self._parent.client.request("POST", "/sandbox/volumes", json=payload)
        return Volume.model_validate(response)

    def list(
        self,
        *,
        team_id: Optional[str] = None,
        page: int = 1,
        per_page: int = 50,
    ) -> VolumeListResponse:
        """List volumes in the caller's personal scope or in a team."""
        params: dict[str, object] = {"page": page, "per_page": per_page}
        resolved_team_id = _resolved_team_id(self._parent.client, team_id)
        if resolved_team_id:
            params["team_id"] = resolved_team_id
        response = self._parent.client.request("GET", "/sandbox/volumes", params=params)
        return VolumeListResponse.model_validate(response)

    def get(self, name: str, *, team_id: Optional[str] = None) -> Volume:
        """Get a volume by name in the caller's scope, or by ref.

        A ``<namespace>/<name>`` ref to another account's PUBLIC volume
        returns its summary only (name, namespace, size, visibility, head
        size, created_at): that response has no volume id, owner fields, head
        checkpoint or account storage. A ref to a missing or PRIVATE foreign
        volume raises APIError 404.
        """
        response = self._parent.client.request(
            "GET",
            _volume_path(name),
            params=_name_params(self._parent.client, team_id, name),
        )
        return Volume.model_validate(response)

    def delete(self, name: str, *, team_id: Optional[str] = None) -> None:
        """Delete a volume and its data; any team member can delete a team
        volume.

        DELETE is retried on transient 5xx, so a 503 while the server is
        still removing data can succeed on a later attempt. A ref to a
        volume owned by another account raises APIError 404.
        """
        self._parent.client.request(
            "DELETE",
            _volume_path(name),
            params=_name_params(self._parent.client, team_id, name),
        )

    def grow(self, name: str, size_gb: int, *, team_id: Optional[str] = None) -> Volume:
        """Grow a volume's capacity to ``size_gb`` whole GiB.

        Grow-only: the server rejects a size at or below the volume's
        current effective size (its set size, or the 100 GiB base image
        size when unset). The new size applies to sandboxes created after
        the change; running attachments keep their current size. A ref to a
        volume owned by another account raises APIError 404.
        """
        response = self._parent.client.request(
            "PATCH",
            _volume_path(name),
            json={"size_gb": size_gb},
            params=_name_params(self._parent.client, team_id, name),
        )
        return Volume.model_validate(response)

    def set_visibility(
        self, name: str, visibility: VolumeVisibility, *, team_id: Optional[str] = None
    ) -> Volume:
        """Make a volume PUBLIC or PRIVATE.

        A PUBLIC volume is mountable and syncable by any account through its
        ``<namespace>/<name>`` ref. Making it PRIVATE stops new recipient
        mounts, syncs and gets; running recipient sandboxes keep their mount.
        Only the volume's owner can change it: its user for a personal
        volume, any member of its team for a team volume. Repeating the
        current value is a no-op.
        """
        response = self._parent.client.request(
            "PATCH",
            _volume_path(name),
            json={"visibility": visibility.value},
            params=_name_params(self._parent.client, team_id, name),
        )
        return Volume.model_validate(response)

    def publish(self, sandbox_id: str, name: str, *, wait: bool = True) -> SandboxCheckpoint:
        """Publish one of a running sandbox's volume attachments.

        ``wait`` is keyword-only so an old third positional argument (the
        removed ``force`` flag) raises instead of silently meaning
        ``wait=False``.
        The server rejects the publish unless the sandbox is attached to a
        writable volume with that name or ref (404), makes the volume
        read-only (400), or has another sandbox's publish in flight (409).
        Publishing a volume owned by another account is rejected: 403 when
        this sandbox has it attached, 404 otherwise. Only its owner can move
        the shared head. With
        wait=True (the default), the returned checkpoint is the published
        head once DURABLE; on servers that report volume-merge status the
        wait also covers the checkpoint's merge, and a failed merge raises
        RuntimeError with merge_error. A failed publish raises RuntimeError
        and an unfinished publish TimeoutError.
        """
        response = self._parent.client.request(
            "POST",
            _publish_path(sandbox_id, name),
            json={},
        )
        checkpoint = SandboxCheckpoint.model_validate(response)
        if not wait:
            return checkpoint
        return self._parent.wait_for_checkpoint(checkpoint.id)

    def sync(
        self,
        sandbox_id: str,
        name: str,
        discard: bool = False,
        wait: bool = True,
        timeout_seconds: float = SYNC_WAIT_TIMEOUT_SECONDS,
    ) -> VolumeSync:
        """Sync one of a running sandbox's volume attachments to the volume's
        committed head, re-attaching the volume at that head.

        Read-only and discard syncs re-attach without publishing; a read-write
        sync publishes this sandbox's unpublished writes first and waits for
        their merge. A volume owned by another account always syncs as
        discard, whatever ``discard`` says: this sandbox's unpublished writes
        are replaced by the volume's current head, and the stored sync
        records that. With wait=True (the default), the returned sync is the
        final state once DONE; a FAILED sync raises VolumeSyncFailedError and
        an unfinished sync TimeoutError. The wait polls at a fixed fast
        cadence for its first seconds and then with exponential backoff,
        sharing the checkpoint waiter's pacing.
        """
        _validate_sync_timeout(timeout_seconds)
        response = self._parent.client.request(
            "POST",
            _sync_path(sandbox_id, name),
            json={"discard": discard},
        )
        sync = VolumeSync.model_validate(response)
        if not wait:
            return sync
        now = time.monotonic()
        deadline = now + timeout_seconds
        phase_start = now
        slow_index = 0
        while sync.state not in ("DONE", "FAILED"):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(_sync_timeout_message(sandbox_id, name, timeout_seconds, sync))
            if time.monotonic() - phase_start >= CHECKPOINT_POLL_FAST_WINDOW:
                time.sleep(min(_checkpoint_poll_delay(slow_index), remaining))
                slow_index += 1
            else:
                time.sleep(min(_jittered_poll_delay(CHECKPOINT_POLL_FAST_DELAY), remaining))
            response = self._parent.client.request("GET", _sync_path(sandbox_id, name))
            polled = VolumeSync.model_validate(response)
            _require_pinned_sync(sandbox_id, name, sync, polled)
            sync = polled
        if sync.state == "FAILED":
            raise VolumeSyncFailedError(sandbox_id, name, sync.failure)
        return sync


class AsyncVolumes:
    """Async volume operations for one owning ``AsyncSandboxClient``.

    Reaches the platform through the parent client's transport and waits for
    publishes with the parent's checkpoint waiter; it owns no transport,
    waiter, batcher, or lifecycle of its own.

    Name and ref arguments are validated before any request: a ref has at most
    one slash and no empty, ``.`` or ``..`` segment, else ValueError.
    """

    def __init__(self, parent: "AsyncSandboxClient"):
        self._parent = parent

    async def create(
        self, name: str, *, team_id: Optional[str] = None, size_gb: Optional[int] = None
    ) -> Volume:
        """Create an empty volume in the caller's personal scope or in a team.

        ``name`` is a bare name, unique in that scope; refs name existing
        volumes and are not accepted here. ``size_gb`` is the requested
        capacity in whole GiB; omitted means the base image size (100 GiB).
        """
        payload: dict[str, object] = {"name": name}
        resolved_team_id = _resolved_team_id(self._parent.client, team_id)
        if resolved_team_id:
            payload["team_id"] = resolved_team_id
        if size_gb is not None:
            payload["size_gb"] = size_gb
        response = await self._parent.client.request("POST", "/sandbox/volumes", json=payload)
        return Volume.model_validate(response)

    async def list(
        self,
        *,
        team_id: Optional[str] = None,
        page: int = 1,
        per_page: int = 50,
    ) -> VolumeListResponse:
        """List volumes in the caller's personal scope or in a team."""
        params: dict[str, object] = {"page": page, "per_page": per_page}
        resolved_team_id = _resolved_team_id(self._parent.client, team_id)
        if resolved_team_id:
            params["team_id"] = resolved_team_id
        response = await self._parent.client.request("GET", "/sandbox/volumes", params=params)
        return VolumeListResponse.model_validate(response)

    async def get(self, name: str, *, team_id: Optional[str] = None) -> Volume:
        """Get a volume by name in the caller's scope, or by ref.

        A ``<namespace>/<name>`` ref to another account's PUBLIC volume
        returns its summary only (name, namespace, size, visibility, head
        size, created_at): that response has no volume id, owner fields, head
        checkpoint or account storage. A ref to a missing or PRIVATE foreign
        volume raises APIError 404.
        """
        response = await self._parent.client.request(
            "GET",
            _volume_path(name),
            params=_name_params(self._parent.client, team_id, name),
        )
        return Volume.model_validate(response)

    async def delete(self, name: str, *, team_id: Optional[str] = None) -> None:
        """Delete a volume and its data; any team member can delete a team
        volume.

        DELETE is retried on transient 5xx, so a 503 while the server is
        still removing data can succeed on a later attempt. A ref to a
        volume owned by another account raises APIError 404.
        """
        await self._parent.client.request(
            "DELETE",
            _volume_path(name),
            params=_name_params(self._parent.client, team_id, name),
        )

    async def grow(self, name: str, size_gb: int, *, team_id: Optional[str] = None) -> Volume:
        """Grow a volume's capacity to ``size_gb`` whole GiB.

        Grow-only: the server rejects a size at or below the volume's
        current effective size (its set size, or the 100 GiB base image
        size when unset). The new size applies to sandboxes created after
        the change; running attachments keep their current size. A ref to a
        volume owned by another account raises APIError 404.
        """
        response = await self._parent.client.request(
            "PATCH",
            _volume_path(name),
            json={"size_gb": size_gb},
            params=_name_params(self._parent.client, team_id, name),
        )
        return Volume.model_validate(response)

    async def set_visibility(
        self, name: str, visibility: VolumeVisibility, *, team_id: Optional[str] = None
    ) -> Volume:
        """Make a volume PUBLIC or PRIVATE.

        A PUBLIC volume is mountable and syncable by any account through its
        ``<namespace>/<name>`` ref. Making it PRIVATE stops new recipient
        mounts, syncs and gets; running recipient sandboxes keep their mount.
        Only the volume's owner can change it: its user for a personal
        volume, any member of its team for a team volume. Repeating the
        current value is a no-op.
        """
        response = await self._parent.client.request(
            "PATCH",
            _volume_path(name),
            json={"visibility": visibility.value},
            params=_name_params(self._parent.client, team_id, name),
        )
        return Volume.model_validate(response)

    async def publish(self, sandbox_id: str, name: str, *, wait: bool = True) -> SandboxCheckpoint:
        """Publish one of a running sandbox's volume attachments.

        ``wait`` is keyword-only so an old third positional argument (the
        removed ``force`` flag) raises instead of silently meaning
        ``wait=False``.
        The server rejects the publish unless the sandbox is attached to a
        writable volume with that name or ref (404), makes the volume
        read-only (400), or has another sandbox's publish in flight (409).
        Publishing a volume owned by another account is rejected: 403 when
        this sandbox has it attached, 404 otherwise. Only its owner can move
        the shared head. With
        wait=True (the default), the returned checkpoint is the published
        head once DURABLE; on servers that report volume-merge status the
        wait also covers the checkpoint's merge, and a failed merge raises
        RuntimeError with merge_error. A failed publish raises RuntimeError
        and an unfinished publish TimeoutError.
        """
        response = await self._parent.client.request(
            "POST",
            _publish_path(sandbox_id, name),
            json={},
        )
        checkpoint = SandboxCheckpoint.model_validate(response)
        if not wait:
            return checkpoint
        return await self._parent.wait_for_checkpoint(checkpoint.id)

    async def sync(
        self,
        sandbox_id: str,
        name: str,
        discard: bool = False,
        wait: bool = True,
        timeout_seconds: float = SYNC_WAIT_TIMEOUT_SECONDS,
    ) -> VolumeSync:
        """Sync one of a running sandbox's volume attachments to the volume's
        committed head, re-attaching the volume at that head.

        Read-only and discard syncs re-attach without publishing; a read-write
        sync publishes this sandbox's unpublished writes first and waits for
        their merge. A volume owned by another account always syncs as
        discard, whatever ``discard`` says: this sandbox's unpublished writes
        are replaced by the volume's current head, and the stored sync
        records that. With wait=True (the default), the returned sync is the
        final state once DONE; a FAILED sync raises VolumeSyncFailedError and
        an unfinished sync TimeoutError. The wait polls at a fixed fast
        cadence for its first seconds and then with exponential backoff,
        sharing the checkpoint waiter's pacing.
        """
        _validate_sync_timeout(timeout_seconds)
        response = await self._parent.client.request(
            "POST",
            _sync_path(sandbox_id, name),
            json={"discard": discard},
        )
        sync = VolumeSync.model_validate(response)
        if not wait:
            return sync
        now = time.monotonic()
        deadline = now + timeout_seconds
        phase_start = now
        slow_index = 0
        while sync.state not in ("DONE", "FAILED"):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(_sync_timeout_message(sandbox_id, name, timeout_seconds, sync))
            if time.monotonic() - phase_start >= CHECKPOINT_POLL_FAST_WINDOW:
                await asyncio.sleep(min(_checkpoint_poll_delay(slow_index), remaining))
                slow_index += 1
            else:
                await asyncio.sleep(
                    min(_jittered_poll_delay(CHECKPOINT_POLL_FAST_DELAY), remaining)
                )
            response = await self._parent.client.request("GET", _sync_path(sandbox_id, name))
            polled = VolumeSync.model_validate(response)
            _require_pinned_sync(sandbox_id, name, sync, polled)
            sync = polled
        if sync.state == "FAILED":
            raise VolumeSyncFailedError(sandbox_id, name, sync.failure)
        return sync
