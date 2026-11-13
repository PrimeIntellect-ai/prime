"""Sandbox volume operations, namespaced under the sandbox clients."""

import asyncio
import math
import time
from typing import TYPE_CHECKING, Optional
from urllib.parse import quote

from ._polling import (
    CHECKPOINT_POLL_FAST_DELAY,
    CHECKPOINT_POLL_FAST_WINDOW,
    _creation_poll_delay,
    _jittered_poll_delay,
)
from .exceptions import VolumeSyncFailedError
from .models import SandboxCheckpoint, Volume, VolumeListResponse, VolumeSync

if TYPE_CHECKING:
    from .core import APIClient, AsyncAPIClient
    from .sandbox import AsyncSandboxClient, SandboxClient


def _resolved_team_id(
    client: "APIClient | AsyncAPIClient", team_id: Optional[str]
) -> Optional[str]:
    """Explicit team wins; otherwise use the team from the client config."""
    return team_id if team_id is not None else client.config.team_id


def _volume_path(name: str) -> str:
    # Quote so a name that escapes the server's character rules cannot be
    # reinterpreted as a different route path; valid names stay unchanged.
    return f"/sandbox/volumes/{quote(name, safe='')}"


def _publish_path(sandbox_id: str, name: str) -> str:
    return f"/sandbox/{sandbox_id}/volumes/{quote(name, safe='')}/publish"


def _sync_path(sandbox_id: str, name: str) -> str:
    return f"/sandbox/{sandbox_id}/volumes/{quote(name, safe='')}/sync"


# The server fails a sync past its own 10-minute deadline (design S11), so
# the default wait budget covers the whole server-side lifecycle.
SYNC_WAIT_TIMEOUT_SECONDS = 600.0


def _validate_sync_timeout(timeout_seconds: float) -> None:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and greater than zero")


def _sync_timeout_message(
    sandbox_id: str, name: str, timeout_seconds: float, sync: VolumeSync
) -> str:
    return (
        f"Volume '{name}' in sandbox {sandbox_id} did not finish syncing "
        f"within {timeout_seconds:g}s (state={sync.state})"
    )


def _name_params(client: "APIClient | AsyncAPIClient", team_id: Optional[str]) -> dict[str, object]:
    resolved_team_id = _resolved_team_id(client, team_id)
    if resolved_team_id:
        return {"team_id": resolved_team_id}
    return {}


class Volumes:
    """Volume operations for one owning ``SandboxClient``.

    Reaches the platform through the parent client's transport and waits for
    publishes with the parent's checkpoint waiter; it owns no transport,
    waiter, batcher, or lifecycle of its own.
    """

    def __init__(self, parent: "SandboxClient"):
        self._parent = parent

    def create(
        self, name: str, *, team_id: Optional[str] = None, size_gb: Optional[int] = None
    ) -> Volume:
        """Create an empty volume in the caller's personal scope or in a team.

        ``size_gb`` is the requested capacity in whole GiB; omitted means
        the base image size (100 GiB).
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
        """Get a volume by name in the caller's scope."""
        response = self._parent.client.request(
            "GET",
            _volume_path(name),
            params=_name_params(self._parent.client, team_id),
        )
        return Volume.model_validate(response)

    def delete(self, name: str, *, team_id: Optional[str] = None) -> None:
        """Delete a volume and its data; deleting a team volume requires
        team admin.

        DELETE is retried on transient 5xx, so a 503 while the server is
        still removing data can succeed on a later attempt.
        """
        self._parent.client.request(
            "DELETE",
            _volume_path(name),
            params=_name_params(self._parent.client, team_id),
        )

    def grow(self, name: str, size_gb: int, *, team_id: Optional[str] = None) -> Volume:
        """Grow a volume's capacity to ``size_gb`` whole GiB.

        Grow-only: the server rejects a size at or below the volume's
        current effective size (its set size, or the 100 GiB base image
        size when unset). The new size applies to sandboxes created after
        the change; running attachments keep their current size.
        """
        response = self._parent.client.request(
            "PATCH",
            _volume_path(name),
            json={"size_gb": size_gb},
            params=_name_params(self._parent.client, team_id),
        )
        return Volume.model_validate(response)

    def publish(self, sandbox_id: str, name: str, *, wait: bool = True) -> SandboxCheckpoint:
        """Publish one of a running sandbox's volume attachments.

        ``wait`` is keyword-only so an old third positional argument (the
        removed ``force`` flag) raises instead of silently meaning
        ``wait=False``.
        The server rejects the publish unless the sandbox is attached to a
        writable volume with that name (404), makes the volume read-only
        (400), or has another sandbox's publish in flight (409). With
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
        their merge. With wait=True (the default), the returned sync is the
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
                raise TimeoutError(
                    _sync_timeout_message(sandbox_id, name, timeout_seconds, sync)
                )
            if time.monotonic() - phase_start >= CHECKPOINT_POLL_FAST_WINDOW:
                time.sleep(min(_creation_poll_delay(slow_index), remaining))
                slow_index += 1
            else:
                time.sleep(min(_jittered_poll_delay(CHECKPOINT_POLL_FAST_DELAY), remaining))
            response = self._parent.client.request("GET", _sync_path(sandbox_id, name))
            sync = VolumeSync.model_validate(response)
        if sync.state == "FAILED":
            raise VolumeSyncFailedError(sandbox_id, name)
        return sync


class AsyncVolumes:
    """Async volume operations for one owning ``AsyncSandboxClient``.

    Reaches the platform through the parent client's transport and waits for
    publishes with the parent's checkpoint waiter; it owns no transport,
    waiter, batcher, or lifecycle of its own.
    """

    def __init__(self, parent: "AsyncSandboxClient"):
        self._parent = parent

    async def create(
        self, name: str, *, team_id: Optional[str] = None, size_gb: Optional[int] = None
    ) -> Volume:
        """Create an empty volume in the caller's personal scope or in a team.

        ``size_gb`` is the requested capacity in whole GiB; omitted means
        the base image size (100 GiB).
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
        """Get a volume by name in the caller's scope."""
        response = await self._parent.client.request(
            "GET",
            _volume_path(name),
            params=_name_params(self._parent.client, team_id),
        )
        return Volume.model_validate(response)

    async def delete(self, name: str, *, team_id: Optional[str] = None) -> None:
        """Delete a volume and its data; deleting a team volume requires
        team admin.

        DELETE is retried on transient 5xx, so a 503 while the server is
        still removing data can succeed on a later attempt.
        """
        await self._parent.client.request(
            "DELETE",
            _volume_path(name),
            params=_name_params(self._parent.client, team_id),
        )

    async def grow(self, name: str, size_gb: int, *, team_id: Optional[str] = None) -> Volume:
        """Grow a volume's capacity to ``size_gb`` whole GiB.

        Grow-only: the server rejects a size at or below the volume's
        current effective size (its set size, or the 100 GiB base image
        size when unset). The new size applies to sandboxes created after
        the change; running attachments keep their current size.
        """
        response = await self._parent.client.request(
            "PATCH",
            _volume_path(name),
            json={"size_gb": size_gb},
            params=_name_params(self._parent.client, team_id),
        )
        return Volume.model_validate(response)

    async def publish(self, sandbox_id: str, name: str, *, wait: bool = True) -> SandboxCheckpoint:
        """Publish one of a running sandbox's volume attachments.

        ``wait`` is keyword-only so an old third positional argument (the
        removed ``force`` flag) raises instead of silently meaning
        ``wait=False``.
        The server rejects the publish unless the sandbox is attached to a
        writable volume with that name (404), makes the volume read-only
        (400), or has another sandbox's publish in flight (409). With
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
        their merge. With wait=True (the default), the returned sync is the
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
                raise TimeoutError(
                    _sync_timeout_message(sandbox_id, name, timeout_seconds, sync)
                )
            if time.monotonic() - phase_start >= CHECKPOINT_POLL_FAST_WINDOW:
                await asyncio.sleep(min(_creation_poll_delay(slow_index), remaining))
                slow_index += 1
            else:
                await asyncio.sleep(
                    min(_jittered_poll_delay(CHECKPOINT_POLL_FAST_DELAY), remaining)
                )
            response = await self._parent.client.request("GET", _sync_path(sandbox_id, name))
            sync = VolumeSync.model_validate(response)
        if sync.state == "FAILED":
            raise VolumeSyncFailedError(sandbox_id, name)
        return sync
