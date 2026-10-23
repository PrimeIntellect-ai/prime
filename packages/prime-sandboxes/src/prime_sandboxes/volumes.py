"""Sandbox volume operations, namespaced under the sandbox clients."""

from typing import TYPE_CHECKING, Optional
from urllib.parse import quote

from .models import SandboxCheckpoint, Volume, VolumeListResponse

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


def _name_params(
    client: "APIClient | AsyncAPIClient", team_id: Optional[str]
) -> dict[str, object]:
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

    def create(self, name: str, *, team_id: Optional[str] = None) -> Volume:
        """Create an empty volume in the caller's personal scope or in a team."""
        payload: dict[str, object] = {"name": name}
        resolved_team_id = _resolved_team_id(self._parent.client, team_id)
        if resolved_team_id:
            payload["team_id"] = resolved_team_id
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

    def publish(self, sandbox_id: str, name: str, wait: bool = True) -> SandboxCheckpoint:
        """Publish one of a running sandbox's volume attachments.

        The server rejects the publish unless the sandbox is attached to a
        writable volume with that name (404), makes the volume read-only
        (400), or has another sandbox's publish in flight (409). With
        wait=True (the default), the returned checkpoint is the published
        head once DURABLE; a failed publish raises RuntimeError and an
        unfinished publish TimeoutError.
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


class AsyncVolumes:
    """Async volume operations for one owning ``AsyncSandboxClient``.

    Reaches the platform through the parent client's transport and waits for
    publishes with the parent's checkpoint waiter; it owns no transport,
    waiter, batcher, or lifecycle of its own.
    """

    def __init__(self, parent: "AsyncSandboxClient"):
        self._parent = parent

    async def create(self, name: str, *, team_id: Optional[str] = None) -> Volume:
        """Create an empty volume in the caller's personal scope or in a team."""
        payload: dict[str, object] = {"name": name}
        resolved_team_id = _resolved_team_id(self._parent.client, team_id)
        if resolved_team_id:
            payload["team_id"] = resolved_team_id
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

    async def publish(self, sandbox_id: str, name: str, wait: bool = True) -> SandboxCheckpoint:
        """Publish one of a running sandbox's volume attachments.

        The server rejects the publish unless the sandbox is attached to a
        writable volume with that name (404), makes the volume read-only
        (400), or has another sandbox's publish in flight (409). With
        wait=True (the default), the returned checkpoint is the published
        head once DURABLE; a failed publish raises RuntimeError and an
        unfinished publish TimeoutError.
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
