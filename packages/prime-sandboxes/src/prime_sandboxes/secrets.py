"""Platform secret operations, namespaced under the sandbox clients."""

from typing import TYPE_CHECKING, List, Optional

from .models import PlatformSecret
from .volumes import _resolved_team_id

if TYPE_CHECKING:
    from .core import APIClient, AsyncAPIClient
    from .sandbox import AsyncSandboxClient, SandboxClient


def _team_params(client: "APIClient | AsyncAPIClient", team_id: Optional[str]) -> dict[str, object]:
    resolved_team_id = _resolved_team_id(client, team_id)
    return {"teamId": resolved_team_id} if resolved_team_id else {}


class Secrets:
    """Platform secret operations for one owning ``SandboxClient``.

    Secret scope follows the caller's configured team, matching sandbox
    create: an explicit ``team_id`` wins, otherwise the team from the client
    config (``PRIME_TEAM_ID`` or the config file) is used, and with no team
    configured the secret is personal-scope. Reaches the platform through the
    parent client's transport; it owns no transport of its own.
    """

    def __init__(self, parent: "SandboxClient"):
        self._parent = parent

    def create(
        self,
        name: str,
        value: str,
        *,
        description: Optional[str] = None,
        is_file: bool = False,
        team_id: Optional[str] = None,
    ) -> PlatformSecret:
        """Create a platform secret in the caller's configured team scope.

        A header rule can then reference it with ``Secret.fill(name)``; a
        sandbox created in the same team scope resolves the reference.
        ``is_file`` treats ``value`` as file content (base64 encoded).
        """
        payload: dict[str, object] = {"name": name, "value": value}
        resolved_team_id = _resolved_team_id(self._parent.client, team_id)
        if resolved_team_id:
            payload["teamId"] = resolved_team_id
        if description:
            payload["description"] = description
        if is_file:
            payload["isFile"] = True
        response = self._parent.client.request("POST", "/secrets/", json=payload)
        return PlatformSecret.model_validate(response["data"])

    def list(self, *, team_id: Optional[str] = None) -> List[PlatformSecret]:
        """List the caller's personal secrets or a team's."""
        response = self._parent.client.request(
            "GET",
            "/secrets/",
            params=_team_params(self._parent.client, team_id),
        )
        return [PlatformSecret.model_validate(item) for item in response.get("data", [])]

    def get(self, secret_id: str, *, team_id: Optional[str] = None) -> PlatformSecret:
        """Get a secret's metadata; values never leave the server."""
        response = self._parent.client.request(
            "GET",
            f"/secrets/{secret_id}",
            params=_team_params(self._parent.client, team_id),
        )
        return PlatformSecret.model_validate(response["data"])

    def update(
        self,
        secret_id: str,
        *,
        name: Optional[str] = None,
        value: Optional[str] = None,
        description: Optional[str] = None,
        team_id: Optional[str] = None,
    ) -> PlatformSecret:
        """Update a secret; only the provided fields change."""
        payload: dict[str, object] = {}
        if name is not None:
            payload["name"] = name
        if value is not None:
            payload["value"] = value
        if description is not None:
            payload["description"] = description
        response = self._parent.client.request(
            "PATCH",
            f"/secrets/{secret_id}",
            json=payload,
            params=_team_params(self._parent.client, team_id),
        )
        return PlatformSecret.model_validate(response["data"])

    def delete(self, secret_id: str, *, team_id: Optional[str] = None) -> None:
        """Delete a secret in the caller's scope."""
        self._parent.client.request(
            "DELETE",
            f"/secrets/{secret_id}",
            params=_team_params(self._parent.client, team_id),
        )


class AsyncSecrets:
    """Async platform secret operations for one owning ``AsyncSandboxClient``.

    Secret scope follows the caller's configured team, matching sandbox
    create: an explicit ``team_id`` wins, otherwise the team from the client
    config (``PRIME_TEAM_ID`` or the config file) is used, and with no team
    configured the secret is personal-scope. Reaches the platform through the
    parent client's transport; it owns no transport of its own.
    """

    def __init__(self, parent: "AsyncSandboxClient"):
        self._parent = parent

    async def create(
        self,
        name: str,
        value: str,
        *,
        description: Optional[str] = None,
        is_file: bool = False,
        team_id: Optional[str] = None,
    ) -> PlatformSecret:
        """Create a platform secret in the caller's configured team scope.

        A header rule can then reference it with ``Secret.fill(name)``; a
        sandbox created in the same team scope resolves the reference.
        ``is_file`` treats ``value`` as file content (base64 encoded).
        """
        payload: dict[str, object] = {"name": name, "value": value}
        resolved_team_id = _resolved_team_id(self._parent.client, team_id)
        if resolved_team_id:
            payload["teamId"] = resolved_team_id
        if description:
            payload["description"] = description
        if is_file:
            payload["isFile"] = True
        response = await self._parent.client.request("POST", "/secrets/", json=payload)
        return PlatformSecret.model_validate(response["data"])

    async def list(self, *, team_id: Optional[str] = None) -> List[PlatformSecret]:
        """List the caller's personal secrets or a team's."""
        response = await self._parent.client.request(
            "GET",
            "/secrets/",
            params=_team_params(self._parent.client, team_id),
        )
        return [PlatformSecret.model_validate(item) for item in response.get("data", [])]

    async def get(self, secret_id: str, *, team_id: Optional[str] = None) -> PlatformSecret:
        """Get a secret's metadata; values never leave the server."""
        response = await self._parent.client.request(
            "GET",
            f"/secrets/{secret_id}",
            params=_team_params(self._parent.client, team_id),
        )
        return PlatformSecret.model_validate(response["data"])

    async def update(
        self,
        secret_id: str,
        *,
        name: Optional[str] = None,
        value: Optional[str] = None,
        description: Optional[str] = None,
        team_id: Optional[str] = None,
    ) -> PlatformSecret:
        """Update a secret; only the provided fields change."""
        payload: dict[str, object] = {}
        if name is not None:
            payload["name"] = name
        if value is not None:
            payload["value"] = value
        if description is not None:
            payload["description"] = description
        response = await self._parent.client.request(
            "PATCH",
            f"/secrets/{secret_id}",
            json=payload,
            params=_team_params(self._parent.client, team_id),
        )
        return PlatformSecret.model_validate(response["data"])

    async def delete(self, secret_id: str, *, team_id: Optional[str] = None) -> None:
        """Delete a secret in the caller's scope."""
        await self._parent.client.request(
            "DELETE",
            f"/secrets/{secret_id}",
            params=_team_params(self._parent.client, team_id),
        )
