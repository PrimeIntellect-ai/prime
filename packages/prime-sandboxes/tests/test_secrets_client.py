import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from prime_sandboxes import (
    APIClient,
    AsyncSandboxClient,
    PlatformSecret,
    SandboxClient,
)


class DummyAPIClient(APIClient):
    def __init__(
        self,
        response: dict[str, Any],
        captured: dict[str, Any] | None = None,
        *,
        team_id: str | None = None,
    ) -> None:
        self.response = response
        self.captured = captured
        self.config = SimpleNamespace(
            team_id=team_id,
            config_dir=Path(tempfile.gettempdir()) / "prime-secrets-tests",
        )

    def request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if self.captured is not None:
            self.captured["method"] = method
            self.captured["path"] = path
            self.captured["json"] = json
            if params is not None:
                self.captured["params"] = params
        return self.response


def _secrets(
    response: dict[str, Any], captured: dict[str, Any] | None = None, *, team_id: str | None = None
) -> Any:
    """A sync secrets namespace over a request-recording API client."""
    return SandboxClient(DummyAPIClient(response, captured, team_id=team_id)).secrets


def _secret_response(**overrides: Any) -> dict[str, Any]:
    response = {
        "id": "secret-id-1",
        "name": "MY_SECRET",
        "description": "Test secret",
        "isFile": False,
        "userId": "user-1",
        "teamId": "team-1",
        "createdAt": "2026-01-15T10:00:00Z",
        "updatedAt": "2026-01-15T10:00:00Z",
    }
    response.update(overrides)
    return response


def test_secret_namespace_is_cached_on_the_parent_client():
    client = SandboxClient(DummyAPIClient({"data": _secret_response()}))
    assert client.secrets is client.secrets
    assert client.secrets._parent is client


def test_secret_create_sends_name_value_and_explicit_team():
    captured: dict[str, Any] = {}
    secrets = _secrets({"data": _secret_response()}, captured, team_id="team-configured")

    secret = secrets.create("MY_SECRET", "secret-value", team_id="team-explicit")

    assert captured == {
        "method": "POST",
        "path": "/secrets/",
        "json": {"name": "MY_SECRET", "value": "secret-value", "teamId": "team-explicit"},
    }
    assert isinstance(secret, PlatformSecret)
    assert secret.id == "secret-id-1"
    assert secret.name == "MY_SECRET"
    assert secret.is_file is False
    assert secret.team_id == "team-1"
    assert secret.created_at.year == 2026


def test_secret_create_uses_configured_team_by_default():
    captured: dict[str, Any] = {}
    secrets = _secrets({"data": _secret_response()}, captured, team_id="team-configured")

    secrets.create("MY_SECRET", "secret-value")

    assert captured["json"] == {
        "name": "MY_SECRET",
        "value": "secret-value",
        "teamId": "team-configured",
    }


def test_secret_create_personal_scope_omits_team_id():
    captured: dict[str, Any] = {}
    secrets = _secrets({"data": _secret_response()}, captured)

    secrets.create("MY_SECRET", "secret-value")

    assert captured["json"] == {"name": "MY_SECRET", "value": "secret-value"}


def test_secret_create_sends_description_and_file_flag():
    captured: dict[str, Any] = {}
    secrets = _secrets({"data": _secret_response()}, captured)

    secrets.create("MY_SECRET", "secret-value", description="My description", is_file=True)

    assert captured["json"] == {
        "name": "MY_SECRET",
        "value": "secret-value",
        "description": "My description",
        "isFile": True,
    }


def test_secret_list_uses_configured_team_by_default():
    captured: dict[str, Any] = {}
    secrets = _secrets({"data": [_secret_response()]}, captured, team_id="team-configured")

    secrets_list = secrets.list()

    assert captured["method"] == "GET"
    assert captured["path"] == "/secrets/"
    assert captured["params"] == {"teamId": "team-configured"}
    assert len(secrets_list) == 1
    assert isinstance(secrets_list[0], PlatformSecret)
    assert secrets_list[0].name == "MY_SECRET"


def test_secret_list_personal_scope_omits_team_id():
    captured: dict[str, Any] = {}
    secrets = _secrets({"data": []}, captured)

    secrets.list()

    assert captured["params"] == {}


def test_secret_get_uses_id_path_and_explicit_team():
    captured: dict[str, Any] = {}
    secrets = _secrets({"data": _secret_response()}, captured, team_id="team-configured")

    secret = secrets.get("secret-id-1", team_id="team-explicit")

    assert captured == {
        "method": "GET",
        "path": "/secrets/secret-id-1",
        "json": None,
        "params": {"teamId": "team-explicit"},
    }
    assert isinstance(secret, PlatformSecret)
    assert secret.id == "secret-id-1"


def test_secret_update_sends_only_provided_fields():
    captured: dict[str, Any] = {}
    secrets = _secrets({"data": _secret_response()}, captured, team_id="team-configured")

    secrets.update("secret-id-1", value="new-value", description="new desc")

    assert captured == {
        "method": "PATCH",
        "path": "/secrets/secret-id-1",
        "json": {"value": "new-value", "description": "new desc"},
        "params": {"teamId": "team-configured"},
    }


def test_secret_delete_uses_id_path_and_team_param():
    captured: dict[str, Any] = {}
    secrets = _secrets({}, captured, team_id="team-configured")

    result = secrets.delete("secret-id-1")

    assert result is None
    assert captured == {
        "method": "DELETE",
        "path": "/secrets/secret-id-1",
        "json": None,
        "params": {"teamId": "team-configured"},
    }


def test_async_secret_namespace_is_cached_on_the_parent_client():
    async def scenario() -> None:
        async with AsyncSandboxClient(api_key="test-key") as client:
            assert client.secrets is client.secrets
            assert client.secrets._parent is client

    asyncio.run(scenario())


def test_async_secret_client_wire_contract(monkeypatch: pytest.MonkeyPatch):
    # The real async client resolves team scope from the environment.
    monkeypatch.setenv("PRIME_TEAM_ID", "team-configured")
    captured: dict[str, Any] = {}

    async def scenario() -> None:
        client = AsyncSandboxClient(api_key="test-key")

        async def request(
            method: str,
            path: str,
            json: dict[str, Any] | None = None,
            params: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> dict[str, Any]:
            captured["method"] = method
            captured["path"] = path
            captured["json"] = json
            captured["params"] = params
            if method == "GET" and path == "/secrets/":
                return {"data": [_secret_response()]}
            return {"data": _secret_response()}

        client.client.request = request

        secret = await client.secrets.create("MY_SECRET", "secret-value")
        assert isinstance(secret, PlatformSecret)
        assert secret.name == "MY_SECRET"
        assert captured["method"] == "POST"
        assert captured["path"] == "/secrets/"
        assert captured["json"] == {
            "name": "MY_SECRET",
            "value": "secret-value",
            "teamId": "team-configured",
        }

        secrets_list = await client.secrets.list(team_id="team-explicit")
        assert len(secrets_list) == 1
        assert captured["params"] == {"teamId": "team-explicit"}

        got = await client.secrets.get("secret-id-1")
        assert isinstance(got, PlatformSecret)
        assert captured["path"] == "/secrets/secret-id-1"
        assert captured["params"] == {"teamId": "team-configured"}

        updated = await client.secrets.update("secret-id-1", value="new-value")
        assert isinstance(updated, PlatformSecret)
        assert captured["method"] == "PATCH"
        assert captured["json"] == {"value": "new-value"}

        assert await client.secrets.delete("secret-id-1") is None
        assert captured["method"] == "DELETE"

        await client.aclose()

    asyncio.run(scenario())


def test_async_secret_personal_scope_omits_team_id(monkeypatch: pytest.MonkeyPatch):
    # An explicitly empty PRIME_TEAM_ID selects the personal scope.
    monkeypatch.setenv("PRIME_TEAM_ID", "")
    captured: dict[str, Any] = {}

    async def scenario() -> None:
        client = AsyncSandboxClient(api_key="test-key")

        async def request(
            method: str,
            path: str,
            json: dict[str, Any] | None = None,
            params: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> dict[str, Any]:
            captured["json"] = json
            captured["params"] = params
            if method == "GET" and path == "/secrets/":
                return {"data": [_secret_response()]}
            return {"data": _secret_response()}

        client.client.request = request
        await client.secrets.create("MY_SECRET", "secret-value")
        assert captured["json"] == {"name": "MY_SECRET", "value": "secret-value"}
        await client.secrets.list()
        assert captured["params"] == {}
        await client.aclose()

    asyncio.run(scenario())
