import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from prime_sandboxes import (
    APIClient,
    APIError,
    AsyncSandboxClient,
    CreateSandboxRequest,
    SandboxCheckpoint,
    SandboxClient,
    Volume,
    VolumeListResponse,
    VolumeMount,
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
            config_dir=Path(tempfile.gettempdir()) / "prime-volumes-tests",
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


def _volumes(
    response: dict[str, Any], captured: dict[str, Any] | None = None, *, team_id: str | None = None
) -> Any:
    """A sync volumes namespace over a request-recording API client."""
    return SandboxClient(DummyAPIClient(response, captured, team_id=team_id)).volumes


def _volume_response(**overrides: Any) -> dict[str, Any]:
    response = {
        "id": "cuid-1",
        "name": "dataset",
        "user_id": None,
        "team_id": "team-1",
        "created_at": "2026-01-01T00:00:00Z",
    }
    response.update(overrides)
    return response


def _volume_list_response() -> dict[str, Any]:
    return {
        "volumes": [_volume_response()],
        "total": 1,
        "page": 1,
        "per_page": 50,
        "has_next": False,
    }


def test_volume_namespace_is_cached_on_the_parent_client():
    client = SandboxClient(DummyAPIClient(_volume_response()))
    assert client.volumes is client.volumes
    assert client.volumes._parent is client


def test_volume_client_create_sends_name_and_explicit_team():
    captured: dict[str, Any] = {}
    volumes = _volumes(_volume_response(), captured, team_id="team-configured")

    volume = volumes.create("dataset", team_id="team-explicit")

    assert captured == {
        "method": "POST",
        "path": "/sandbox/volumes",
        "json": {"name": "dataset", "team_id": "team-explicit"},
    }
    assert isinstance(volume, Volume)
    assert volume.id == "cuid-1"
    assert volume.team_id == "team-1"
    assert volume.user_id is None
    assert volume.created_at.year == 2026


def test_volume_client_create_uses_configured_team_by_default():
    captured: dict[str, Any] = {}
    volumes = _volumes(_volume_response(), captured, team_id="team-configured")

    volumes.create("dataset")

    assert captured["json"] == {"name": "dataset", "team_id": "team-configured"}


def test_volume_client_create_personal_scope_omits_team_id():
    captured: dict[str, Any] = {}
    volumes = _volumes(_volume_response(), captured)

    volumes.create("dataset")

    assert captured["json"] == {"name": "dataset"}


def test_volume_client_list_forwards_pagination_and_explicit_team():
    captured: dict[str, Any] = {}
    response_body = _volume_list_response()
    volumes = _volumes(response_body, captured, team_id="team-configured")

    response = volumes.list(team_id="team-explicit", page=2, per_page=25)

    assert captured["method"] == "GET"
    assert captured["path"] == "/sandbox/volumes"
    assert captured["params"] == {
        "page": 2,
        "per_page": 25,
        "team_id": "team-explicit",
    }
    assert isinstance(response, VolumeListResponse)
    assert response.total == 1
    assert response.page == 1
    assert response.per_page == 50
    assert response.has_next is False
    assert isinstance(response.volumes[0], Volume)
    assert response.volumes[0].name == "dataset"


def test_volume_client_list_uses_configured_team_by_default():
    captured: dict[str, Any] = {}
    volumes = _volumes(_volume_list_response(), captured, team_id="team-configured")

    volumes.list()

    assert captured["params"] == {"page": 1, "per_page": 50, "team_id": "team-configured"}


def test_volume_client_list_personal_scope_omits_team_id():
    captured: dict[str, Any] = {}
    volumes = _volumes(_volume_list_response(), captured)

    volumes.list()

    assert captured["params"] == {"page": 1, "per_page": 50}


def test_volume_client_get_uses_name_path_and_explicit_team():
    captured: dict[str, Any] = {}
    volumes = _volumes(_volume_response(), captured, team_id="team-configured")

    volume = volumes.get("dataset", team_id="team-explicit")

    assert captured == {
        "method": "GET",
        "path": "/sandbox/volumes/dataset",
        "json": None,
        "params": {"team_id": "team-explicit"},
    }
    assert isinstance(volume, Volume)
    assert volume.name == "dataset"


def test_volume_client_get_uses_configured_team_by_default():
    captured: dict[str, Any] = {}
    volumes = _volumes(_volume_response(), captured, team_id="team-configured")

    volumes.get("dataset")

    assert captured["params"] == {"team_id": "team-configured"}


def test_volume_client_delete_returns_none_on_204():
    captured: dict[str, Any] = {}
    # The real client returns {} for a 204 response.
    volumes = _volumes({}, captured, team_id="team-configured")

    result = volumes.delete("dataset")

    assert result is None
    assert captured == {
        "method": "DELETE",
        "path": "/sandbox/volumes/dataset",
        "json": None,
        "params": {"team_id": "team-configured"},
    }


def test_volume_client_delete_personal_scope_omits_team_id():
    captured: dict[str, Any] = {}
    volumes = _volumes({}, captured)

    volumes.delete("dataset")

    assert captured["params"] == {}


def test_async_volume_namespace_is_cached_on_the_parent_client():
    async def scenario() -> None:
        async with AsyncSandboxClient(api_key="test-key") as client:
            assert client.volumes is client.volumes
            assert client.volumes._parent is client

    asyncio.run(scenario())


def test_async_volume_client_crud_wire_contract(monkeypatch: pytest.MonkeyPatch):
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
            if params is not None:
                captured["params"] = params
            return response

        response = _volume_response()
        client.client.request = request

        volume = await client.volumes.create("dataset", team_id="team-explicit")
        assert volume.id == "cuid-1"
        assert captured["json"] == {"name": "dataset", "team_id": "team-explicit"}

        response = _volume_list_response()
        page = await client.volumes.list(page=2, per_page=25)
        assert isinstance(page, VolumeListResponse)
        assert captured["params"] == {"page": 2, "per_page": 25, "team_id": "team-configured"}

        response = _volume_response()
        got = await client.volumes.get("dataset")
        assert isinstance(got, Volume)
        assert captured["path"] == "/sandbox/volumes/dataset"
        assert captured["params"] == {"team_id": "team-configured"}

        response = {}
        assert await client.volumes.delete("dataset") is None
        assert captured["method"] == "DELETE"
        assert captured["params"] == {"team_id": "team-configured"}

        await client.aclose()

    asyncio.run(scenario())


def test_async_volume_personal_scope_omits_team_id(monkeypatch: pytest.MonkeyPatch):
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
            return _volume_response()

        client.client.request = request
        await client.volumes.create("dataset")
        assert captured["json"] == {"name": "dataset"}
        await client.volumes.get("dataset")
        assert captured["params"] == {}
        await client.aclose()

    asyncio.run(scenario())


def test_volume_delete_swallows_real_204_response():
    from prime_sandboxes.core.client import APIClient as CoreAPIClient

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == "/api/v1/sandbox/volumes/dataset"
        assert request.url.params["team_id"] == "team-1"
        return httpx.Response(204, request=request)

    api_client = CoreAPIClient(api_key="test-key")
    api_client.client = httpx.Client(transport=httpx.MockTransport(handler))

    assert SandboxClient(api_client).volumes.delete("dataset", team_id="team-1") is None


def test_async_volume_delete_swallows_real_204_response():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == "/api/v1/sandbox/volumes/dataset"
        assert request.url.params["team_id"] == "team-1"
        return httpx.Response(204, request=request)

    sandbox_client = AsyncSandboxClient(api_key="test-key")
    sandbox_client.client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def scenario() -> None:
        assert await sandbox_client.volumes.delete("dataset", team_id="team-1") is None
        await sandbox_client.aclose()

    asyncio.run(scenario())


def test_volume_client_quotes_names_outside_the_valid_character_set():
    captured: dict[str, Any] = {}
    volumes = _volumes(_volume_response(), captured)

    volumes.get("a/b?c=d")

    assert captured["path"] == "/sandbox/volumes/a%2Fb%3Fc%3Dd"

    volumes.delete("a/b")

    assert captured["path"] == "/sandbox/volumes/a%2Fb"


def test_create_sandbox_request_omits_unset_volumes():
    request = CreateSandboxRequest(name="s", docker_image="ubuntu:22.04")

    payload = request.model_dump(by_alias=False, exclude_none=True)

    assert "volumes" not in payload


def test_create_sandbox_request_dumps_volume_mounts_in_snake_case():
    request = CreateSandboxRequest(
        name="s",
        docker_image="ubuntu:22.04",
        volumes=[
            VolumeMount(name="dataset", mount_path="/data"),
            VolumeMount(name="logs", mount_path="/logs", read_only=True),
        ],
    )

    payload = request.model_dump(by_alias=False, exclude_none=True)

    assert payload["volumes"] == [
        {"name": "dataset", "mount_path": "/data", "read_only": False},
        {"name": "logs", "mount_path": "/logs", "read_only": True},
    ]


def test_create_sandbox_request_accepts_explicit_empty_volumes():
    request = CreateSandboxRequest(name="s", docker_image="ubuntu:22.04", volumes=[])

    payload = request.model_dump(by_alias=False, exclude_none=True)

    assert payload["volumes"] == []


class RoutingAPIClient(APIClient):
    """APIClient whose responses come from a handler that records every call."""

    def __init__(self, handler: Any) -> None:
        super().__init__(api_key="test-key")
        self.handler = handler
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append((method, path, json))
        return self.handler(method, path, json)


def _checkpoint_response(state: str = "PENDING") -> dict[str, Any]:
    now = "2026-01-01T00:00:00Z"
    return {
        "id": "ckpt-1",
        "sandbox_id": "sb-1",
        "parent_id": None,
        "team_id": None,
        "state": state,
        "depth": 1,
        "docker_image": "python:3.11-slim",
        "created_at": now,
        "updated_at": now,
    }


def _publish_handler(states: list[str]) -> Any:
    """Publish returns PENDING, then batched checkpoint lookups return states."""
    responses = iter(states)

    def handler(method: str, path: str, json: dict[str, Any] | None) -> dict[str, Any]:
        if method == "POST" and path == "/sandbox/sb-1/volumes/dataset/publish":
            return _checkpoint_response()
        assert method == "POST" and path == "/sandbox/checkpoints:batchGet"
        return {"checkpoints": [_checkpoint_response(next(responses))], "errors": []}

    return handler


@pytest.fixture
def waiter_clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Fast-forward the checkpoint waiter's clocks so waits run instantly."""
    from prime_sandboxes import sandbox as module

    sleeps: list[float] = []
    now = [0.0]

    def sleep(delay: float) -> None:
        sleeps.append(delay)
        now[0] += delay

    async def async_sleep(delay: float) -> None:
        sleep(delay)

    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(module.time, "sleep", sleep)
    monkeypatch.setattr(module.asyncio, "sleep", async_sleep)
    monkeypatch.setattr(module.random, "uniform", lambda low, high: 0.0)
    return sleeps


def test_publish_sends_exact_post_path_and_body():
    client = RoutingAPIClient(lambda method, path, json: _checkpoint_response())
    volumes = SandboxClient(client).volumes  # type: ignore[arg-type]

    checkpoint = volumes.publish("sb-1", "dataset", wait=False)

    assert client.calls == [("POST", "/sandbox/sb-1/volumes/dataset/publish", {})]
    assert isinstance(checkpoint, SandboxCheckpoint)
    assert checkpoint.id == "ckpt-1"
    assert checkpoint.state == "PENDING"


def test_publish_quotes_name_in_path():
    client = RoutingAPIClient(lambda method, path, json: _checkpoint_response())
    volumes = SandboxClient(client).volumes  # type: ignore[arg-type]

    volumes.publish("sb-1", "a/b", wait=False)

    assert client.calls == [("POST", "/sandbox/sb-1/volumes/a%2Fb/publish", {})]


def test_publish_wait_true_polls_until_durable(waiter_clock):
    handler = _publish_handler(["PENDING", "FROZEN", "DURABLE"])
    client = RoutingAPIClient(handler)
    volumes = SandboxClient(client).volumes  # type: ignore[arg-type]

    checkpoint = volumes.publish("sb-1", "dataset")

    assert checkpoint.state == "DURABLE"
    poll_calls = client.calls[1:]
    assert len(poll_calls) == 3
    assert all(call[1] == "/sandbox/checkpoints:batchGet" for call in poll_calls)


def test_publish_wait_failed_raises_runtime_error(waiter_clock):
    handler = _publish_handler(["FAILED"])
    client = RoutingAPIClient(handler)
    volumes = SandboxClient(client).volumes  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="ckpt-1.*FAILED"):
        volumes.publish("sb-1", "dataset")


def test_publish_wait_timeout_raises_timeout_error(waiter_clock):
    handler = _publish_handler(["PENDING"] * 200)
    client = RoutingAPIClient(handler)
    volumes = SandboxClient(client).volumes  # type: ignore[arg-type]

    with pytest.raises(TimeoutError, match="ckpt-1.*did not become DURABLE"):
        volumes.publish("sb-1", "dataset")

    # The 200 queued responses exhaust the waiter's default 300s deadline.
    assert len(client.calls[1:]) < 200


def _async_publish_client(handler: Any) -> tuple[AsyncSandboxClient, list[tuple[str, str, Any]]]:
    """An async sandbox client whose API request method is a recording double."""
    client = AsyncSandboxClient(api_key="test-key")
    calls: list[tuple[str, str, Any]] = []

    async def request(
        method: str, path: str, json: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        calls.append((method, path, json))
        return handler(method, path, json)

    client.client.request = request
    return client, calls


def test_async_publish_sends_exact_post_path_and_body():
    client, calls = _async_publish_client(lambda method, path, json: _checkpoint_response())

    async def scenario() -> SandboxCheckpoint:
        async with client:
            return await client.volumes.publish("sb-1", "dataset", wait=False)

    checkpoint = asyncio.run(scenario())

    assert calls == [("POST", "/sandbox/sb-1/volumes/dataset/publish", {})]
    assert checkpoint.id == "ckpt-1"


def test_async_publish_wait_true_polls_until_durable(waiter_clock):
    client, calls = _async_publish_client(_publish_handler(["PENDING", "FROZEN", "DURABLE"]))

    async def scenario() -> SandboxCheckpoint:
        async with client:
            return await client.volumes.publish("sb-1", "dataset")

    checkpoint = asyncio.run(scenario())

    assert checkpoint.state == "DURABLE"
    assert len(calls) == 4


def test_async_publish_wait_failed_raises_runtime_error(waiter_clock):
    client, _ = _async_publish_client(_publish_handler(["FAILED"]))

    async def scenario() -> None:
        async with client:
            with pytest.raises(RuntimeError, match="ckpt-1.*FAILED"):
                await client.volumes.publish("sb-1", "dataset")

    asyncio.run(scenario())


def test_async_publish_wait_timeout_raises_timeout_error(waiter_clock):
    client, _ = _async_publish_client(_publish_handler(["PENDING"] * 200))

    async def scenario() -> None:
        async with client:
            with pytest.raises(TimeoutError, match="ckpt-1.*did not become DURABLE"):
                await client.volumes.publish("sb-1", "dataset")

    asyncio.run(scenario())


def test_publish_conflict_surfaces_api_error():
    from prime_sandboxes.core.client import APIClient as CoreAPIClient

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/v1/sandbox/sb-1/volumes/dataset/publish"
        return httpx.Response(409, json={"detail": "another publish is PENDING"}, request=request)

    api_client = CoreAPIClient(api_key="test-key")
    api_client.client = httpx.Client(transport=httpx.MockTransport(handler))

    with pytest.raises(APIError, match="409"):
        SandboxClient(api_client).volumes.publish("sb-1", "dataset")


def test_publish_batch_get_fallback_uses_get_checkpoints():
    from prime_sandboxes.core.client import APIClient as CoreAPIClient

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/publish"):
            return httpx.Response(202, json=_checkpoint_response(), request=request)
        if request.method == "POST" and request.url.path.endswith("checkpoints:batchGet"):
            return httpx.Response(404, json={"detail": "unsupported"}, request=request)
        assert request.method == "GET" and request.url.path == "/api/v1/sandbox/checkpoints/ckpt-1"
        return httpx.Response(200, json=_checkpoint_response("DURABLE"), request=request)

    api_client = CoreAPIClient(api_key="test-key")
    api_client.client = httpx.Client(transport=httpx.MockTransport(handler))

    checkpoint = SandboxClient(api_client).volumes.publish("sb-1", "dataset")

    assert checkpoint.state == "DURABLE"


def test_publish_wait_polls_until_merge_not_pending(waiter_clock):
    def handler(method: str, path: str, json: dict[str, Any] | None) -> dict[str, Any]:
        if method == "POST" and path == "/sandbox/sb-1/volumes/dataset/publish":
            return _checkpoint_response()
        assert method == "POST" and path == "/sandbox/checkpoints:batchGet"
        row = _checkpoint_response("DURABLE")
        row["merge"] = "pending" if len(client.calls) == 2 else "merged"
        return {"checkpoints": [row], "errors": []}

    client = RoutingAPIClient(handler)
    volumes = SandboxClient(client).volumes  # type: ignore[arg-type]

    checkpoint = volumes.publish("sb-1", "dataset")

    assert checkpoint.state == "DURABLE"
    assert checkpoint.merge == "merged"
    assert len(client.calls[1:]) == 2


def test_async_sandbox_client_closes_owned_api_client():
    async def scenario() -> None:
        sandbox_client = AsyncSandboxClient(api_key="test-key")
        await sandbox_client.aclose()
        assert sandbox_client.client.client.is_closed is True

    asyncio.run(scenario())


def test_volume_model_parses_head_checkpoint_id():
    volume = Volume.model_validate(_volume_response(head_checkpoint_id="ckpt-9"))
    assert volume.head_checkpoint_id == "ckpt-9"

    # Older responses omit the field entirely.
    assert Volume.model_validate(_volume_response()).head_checkpoint_id is None

    response = VolumeListResponse.model_validate(_volume_list_response())
    assert response.volumes[0].head_checkpoint_id is None


def test_volume_model_parses_storage_fields():
    volume = Volume.model_validate(
        _volume_response(
            occupied_bytes=5 * 1024**3,
            account_storage_used_bytes=6 * 1024**3,
            account_storage_limit_gb=1024,
        )
    )
    assert volume.occupied_bytes == 5 * 1024**3
    assert volume.account_storage_used_bytes == 6 * 1024**3
    assert volume.account_storage_limit_gb == 1024

    # Older responses omit the fields entirely.
    legacy = Volume.model_validate(_volume_response())
    assert legacy.occupied_bytes == 0
    assert legacy.account_storage_used_bytes is None
    assert legacy.account_storage_limit_gb is None


def test_volume_list_response_parses_account_storage_totals():
    response = VolumeListResponse.model_validate(
        _volume_list_response() | {"storage_used_bytes": 2048, "storage_limit_gb": 1024}
    )
    assert response.storage_used_bytes == 2048
    assert response.storage_limit_gb == 1024

    # Older responses omit the fields entirely.
    legacy = VolumeListResponse.model_validate(_volume_list_response())
    assert legacy.storage_used_bytes is None
    assert legacy.storage_limit_gb is None


def test_publish_response_parses_occupied_bytes():
    client = RoutingAPIClient(
        lambda method, path, json: _checkpoint_response() | {"occupied_bytes": 4096}
    )
    volumes = SandboxClient(client).volumes  # type: ignore[arg-type]

    checkpoint = volumes.publish("sb-1", "dataset", wait=False)

    assert checkpoint.occupied_bytes == 4096
    # Older responses omit the field entirely.
    assert SandboxCheckpoint.model_validate(_checkpoint_response()).occupied_bytes == 0
