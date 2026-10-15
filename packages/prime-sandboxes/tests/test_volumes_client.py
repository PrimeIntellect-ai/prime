import asyncio
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from prime_sandboxes import (
    APIClient,
    APIError,
    AsyncAPIClient,
    AsyncSandboxClient,
    AsyncVolumesClient,
    CreateSandboxRequest,
    SandboxCheckpoint,
    Volume,
    VolumeListResponse,
    VolumeMount,
    VolumesClient,
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
        self.config = SimpleNamespace(team_id=team_id)

    def request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.captured is not None:
            self.captured["method"] = method
            self.captured["path"] = path
            self.captured["json"] = json
            if params is not None:
                self.captured["params"] = params
        return self.response


class DummyAsyncAPIClient:
    def __init__(
        self,
        response: dict[str, Any],
        captured: dict[str, Any] | None = None,
        *,
        team_id: str | None = None,
    ) -> None:
        self.response = response
        self.captured = captured
        self.config = SimpleNamespace(team_id=team_id)

    async def request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.captured is not None:
            self.captured["method"] = method
            self.captured["path"] = path
            self.captured["json"] = json
            if params is not None:
                self.captured["params"] = params
        return self.response


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


def test_volume_client_create_sends_name_and_explicit_team():
    captured: dict[str, Any] = {}
    client = VolumesClient(DummyAPIClient(_volume_response(), captured, team_id="team-configured"))

    volume = client.create("dataset", team_id="team-explicit")

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
    client = VolumesClient(DummyAPIClient(_volume_response(), captured, team_id="team-configured"))

    client.create("dataset")

    assert captured["json"] == {"name": "dataset", "team_id": "team-configured"}


def test_volume_client_create_personal_scope_omits_team_id():
    captured: dict[str, Any] = {}
    client = VolumesClient(DummyAPIClient(_volume_response(), captured))

    client.create("dataset")

    assert captured["json"] == {"name": "dataset"}


def test_volume_client_list_forwards_pagination_and_explicit_team():
    captured: dict[str, Any] = {}
    response_body = _volume_list_response()
    client = VolumesClient(DummyAPIClient(response_body, captured, team_id="team-configured"))

    response = client.list(team_id="team-explicit", page=2, per_page=25)

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
    client = VolumesClient(
        DummyAPIClient(_volume_list_response(), captured, team_id="team-configured")
    )

    client.list()

    assert captured["params"] == {"page": 1, "per_page": 50, "team_id": "team-configured"}


def test_volume_client_list_personal_scope_omits_team_id():
    captured: dict[str, Any] = {}
    client = VolumesClient(DummyAPIClient(_volume_list_response(), captured))

    client.list()

    assert captured["params"] == {"page": 1, "per_page": 50}


def test_volume_client_get_uses_name_path_and_explicit_team():
    captured: dict[str, Any] = {}
    client = VolumesClient(DummyAPIClient(_volume_response(), captured, team_id="team-configured"))

    volume = client.get("dataset", team_id="team-explicit")

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
    client = VolumesClient(DummyAPIClient(_volume_response(), captured, team_id="team-configured"))

    client.get("dataset")

    assert captured["params"] == {"team_id": "team-configured"}


def test_volume_client_delete_returns_none_on_204():
    captured: dict[str, Any] = {}
    # The real client returns {} for a 204 response.
    client = VolumesClient(DummyAPIClient({}, captured, team_id="team-configured"))

    result = client.delete("dataset")

    assert result is None
    assert captured == {
        "method": "DELETE",
        "path": "/sandbox/volumes/dataset",
        "json": None,
        "params": {"team_id": "team-configured"},
    }


def test_volume_client_delete_personal_scope_omits_team_id():
    captured: dict[str, Any] = {}
    client = VolumesClient(DummyAPIClient({}, captured))

    client.delete("dataset")

    assert captured["params"] == {}


def test_async_volume_client_crud_wire_contract():
    captured: dict[str, Any] = {}

    def make_client(response: dict[str, Any]) -> AsyncVolumesClient:
        return AsyncVolumesClient(  # type: ignore[arg-type]
            DummyAsyncAPIClient(response, captured, team_id="team-configured")
        )

    volume = asyncio.run(make_client(_volume_response()).create("dataset", team_id="team-explicit"))
    assert volume.id == "cuid-1"

    page = asyncio.run(make_client(_volume_list_response()).list(page=2, per_page=25))
    assert isinstance(page, VolumeListResponse)
    assert captured["params"] == {
        "page": 2,
        "per_page": 25,
        "team_id": "team-configured",
    }

    got = asyncio.run(make_client(_volume_response()).get("dataset"))
    assert isinstance(got, Volume)
    assert captured["path"] == "/sandbox/volumes/dataset"
    assert captured["params"] == {"team_id": "team-configured"}

    assert asyncio.run(make_client({}).delete("dataset")) is None
    assert captured["method"] == "DELETE"
    assert captured["params"] == {"team_id": "team-configured"}


def test_volume_delete_swallows_real_204_response():
    from prime_sandboxes.core.client import APIClient as CoreAPIClient

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == "/api/v1/sandbox/volumes/dataset"
        assert request.url.params["team_id"] == "team-1"
        return httpx.Response(204, request=request)

    api_client = CoreAPIClient(api_key="test-key")
    api_client.client = httpx.Client(transport=httpx.MockTransport(handler))

    assert VolumesClient(api_client).delete("dataset", team_id="team-1") is None


def test_async_volume_delete_swallows_real_204_response():
    from prime_sandboxes.core.client import AsyncAPIClient as CoreAsyncAPIClient

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == "/api/v1/sandbox/volumes/dataset"
        assert request.url.params["team_id"] == "team-1"
        return httpx.Response(204, request=request)

    api_client = CoreAsyncAPIClient(api_key="test-key")
    api_client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    assert asyncio.run(AsyncVolumesClient(api_client).delete("dataset", team_id="team-1")) is None


def test_volume_client_quotes_names_outside_the_valid_character_set():
    captured: dict[str, Any] = {}
    client = VolumesClient(DummyAPIClient(_volume_response(), captured))

    client.get("a/b?c=d")

    assert captured["path"] == "/sandbox/volumes/a%2Fb%3Fc%3Dd"

    client.delete("a/b")

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


class RoutingAsyncAPIClient(AsyncAPIClient):
    def __init__(self, handler: Any) -> None:
        super().__init__(api_key="test-key")
        self.handler = handler
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    async def request(
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
    volumes = VolumesClient(client)  # type: ignore[arg-type]

    checkpoint = volumes.publish("sb-1", "dataset", wait=False)

    assert client.calls == [("POST", "/sandbox/sb-1/volumes/dataset/publish", {"force": False})]
    assert isinstance(checkpoint, SandboxCheckpoint)
    assert checkpoint.id == "ckpt-1"
    assert checkpoint.state == "PENDING"


def test_publish_force_true_quotes_name_in_path():
    client = RoutingAPIClient(lambda method, path, json: _checkpoint_response())
    volumes = VolumesClient(client)  # type: ignore[arg-type]

    volumes.publish("sb-1", "a/b", force=True, wait=False)

    assert client.calls == [("POST", "/sandbox/sb-1/volumes/a%2Fb/publish", {"force": True})]


def test_publish_wait_true_polls_until_durable(waiter_clock):
    handler = _publish_handler(["PENDING", "FROZEN", "DURABLE"])
    client = RoutingAPIClient(handler)
    volumes = VolumesClient(client)  # type: ignore[arg-type]

    checkpoint = volumes.publish("sb-1", "dataset")

    assert checkpoint.state == "DURABLE"
    poll_calls = client.calls[1:]
    assert len(poll_calls) == 3
    assert all(call[1] == "/sandbox/checkpoints:batchGet" for call in poll_calls)


def test_publish_wait_failed_raises_runtime_error(waiter_clock):
    handler = _publish_handler(["FAILED"])
    client = RoutingAPIClient(handler)
    volumes = VolumesClient(client)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="ckpt-1.*FAILED"):
        volumes.publish("sb-1", "dataset")


def test_publish_wait_timeout_raises_timeout_error(waiter_clock):
    handler = _publish_handler(["PENDING"] * 200)
    client = RoutingAPIClient(handler)
    volumes = VolumesClient(client)  # type: ignore[arg-type]

    with pytest.raises(TimeoutError, match="ckpt-1.*did not become DURABLE"):
        volumes.publish("sb-1", "dataset")

    # The 200 queued responses exhaust the waiter's default 300s deadline.
    assert len(client.calls[1:]) < 200


def test_async_publish_sends_exact_post_path_and_body():
    client = RoutingAsyncAPIClient(lambda method, path, json: _checkpoint_response())

    checkpoint = asyncio.run(
        AsyncVolumesClient(client).publish("sb-1", "dataset", force=True, wait=False)  # type: ignore[arg-type]
    )

    assert client.calls == [("POST", "/sandbox/sb-1/volumes/dataset/publish", {"force": True})]
    assert checkpoint.id == "ckpt-1"


def test_async_publish_wait_true_polls_until_durable(waiter_clock):
    client = RoutingAsyncAPIClient(_publish_handler(["PENDING", "FROZEN", "DURABLE"]))
    volumes = AsyncVolumesClient(client)  # type: ignore[arg-type]

    async def scenario() -> SandboxCheckpoint:
        return await volumes.publish("sb-1", "dataset")

    checkpoint = asyncio.run(scenario())

    assert checkpoint.state == "DURABLE"
    assert len(client.calls) == 4


def test_async_publish_wait_failed_raises_runtime_error(waiter_clock):
    client = RoutingAsyncAPIClient(_publish_handler(["FAILED"]))
    volumes = AsyncVolumesClient(client)  # type: ignore[arg-type]

    async def scenario() -> None:
        with pytest.raises(RuntimeError, match="ckpt-1.*FAILED"):
            await volumes.publish("sb-1", "dataset")

    asyncio.run(scenario())


def test_async_publish_wait_timeout_raises_timeout_error(waiter_clock):
    client = RoutingAsyncAPIClient(_publish_handler(["PENDING"] * 200))
    volumes = AsyncVolumesClient(client)  # type: ignore[arg-type]

    async def scenario() -> None:
        with pytest.raises(TimeoutError, match="ckpt-1.*did not become DURABLE"):
            await volumes.publish("sb-1", "dataset")

    asyncio.run(scenario())


def test_publish_conflict_surfaces_api_error_even_with_force():
    from prime_sandboxes.core.client import APIClient as CoreAPIClient

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/v1/sandbox/sb-1/volumes/dataset/publish"
        return httpx.Response(409, json={"detail": "another publish is PENDING"}, request=request)

    api_client = CoreAPIClient(api_key="test-key")
    api_client.client = httpx.Client(transport=httpx.MockTransport(handler))

    with pytest.raises(APIError, match="409"):
        VolumesClient(api_client).publish("sb-1", "dataset", force=True)


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

    checkpoint = VolumesClient(api_client).publish("sb-1", "dataset")

    assert checkpoint.state == "DURABLE"


def test_async_sandbox_client_does_not_close_borrowed_api_client():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={}, request=request)

    borrowed = AsyncAPIClient(api_key="test-key")
    borrowed.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def scenario() -> None:
        sandbox_client = AsyncSandboxClient(api_client=borrowed)
        assert sandbox_client._owns_client is False
        await sandbox_client.aclose()
        assert borrowed.client.is_closed is False

    asyncio.run(scenario())


def test_async_sandbox_client_closes_owned_api_client():
    async def scenario() -> None:
        sandbox_client = AsyncSandboxClient(api_key="test-key")
        assert sandbox_client._owns_client is True
        await sandbox_client.aclose()
        assert sandbox_client.client.client.is_closed is True

    asyncio.run(scenario())


def test_async_volume_client_aclose_closes_waiter_before_transport(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/publish"):
            return httpx.Response(202, json=_checkpoint_response(), request=request)
        return httpx.Response(
            200,
            json={"checkpoints": [_checkpoint_response("DURABLE")], "errors": []},
            request=request,
        )

    api_client = AsyncAPIClient(api_key="test-key")
    api_client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    volumes = AsyncVolumesClient(api_client)
    order: list[str] = []

    async def scenario() -> None:
        checkpoint = await volumes.publish("sb-1", "dataset")
        assert checkpoint.state == "DURABLE"
        waiter = volumes._sandbox_client
        assert waiter is not None
        original_waiter_close = waiter.aclose
        original_transport_close = api_client.aclose

        async def waiter_close() -> None:
            order.append("waiter")
            await original_waiter_close()

        async def transport_close() -> None:
            order.append("transport")
            await original_transport_close()

        monkeypatch.setattr(waiter, "aclose", waiter_close)
        monkeypatch.setattr(api_client, "aclose", transport_close)
        await volumes.aclose()

    asyncio.run(scenario())
    assert order == ["waiter", "transport"]
    assert api_client.client.is_closed is True


def test_async_publish_repeated_waits_reuse_one_waiter():
    def handler(method: str, path: str, json: dict[str, Any] | None) -> dict[str, Any]:
        if method == "POST" and path == "/sandbox/sb-1/volumes/dataset/publish":
            return _checkpoint_response()
        return {"checkpoints": [_checkpoint_response("DURABLE")], "errors": []}

    client = RoutingAsyncAPIClient(handler)
    volumes = AsyncVolumesClient(client)  # type: ignore[arg-type]

    async def scenario() -> SandboxCheckpoint:
        checkpoint = await volumes.publish("sb-1", "dataset")
        waiter = volumes._sandbox_client
        assert waiter is not None and waiter.client is client
        second = await volumes.publish("sb-1", "dataset", force=True)
        assert volumes._sandbox_client is waiter
        assert checkpoint.id == second.id
        return second

    checkpoint = asyncio.run(scenario())
    assert checkpoint.state == "DURABLE"


def test_volume_model_parses_head_checkpoint_id():
    volume = Volume.model_validate(_volume_response(head_checkpoint_id="ckpt-9"))
    assert volume.head_checkpoint_id == "ckpt-9"

    # Older responses omit the field entirely.
    assert Volume.model_validate(_volume_response()).head_checkpoint_id is None

    response = VolumeListResponse.model_validate(_volume_list_response())
    assert response.volumes[0].head_checkpoint_id is None
