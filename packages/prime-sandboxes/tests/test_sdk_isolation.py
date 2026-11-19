"""Explicit SDK configuration and client-local auth-cache isolation."""

import asyncio
import inspect
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

import prime_sandboxes.core.config as config_module
import prime_sandboxes.sandbox as sandbox_module
from prime_sandboxes import (
    APIClient,
    AsyncAPIClient,
    AsyncSandboxClient,
    Config,
    CreateSandboxRequest,
    SandboxClient,
)


@pytest.fixture
def hostile_ambient(monkeypatch, tmp_path):
    prime_dir = tmp_path / ".prime"
    (prime_dir / "environments").mkdir(parents=True)
    profile = {
        "api_key": "profile-key",
        "base_url": "https://profile.example",
        "team_id": "profile-team",
        "user_id": "profile-user",
    }
    (prime_dir / "config.json").write_text(json.dumps(profile))
    (prime_dir / "environments" / "hostile.json").write_text(json.dumps(profile))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for name, value in {
        "PRIME_CONTEXT": "hostile",
        "PRIME_API_KEY": "ambient-key",
        "PRIME_API_BASE_URL": "https://ambient.example/api/v1",
        "PRIME_BASE_URL": "https://other-ambient.example",
        "PRIME_TEAM_ID": "ambient-team",
        "PRIME_USER_ID": "ambient-user",
    }.items():
        monkeypatch.setenv(name, value)
    return prime_dir


@pytest.fixture
def no_config_or_cache_io(monkeypatch, hostile_ambient):
    # Check call counts too: legacy cache helpers deliberately swallow I/O errors.
    getenv_guard = Mock(side_effect=AssertionError("Unexpected config environment read"))
    # Replace only Config's os reference; httpx proxy/TLS environment reads stay unchanged.
    monkeypatch.setattr(config_module, "os", SimpleNamespace(getenv=getenv_guard))
    guards = [getenv_guard]
    for target, name in (
        (Path, "home"),
        (Config, "_load_config"),
        (Config, "_load_context"),
        (sandbox_module, "_load_auth_cache"),
        (sandbox_module, "_write_auth_cache"),
    ):
        guard = Mock(side_effect=AssertionError(f"Unexpected {name} call"))
        monkeypatch.setattr(target, name, guard)
        guards.append(guard)
    yield
    for guard in guards:
        guard.assert_not_called()


def _auth(token, *, expired=False):
    return {
        "gateway_url": "https://gateway.example",
        "user_ns": "namespace",
        "job_id": "job",
        "token": token,
        "expires_at": (
            datetime.now(timezone.utc) + timedelta(minutes=-1 if expired else 30)
        ).isoformat(),
    }


def _sandbox_response(request):
    payload = json.loads(request.content)
    return {
        "id": "same-sandbox",
        "name": payload["name"],
        "dockerImage": payload["docker_image"],
        "cpuCores": 1,
        "memoryGB": 1,
        "diskSizeGB": 5,
        "diskMountPath": "/data",
        "gpuCount": 0,
        "status": "RUNNING",
        "timeoutMinutes": 60,
        "createdAt": "2026-01-01T00:00:00Z",
        "updatedAt": "2026-01-01T00:00:00Z",
        "gatewayAuth": _auth(request.headers["authorization"] + ":created"),
    }


@pytest.fixture
def wire(monkeypatch):
    requests = []

    def reply(request):
        requests.append(request)
        if request.url.path == "/api/v1/sandbox":
            body = _sandbox_response(request)
        else:
            assert request.url.path == "/api/v1/sandbox/same-sandbox/auth"
            body = _auth(request.headers["authorization"] + ":refreshed")
        return httpx.Response(200, json=body, request=request)

    def send(_client, request, **kwargs):
        return reply(request)

    async def async_send(_client, request, **kwargs):
        return reply(request)

    monkeypatch.setattr(httpx.Client, "send", send)
    monkeypatch.setattr(httpx.AsyncClient, "send", async_send)
    return requests


async def _result(value):
    return await value if inspect.isawaitable(value) else value


def _new_client(mode, **kwargs):
    if mode == "async":
        return AsyncSandboxClient(**kwargs)
    persist = kwargs.pop("persist_auth_cache", True)
    return SandboxClient(APIClient(**kwargs), persist_auth_cache=persist)


async def _close(client):
    if isinstance(client, AsyncSandboxClient):
        await client.aclose()
    else:
        client.client.client.close()


@pytest.mark.parametrize(
    "api_key,base_url",
    [
        ("", "https://explicit.example"),
        ("key", ""),
        (" ", "https://explicit.example"),
        ("key", " "),
    ],
)
def test_explicit_config_requires_values(no_config_or_cache_io, api_key, base_url):
    with pytest.raises(ValueError):
        Config.from_values(api_key=api_key, base_url=base_url)


@pytest.mark.parametrize("client_type", [APIClient, AsyncAPIClient, AsyncSandboxClient])
@pytest.mark.parametrize("api_key", ["override-key", ""])
def test_explicit_config_rejects_api_key_override(no_config_or_cache_io, client_type, api_key):
    config = Config.from_values(api_key="explicit-key", base_url="https://explicit.example")
    kwargs = {"persist_auth_cache": False} if client_type is AsyncSandboxClient else {}
    with pytest.raises(ValueError, match="either api_key or config"):
        client_type(api_key=api_key, config=config, **kwargs)


@pytest.mark.asyncio
async def test_explicit_config_rejects_persistent_cache(no_config_or_cache_io, monkeypatch):
    config = Config.from_values(api_key="explicit-key", base_url="https://explicit.example")
    api = APIClient(config=config)
    async_http_client = Mock(side_effect=AssertionError("HTTP client allocated before validation"))
    monkeypatch.setattr(httpx, "AsyncClient", async_http_client)
    try:
        with pytest.raises(ValueError):
            SandboxClient(api)
        with pytest.raises(ValueError):
            AsyncSandboxClient(config=config)
        async_http_client.assert_not_called()
    finally:
        api.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_explicit_clients_isolate_scope_credentials_and_cache(
    mode, no_config_or_cache_io, wire
):
    personal = Config.from_values(
        api_key="personal-key", base_url="https://personal.example/api/v1/"
    )
    team = Config.from_values(api_key="team-key", base_url="https://team.example", team_id="team-1")
    assert personal.config_dir is personal.config_file is personal.environments_dir is None
    assert personal.user_id is personal.team_id is None
    clients = [_new_client(mode, config=c, persist_auth_cache=False) for c in (personal, team)]
    try:
        for client in clients:
            created = await _result(
                client.create(CreateSandboxRequest(name="test", docker_image="python"))
            )
            assert created.id == "same-sandbox"
        assert [str(r.url) for r in wire] == [
            "https://personal.example/api/v1/sandbox",
            "https://team.example/api/v1/sandbox",
        ]
        assert [r.headers["authorization"] for r in wire] == [
            "Bearer personal-key",
            "Bearer team-key",
        ]
        assert "team_id" not in json.loads(wire[0].content)
        assert json.loads(wire[1].content)["team_id"] == "team-1"
        for client, key in zip(clients, ("personal-key", "team-key")):
            cached = await _result(client._auth_cache.get_or_refresh("same-sandbox"))
            assert cached["token"] == f"Bearer {key}:created"
        assert len(wire) == 2  # create seeds only the owning client's cache

        await _result(clients[0]._auth_cache.invalidate("same-sandbox"))
        refreshed = await _result(clients[0]._auth_cache.get_or_refresh("same-sandbox"))
        assert refreshed["token"] == "Bearer personal-key:refreshed"
        assert str(wire[-1].url) == "https://personal.example/api/v1/sandbox/same-sandbox/auth"
        team_auth = await _result(clients[1]._auth_cache.get_or_refresh("same-sandbox"))
        assert team_auth["token"] == "Bearer team-key:created"
        assert len(wire) == 3
        for client in clients:
            await _result(client.clear_auth_cache())
    finally:
        for client in clients:
            await _close(client)


def test_sync_expired_memory_auth_refresh_is_coalesced(no_config_or_cache_io, monkeypatch):
    config = Config.from_values(api_key="sync-key", base_url="https://sync.example")
    client = SandboxClient(APIClient(config=config), persist_auth_cache=False)
    cache = client._auth_cache
    cache.set("same-sandbox", _auth("expired", expired=True))
    entered, release, followers_waiting = threading.Event(), threading.Event(), threading.Event()
    requests = []

    def send(_client, request, **kwargs):
        requests.append(request)
        entered.set()
        assert release.wait(5), "refresh was not released"
        return httpx.Response(200, json=_auth("fresh"), request=request)

    monkeypatch.setattr(httpx.Client, "send", send)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            leader = pool.submit(cache.get_or_refresh, "same-sandbox")
            try:
                assert entered.wait(5), "refresh did not start"
                event = cache._inflight["same-sandbox"]
                original_wait = event.wait
                waiting = 0
                lock = threading.Lock()

                def wait(timeout=None):
                    nonlocal waiting
                    with lock:
                        waiting += 1
                        if waiting == 3:
                            followers_waiting.set()
                    return original_wait(timeout)

                monkeypatch.setattr(event, "wait", wait)
                followers = [pool.submit(cache.get_or_refresh, "same-sandbox") for _ in range(3)]
                assert followers_waiting.wait(5), "followers did not join in-flight refresh"
                assert len(requests) == 1
            finally:
                release.set()
            tokens = [future.result(timeout=5)["token"] for future in [leader, *followers]]
            assert tokens == ["fresh"] * 4
        assert cache.get_or_refresh("same-sandbox")["token"] == "fresh"
        assert len(requests) == 1
        cache.invalidate("same-sandbox")
        cache.get_or_refresh("same-sandbox")
        assert len(requests) == 2
        client.clear_auth_cache()
    finally:
        release.set()
        client.client.client.close()


@pytest.mark.asyncio
async def test_async_expired_memory_auth_refresh_is_coalesced(no_config_or_cache_io, monkeypatch):
    config = Config.from_values(api_key="async-key", base_url="https://async.example")
    async with AsyncSandboxClient(config=config, persist_auth_cache=False) as client:
        cache = client._auth_cache
        await cache.set("same-sandbox", _auth("expired", expired=True))
        entered, release, followers_waiting = asyncio.Event(), asyncio.Event(), asyncio.Event()
        requests = []

        async def send(_client, request, **kwargs):
            requests.append(request)
            entered.set()
            await release.wait()
            return httpx.Response(200, json=_auth("fresh"), request=request)

        monkeypatch.setattr(httpx.AsyncClient, "send", send)
        leader = asyncio.create_task(cache.get_or_refresh("same-sandbox"))
        tasks = [leader]
        waiting = 0

        async def follower():
            nonlocal waiting
            waiting += 1
            if waiting == 3:
                followers_waiting.set()
            return await cache.get_or_refresh("same-sandbox")

        try:
            await asyncio.wait_for(entered.wait(), 5)
            tasks.extend(asyncio.create_task(follower()) for _ in range(3))
            await asyncio.wait_for(followers_waiting.wait(), 5)
            assert len(requests) == 1
            assert not any(task.done() for task in tasks)
            release.set()
            results = await asyncio.wait_for(asyncio.gather(*tasks), 5)
            assert [result["token"] for result in results] == ["fresh"] * 4
            assert (await cache.get_or_refresh("same-sandbox"))["token"] == "fresh"
            assert len(requests) == 1
            await cache.invalidate("same-sandbox")
            await cache.get_or_refresh("same-sandbox")
            assert len(requests) == 2
            await client.clear_auth_cache()
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_legacy_constructor_keeps_ambient_config_and_file_cache(mode, hostile_ambient, wire):
    client = _new_client(mode)
    try:
        assert client.client.config.api_key == "ambient-key"
        assert client.client.config.base_url == "https://ambient.example"
        assert client.client.config.team_id == "ambient-team"
        assert client.client.config.user_id == "ambient-user"
        auth = await _result(client._auth_cache.get_or_refresh("same-sandbox"))
        cache_file = hostile_ambient / "sandbox_auth_cache.json"
        assert json.loads(cache_file.read_text())["same-sandbox"] == auth
        assert str(wire[0].url) == "https://ambient.example/api/v1/sandbox/same-sandbox/auth"
        assert wire[0].headers["authorization"] == "Bearer ambient-key"
        second = _new_client(mode)
        try:
            assert await _result(second._auth_cache.get_or_refresh("same-sandbox")) == auth
            assert len(wire) == 1
            await _result(second.clear_auth_cache())
            assert json.loads(cache_file.read_text()) == {}
        finally:
            await _close(second)
    finally:
        await _close(client)
