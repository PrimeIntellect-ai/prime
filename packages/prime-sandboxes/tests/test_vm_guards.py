"""Tests for VM-unsupported operation guards in SandboxClient / AsyncSandboxClient.

Mirrors the CLI's ``_guard_vm_unsupported`` behavior: the SDK should fail
fast with a clear ``APIError`` when a caller invokes an operation that is
not supported for VM-backed sandboxes, without making any HTTP calls.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, cast

import pytest

from prime_sandboxes import APIError
from prime_sandboxes.core.client import APIClient
from prime_sandboxes.sandbox import AsyncSandboxClient, SandboxAuthCache, SandboxClient


def _auth_payload():
    return {
        "gateway_url": "https://gateway.example.com",
        "user_ns": "ns",
        "job_id": "job",
        "token": "tok",
        "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat(),
    }


class _FakeCache:
    def __init__(self, is_vm: bool):
        self._is_vm = is_vm

    def get_or_refresh(self, _sandbox_id: str):
        return _auth_payload()

    def is_vm(self, _sandbox_id: str) -> bool:
        return self._is_vm


class _AsyncFakeCache:
    def __init__(self, is_vm: bool):
        self._is_vm = is_vm

    async def get_or_refresh(self, _sandbox_id: str):
        return _auth_payload()

    async def is_vm(self, _sandbox_id: str) -> bool:
        return self._is_vm


class _RecordingAPIClient:
    """Minimal stand-in for APIClient.request that records calls."""

    def __init__(self, response: Any = None):
        self.calls = []
        self._response = response if response is not None else {}

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        self.calls.append((method, path, kwargs))
        return self._response


class _AsyncRecordingAPIClient:
    def __init__(self, response: Any = None):
        self.calls = []
        self._response = response if response is not None else {}

    async def request(self, method: str, path: str, **kwargs: Any) -> Any:
        self.calls.append((method, path, kwargs))
        return self._response

    async def aclose(self) -> None:
        return None


def _make_sync_client(is_vm: bool) -> tuple[SandboxClient, _RecordingAPIClient]:
    client = SandboxClient(APIClient(api_key="test-key"))
    recording = _RecordingAPIClient()
    # Swap in recording http client + fake cache
    cast(Any, client).client = recording
    cast(Any, client)._auth_cache = _FakeCache(is_vm=is_vm)
    return client, recording


def _make_async_client(is_vm: bool) -> tuple[AsyncSandboxClient, _AsyncRecordingAPIClient]:
    client = AsyncSandboxClient(api_key="test-key")
    recording = _AsyncRecordingAPIClient()
    cast(Any, client).client = recording
    cast(Any, client)._auth_cache = _AsyncFakeCache(is_vm=is_vm)
    return client, recording


# ---------------------------------------------------------------------------
# Sync guard tests
# ---------------------------------------------------------------------------


def test_sync_expose_blocked_for_vm():
    client, recording = _make_sync_client(is_vm=True)
    with pytest.raises(APIError) as exc_info:
        client.expose("sbx-vm", 8000)
    assert "Port exposure" in str(exc_info.value)
    assert "VM sandboxes" in str(exc_info.value)
    assert recording.calls == []


def test_sync_unexpose_blocked_for_vm():
    client, recording = _make_sync_client(is_vm=True)
    with pytest.raises(APIError) as exc_info:
        client.unexpose("sbx-vm", "exp-1")
    assert "Port unexpose" in str(exc_info.value)
    assert recording.calls == []


def test_sync_list_exposed_ports_blocked_for_vm():
    client, recording = _make_sync_client(is_vm=True)
    with pytest.raises(APIError) as exc_info:
        client.list_exposed_ports("sbx-vm")
    assert "Port listing" in str(exc_info.value)
    assert recording.calls == []


def test_sync_create_ssh_session_blocked_for_vm():
    client, recording = _make_sync_client(is_vm=True)
    with pytest.raises(APIError) as exc_info:
        client.create_ssh_session("sbx-vm")
    assert "SSH" in str(exc_info.value)
    assert recording.calls == []


def test_sync_close_ssh_session_blocked_for_vm():
    client, recording = _make_sync_client(is_vm=True)
    with pytest.raises(APIError) as exc_info:
        client.close_ssh_session("sbx-vm", "sess-1")
    assert "SSH" in str(exc_info.value)
    assert recording.calls == []


# ---------------------------------------------------------------------------
# Sync: container path still reaches the HTTP client
# ---------------------------------------------------------------------------


def test_sync_expose_allowed_for_container():
    client, recording = _make_sync_client(is_vm=False)
    cast(Any, recording)._response = {
        "exposure_id": "exp-1",
        "sandbox_id": "sbx-c",
        "port": 8000,
        "name": None,
        "url": "https://u",
        "tls_socket": "tls",
    }
    client.expose("sbx-c", 8000)
    assert any(method == "POST" and "/expose" in path for method, path, _ in recording.calls)


def test_sync_unexpose_allowed_for_container():
    client, recording = _make_sync_client(is_vm=False)
    client.unexpose("sbx-c", "exp-1")
    assert any(
        method == "DELETE" and path.endswith("/expose/exp-1") for method, path, _ in recording.calls
    )


def test_sync_list_exposed_ports_allowed_for_container():
    client, recording = _make_sync_client(is_vm=False)
    cast(Any, recording)._response = {"exposures": []}
    client.list_exposed_ports("sbx-c")
    assert any(method == "GET" and path.endswith("/expose") for method, path, _ in recording.calls)


def test_sync_create_ssh_session_allowed_for_container():
    client, recording = _make_sync_client(is_vm=False)
    cast(Any, recording)._response = {
        "session_id": "s",
        "exposure_id": "e",
        "sandbox_id": "sbx-c",
        "host": "h",
        "port": 22,
        "external_endpoint": "h:22",
        "expires_at": datetime.now(timezone.utc).isoformat(),
        "ttl_seconds": 300,
        "gateway_url": "https://g",
        "user_ns": "ns",
        "job_id": "job",
        "token": "tok",
    }
    client.create_ssh_session("sbx-c")
    assert any(
        method == "POST" and path.endswith("/ssh-session") for method, path, _ in recording.calls
    )


def test_sync_close_ssh_session_allowed_for_container():
    client, recording = _make_sync_client(is_vm=False)
    client.close_ssh_session("sbx-c", "sess-1")
    assert any(
        method == "DELETE" and path.endswith("/ssh-session/sess-1")
        for method, path, _ in recording.calls
    )


def test_sync_list_all_exposed_ports_not_guarded():
    """list_all_exposed_ports has no sandbox_id and is not VM-guarded (matches CLI)."""
    client = SandboxClient(APIClient(api_key="test-key"))
    recording = _RecordingAPIClient(response={"exposures": []})
    cast(Any, client).client = recording
    # Cache raises if queried: proves the call doesn't touch the guard

    class _BoomCache:
        def is_vm(self, _sandbox_id: str) -> bool:
            raise AssertionError("is_vm should not be called for list_all_exposed_ports")

    cast(Any, client)._auth_cache = _BoomCache()
    client.list_all_exposed_ports()
    assert any(
        method == "GET" and path == "/sandbox/expose/all" for method, path, _ in recording.calls
    )


# ---------------------------------------------------------------------------
# Public is_vm helper
# ---------------------------------------------------------------------------


def test_sync_is_vm_delegates_to_cache_true():
    client, _ = _make_sync_client(is_vm=True)
    assert client.is_vm("sbx-vm") is True


def test_sync_is_vm_delegates_to_cache_false():
    client, _ = _make_sync_client(is_vm=False)
    assert client.is_vm("sbx-c") is False


def test_sync_is_vm_hits_backend_on_cold_cache(tmp_path):
    """SandboxAuthCache.is_vm falls back to GET /sandbox/<id> when cached flag is missing."""

    class _FakeAPIClient:
        def __init__(self):
            self.calls = 0

        def request(self, method: str, path: str, **_kwargs):
            if method == "GET" and path == "/sandbox/sbx-1":
                self.calls += 1
                return {
                    "id": "sbx-1",
                    "name": "vm-box",
                    "dockerImage": "img",
                    "startCommand": None,
                    "cpuCores": 1.0,
                    "memoryGB": 2.0,
                    "diskSizeGB": 10.0,
                    "diskMountPath": "/sandbox-workspace",
                    "gpuCount": 0,
                    "gpuType": None,
                    "vm": True,
                    "networkAccess": True,
                    "status": "RUNNING",
                    "timeoutMinutes": 60,
                    "environmentVars": None,
                    "secrets": None,
                    "advancedConfigs": None,
                    "labels": [],
                    "createdAt": datetime.now(timezone.utc).isoformat(),
                    "updatedAt": datetime.now(timezone.utc).isoformat(),
                    "startedAt": None,
                    "terminatedAt": None,
                    "exitCode": None,
                    "errorType": None,
                    "errorMessage": None,
                    "userId": "user",
                    "teamId": "team",
                    "kubernetesJobId": None,
                    "registryCredentialsId": None,
                }
            raise AssertionError(f"Unexpected request: {method} {path}")

    cache = SandboxAuthCache(tmp_path / "auth_cache.json", _FakeAPIClient())
    cache.set("sbx-1", _auth_payload())

    assert cache.is_vm("sbx-1") is True
    # Second call uses the cached flag: backend hit count stays at 1.
    assert cache.is_vm("sbx-1") is True
    assert cache.client.calls == 1


# ---------------------------------------------------------------------------
# Async guard tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_expose_blocked_for_vm():
    client, recording = _make_async_client(is_vm=True)
    try:
        with pytest.raises(APIError) as exc_info:
            await client.expose("sbx-vm", 8000)
        assert "Port exposure" in str(exc_info.value)
        assert recording.calls == []
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_async_unexpose_blocked_for_vm():
    client, recording = _make_async_client(is_vm=True)
    try:
        with pytest.raises(APIError) as exc_info:
            await client.unexpose("sbx-vm", "exp-1")
        assert "Port unexpose" in str(exc_info.value)
        assert recording.calls == []
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_async_list_exposed_ports_blocked_for_vm():
    client, recording = _make_async_client(is_vm=True)
    try:
        with pytest.raises(APIError) as exc_info:
            await client.list_exposed_ports("sbx-vm")
        assert "Port listing" in str(exc_info.value)
        assert recording.calls == []
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_async_create_ssh_session_blocked_for_vm():
    client, recording = _make_async_client(is_vm=True)
    try:
        with pytest.raises(APIError) as exc_info:
            await client.create_ssh_session("sbx-vm")
        assert "SSH" in str(exc_info.value)
        assert recording.calls == []
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_async_close_ssh_session_blocked_for_vm():
    client, recording = _make_async_client(is_vm=True)
    try:
        with pytest.raises(APIError) as exc_info:
            await client.close_ssh_session("sbx-vm", "sess-1")
        assert "SSH" in str(exc_info.value)
        assert recording.calls == []
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_async_expose_allowed_for_container():
    client, recording = _make_async_client(is_vm=False)
    cast(Any, recording)._response = {
        "exposure_id": "exp-1",
        "sandbox_id": "sbx-c",
        "port": 8000,
        "name": None,
        "url": "https://u",
        "tls_socket": "tls",
    }
    try:
        await client.expose("sbx-c", 8000)
        assert any(method == "POST" and "/expose" in path for method, path, _ in recording.calls)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_async_is_vm_public_helper():
    client, _ = _make_async_client(is_vm=True)
    try:
        assert (await client.is_vm("sbx-vm")) is True
    finally:
        await client.aclose()
