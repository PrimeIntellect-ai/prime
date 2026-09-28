"""Tests for the VM-only create wire contract and the public is_vm helpers.

The container-era VM-guarded operations (port exposure, SSH sessions) are gone;
what remains pinned here is the is_vm lookup contract and the create() runtime
injection.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from prime_sandboxes import CreateSandboxRequest, Sandbox
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


def _make_sync_client(is_vm: bool) -> SandboxClient:
    client = SandboxClient(APIClient(api_key="test-key"))
    cast(Any, client)._auth_cache = _FakeCache(is_vm=is_vm)
    return client


def _make_async_client(is_vm: bool) -> AsyncSandboxClient:
    client = AsyncSandboxClient(api_key="test-key")
    cast(Any, client)._auth_cache = _AsyncFakeCache(is_vm=is_vm)
    return client


# ---------------------------------------------------------------------------
# Public is_vm helper
# ---------------------------------------------------------------------------


def test_sync_is_vm_delegates_to_cache_true():
    client = _make_sync_client(is_vm=True)
    assert client.is_vm("sbx-vm") is True


def test_sync_is_vm_delegates_to_cache_false():
    client = _make_sync_client(is_vm=False)
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
                    "registryCredentialsId": None,
                }
            raise AssertionError(f"Unexpected request: {method} {path}")

    cache = SandboxAuthCache(tmp_path / "auth_cache.json", _FakeAPIClient())
    cache.set("sbx-1", _auth_payload())

    assert cache.is_vm("sbx-1") is True
    # Second call uses the cached flag: backend hit count stays at 1.
    assert cache.is_vm("sbx-1") is True
    assert cache.client.calls == 1


@pytest.mark.asyncio
async def test_async_is_vm_public_helper():
    client = _make_async_client(is_vm=True)
    try:
        assert (await client.is_vm("sbx-vm")) is True
    finally:
        await client.aclose()


def _sandbox_response() -> dict:
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
        "status": "RUNNING",
        "timeoutMinutes": 60,
        "labels": [],
        "createdAt": "2026-01-01T00:00:00+00:00",
        "updatedAt": "2026-01-01T00:00:00+00:00",
    }


# ---------------------------------------------------------------------------
# Create always sends vm=true
# ---------------------------------------------------------------------------


def test_sync_create_injects_vm_true_on_the_wire():
    client = _make_sync_client(is_vm=True)
    client.client = MagicMock()
    client.client.config.team_id = None
    client.client.request.return_value = _sandbox_response()

    sandbox = client.create(CreateSandboxRequest(name="t", docker_image="img"))

    payload = client.client.request.call_args.kwargs["json"]
    assert payload["vm"] is True
    assert isinstance(sandbox, Sandbox)


@pytest.mark.asyncio
async def test_async_create_injects_vm_true_on_the_wire():
    client = _make_async_client(is_vm=True)
    client.client = MagicMock()
    client.client.config.team_id = None
    client.client.request = AsyncMock(return_value=_sandbox_response())

    sandbox = await client.create(CreateSandboxRequest(name="t", docker_image="img"))

    payload = client.client.request.call_args.kwargs["json"]
    assert payload["vm"] is True
    assert isinstance(sandbox, Sandbox)


def _ssh_response() -> dict:
    return {
        "session_id": "session-1",
        "sandbox_id": "sbx-1",
        "host": "ssh.example.com",
        "port": 2222,
        "expires_at": "2026-01-01T00:05:00+00:00",
        "ttl_seconds": 300,
    }


def test_sync_vm_ssh_session_sends_public_key():
    client = _make_sync_client(is_vm=True)
    client.client = MagicMock()
    client.client.request.return_value = _ssh_response()

    session = client.create_ssh_session("sbx-1", "ssh-ed25519 key", ttl_seconds=300)
    client.close_ssh_session("sbx-1", session.session_id)

    assert client.client.request.call_args_list[0].kwargs["json"] == {
        "public_key": "ssh-ed25519 key",
        "ttl_seconds": 300,
    }
    assert client.client.request.call_args_list[1].args == (
        "DELETE",
        "/sandbox/sbx-1/ssh-session/session-1",
    )


@pytest.mark.asyncio
async def test_async_vm_ssh_session_sends_public_key():
    client = _make_async_client(is_vm=True)
    client.client = MagicMock()
    client.client.request = AsyncMock(return_value=_ssh_response())

    session = await client.create_ssh_session("sbx-1", "ssh-ed25519 key")
    await client.close_ssh_session("sbx-1", session.session_id)

    assert client.client.request.await_args_list[0].kwargs["json"] == {
        "public_key": "ssh-ed25519 key"
    }
