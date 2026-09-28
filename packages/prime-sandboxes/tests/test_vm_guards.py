"""Tests for the VM-only create wire contract.

The container-era VM-guarded operations (port exposure, SSH sessions, the
is_vm lookup) are gone; what remains pinned here is the create() runtime
injection.
"""

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from prime_sandboxes import CreateSandboxRequest, Sandbox
from prime_sandboxes.core.client import APIClient
from prime_sandboxes.sandbox import AsyncSandboxClient, SandboxClient


def _auth_payload():
    return {
        "gateway_url": "https://gateway.example.com",
        "user_ns": "ns",
        "job_id": "job",
        "token": "tok",
        "expires_at": "2026-01-01T00:30:00+00:00",
    }


class _FakeCache:
    def get_or_refresh(self, _sandbox_id: str):
        return _auth_payload()


class _AsyncFakeCache:
    async def get_or_refresh(self, _sandbox_id: str):
        return _auth_payload()


def _make_sync_client() -> SandboxClient:
    client = SandboxClient(APIClient(api_key="test-key"))
    cast(Any, client)._auth_cache = _FakeCache()
    return client


def _make_async_client() -> AsyncSandboxClient:
    client = AsyncSandboxClient(api_key="test-key")
    cast(Any, client)._auth_cache = _AsyncFakeCache()
    return client


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
    client = _make_sync_client()
    client.client = MagicMock()
    client.client.config.team_id = None
    client.client.request.return_value = _sandbox_response()

    sandbox = client.create(CreateSandboxRequest(name="t", docker_image="img"))

    payload = client.client.request.call_args.kwargs["json"]
    assert payload["vm"] is True
    assert isinstance(sandbox, Sandbox)


@pytest.mark.asyncio
async def test_async_create_injects_vm_true_on_the_wire():
    client = _make_async_client()
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
    client = _make_sync_client()
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
    client = _make_async_client()
    client.client = MagicMock()
    client.client.request = AsyncMock(return_value=_ssh_response())

    session = await client.create_ssh_session("sbx-1", "ssh-ed25519 key")
    await client.close_ssh_session("sbx-1", session.session_id)

    assert client.client.request.await_args_list[0].kwargs["json"] == {
        "public_key": "ssh-ed25519 key"
    }
