"""Tests for gateway sandbox-not-found error mapping."""

import httpx
import pytest

from prime_sandboxes.core.client import APIClient
from prime_sandboxes.exceptions import SandboxNotRunningError
from prime_sandboxes.sandbox import AsyncSandboxClient, SandboxClient


class _DummyAuthCache:
    def get_or_refresh(self, _sandbox_id: str):
        return {
            "gateway_url": "https://gateway.example.com",
            "user_ns": "ns",
            "job_id": "job",
            "token": "tok",
        }


class _AsyncDummyAuthCache:
    async def get_or_refresh_async(self, _sandbox_id: str):
        return {
            "gateway_url": "https://gateway.example.com",
            "user_ns": "ns",
            "job_id": "job",
            "token": "tok",
        }


def _sandbox_not_found_error() -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://gateway.example.com/ns/job/exec")
    response = httpx.Response(
        502,
        request=request,
        json={"message": "The sandbox was not found", "code": 502},
    )
    return httpx.HTTPStatusError("bad gateway", request=request, response=response)


def test_sync_execute_command_maps_gateway_502_to_not_running():
    client = SandboxClient(APIClient(api_key="test-key"))
    client._auth_cache = _DummyAuthCache()

    def _raise_gateway_error(*args, **kwargs):
        raise _sandbox_not_found_error()

    client._gateway_post = _raise_gateway_error
    client._get_sandbox_error_context = lambda _sandbox_id: {
        "status": "RUNNING",
        "error_type": None,
        "error_message": None,
    }

    with pytest.raises(SandboxNotRunningError) as exc_info:
        client.execute_command("sbx-123", "ls /")

    assert "sandbox is no longer running" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_async_execute_command_maps_gateway_502_to_not_running():
    client = AsyncSandboxClient(api_key="test-key")
    client._auth_cache = _AsyncDummyAuthCache()

    async def _raise_gateway_error(*args, **kwargs):
        raise _sandbox_not_found_error()

    async def _fake_error_context(_sandbox_id: str):
        return {
            "status": "RUNNING",
            "error_type": None,
            "error_message": None,
        }

    client._gateway_post = _raise_gateway_error
    client._get_sandbox_error_context = _fake_error_context

    with pytest.raises(SandboxNotRunningError) as exc_info:
        await client.execute_command("sbx-123", "ls /")

    assert "sandbox is no longer running" in str(exc_info.value).lower()
