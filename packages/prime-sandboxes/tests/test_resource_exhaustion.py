"""Resource-exhaustion diagnostics at the SDK error boundaries."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from prime_sandboxes import (
    APIClient,
    APIError,
    AsyncAPIClient,
    AsyncSandboxClient,
    SandboxClient,
    SandboxDiskFullError,
    SandboxNotRunningError,
    SandboxOOMError,
)
from prime_sandboxes.sandbox import _build_terminated_message

TERMINATION_CASES = [
    ("OOM_KILLED", SandboxOOMError, "The sandbox ran out of memory."),
    ("SANDBOX_UNRESPONSIVE", SandboxNotRunningError, "stopped responding and was terminated"),
    ("NODE_FAILURE", SandboxNotRunningError, "host failed; this was not caused by the sandbox"),
    (None, SandboxNotRunningError, "The sandbox was terminated."),
]


@pytest.mark.parametrize("error_type,exception_type,expected", TERMINATION_CASES)
def test_build_terminated_message(error_type, exception_type, expected):
    message = _build_terminated_message(
        "python worker.py",
        {"status": "TERMINATED", "error_type": error_type, "error_message": "runtime evidence"},
    )

    assert "Command 'python worker.py' failed: sandbox is no longer running." in message
    assert expected in message
    assert message.endswith("Details: runtime evidence")
    if error_type == "OOM_KILLED":
        assert "Consider requesting more memory or reducing memory usage." in message
    elif error_type == "SANDBOX_UNRESPONSIVE":
        assert "memory" not in message
        assert "disk" not in message
    elif error_type == "NODE_FAILURE":
        assert "Retry with a new sandbox." in message


@pytest.mark.parametrize("error_type,exception_type,expected", TERMINATION_CASES)
def test_sync_error_context_maps_termination(error_type, exception_type, expected):
    def context_response(request):
        assert request.url.path == "/api/v1/sandbox/sandbox-1/error-context"
        return httpx.Response(
            200,
            json={
                "status": "TERMINATED",
                "errorType": error_type,
                "errorMessage": "runtime evidence",
            },
        )

    api = APIClient(api_key="test-key")
    api.client.close()
    with httpx.Client(transport=httpx.MockTransport(context_response)) as api.client:
        client = SandboxClient.__new__(SandboxClient)
        client.client = api
        request = httpx.Request("POST", "https://gateway.example/exec")
        error = httpx.HTTPStatusError(
            "not running", request=request, response=httpx.Response(409, request=request)
        )
        with pytest.raises(exception_type) as exc_info:
            client._should_retry_409("sandbox-1", error, 0, command="python worker.py")

    assert type(exc_info.value) is exception_type
    assert exc_info.value.error_type == error_type
    assert exc_info.value.status == "TERMINATED"
    assert expected in str(exc_info.value)
    assert "Details: runtime evidence" in str(exc_info.value)
    assert exc_info.value.__cause__ is error


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type,exception_type,expected", TERMINATION_CASES)
async def test_async_error_context_maps_termination(error_type, exception_type, expected):
    def context_response(request):
        assert request.url.path == "/api/v1/sandbox/sandbox-1/error-context"
        return httpx.Response(
            200,
            json={
                "status": "TERMINATED",
                "errorType": error_type,
                "errorMessage": "runtime evidence",
            },
        )

    api = AsyncAPIClient(api_key="test-key")
    await api.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(context_response)) as api.client:
        client = AsyncSandboxClient.__new__(AsyncSandboxClient)
        client.client = api
        request = httpx.Request("POST", "https://gateway.example/exec")
        error = httpx.HTTPStatusError(
            "not running", request=request, response=httpx.Response(409, request=request)
        )
        with pytest.raises(exception_type) as exc_info:
            await client._should_retry_409("sandbox-1", error, 0, command="python worker.py")

    assert type(exc_info.value) is exception_type
    assert exc_info.value.error_type == error_type
    assert exc_info.value.status == "TERMINATED"
    assert expected in str(exc_info.value)
    assert "Details: runtime evidence" in str(exc_info.value)
    assert exc_info.value.__cause__ is error


@pytest.mark.parametrize("method", ["upload_file", "upload_bytes"])
def test_sync_upload_disk_full(method, tmp_path, monkeypatch):
    requests = []

    def disk_full(request):
        requests.append(request)
        return httpx.Response(507, json={"message": "failed to create directory: no space left"})

    transport = httpx.MockTransport(disk_full)
    http_client = httpx.Client
    monkeypatch.setattr(
        "prime_sandboxes.sandbox.httpx.Client",
        lambda **kwargs: http_client(transport=transport, **kwargs),
    )
    client = SandboxClient.__new__(SandboxClient)
    client._auth_cache = SimpleNamespace(
        get_or_refresh=lambda _sandbox_id: {
            "gateway_url": "https://gateway.example",
            "user_ns": "ns",
            "job_id": "job",
            "token": "tok",
        }
    )
    local_path = tmp_path / "file"
    local_path.write_bytes(b"data")
    args = (str(local_path),) if method == "upload_file" else (b"data", "file")

    with pytest.raises(SandboxDiskFullError) as exc_info:
        getattr(client, method)("sandbox-1", "/tmp/file", *args)

    assert isinstance(exc_info.value, APIError)
    assert "sandbox disk is full (no space left)" in str(exc_info.value)
    assert "failed to create directory: no space left" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, httpx.HTTPStatusError)
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["upload_file", "upload_bytes"])
async def test_async_upload_disk_full(method, tmp_path):
    requests = []

    def disk_full(request):
        requests.append(request)
        return httpx.Response(507, json={"message": "failed to create directory: no space left"})

    client = AsyncSandboxClient.__new__(AsyncSandboxClient)
    client._auth_cache = SimpleNamespace(
        get_or_refresh=AsyncMock(
            return_value={
                "gateway_url": "https://gateway.example",
                "user_ns": "ns",
                "job_id": "job",
                "token": "tok",
            }
        )
    )
    local_path = tmp_path / "file"
    local_path.write_bytes(b"data")
    args = (str(local_path),) if method == "upload_file" else (b"data", "file")
    async with httpx.AsyncClient(transport=httpx.MockTransport(disk_full)) as gateway:
        client._gateway_client = gateway
        with pytest.raises(SandboxDiskFullError) as exc_info:
            await getattr(client, method)("sandbox-1", "/tmp/file", *args)

    assert isinstance(exc_info.value, APIError)
    assert "sandbox disk is full (no space left)" in str(exc_info.value)
    assert "failed to create directory: no space left" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, httpx.HTTPStatusError)
    assert len(requests) == 1
