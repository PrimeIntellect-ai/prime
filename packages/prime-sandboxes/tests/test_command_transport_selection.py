"""Tests for CPU/GPU command transport selection."""

from datetime import datetime, timedelta, timezone

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError

from prime_sandboxes._proto.process import process_pb2
from prime_sandboxes.core.client import APIClient
from prime_sandboxes.models import CommandResponse
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
    def __init__(self, is_gpu: bool):
        self._is_gpu = is_gpu

    def get_or_refresh(self, _sandbox_id: str):
        return _auth_payload()

    def is_gpu(self, _sandbox_id: str) -> bool:
        return self._is_gpu


class _AsyncFakeCache:
    def __init__(self, is_gpu: bool):
        self._is_gpu = is_gpu

    async def get_or_refresh_async(self, _sandbox_id: str):
        return _auth_payload()

    async def is_gpu_async(self, _sandbox_id: str) -> bool:
        return self._is_gpu


def test_sync_execute_command_uses_connect_for_gpu():
    client = SandboxClient(APIClient(api_key="test-key"))
    client._auth_cache = _FakeCache(is_gpu=True)

    called = {"connect": False}

    def _connect(*_args, **_kwargs):
        called["connect"] = True
        return CommandResponse(stdout="ok", stderr="", exit_code=0)

    def _rest(*_args, **_kwargs):
        raise AssertionError("REST path should not be used for GPU sandboxes")

    client._execute_command_connect_rpc = _connect
    client._execute_command_rest = _rest

    result = client.execute_command("sbx-gpu", "echo hi")

    assert called["connect"]
    assert result.exit_code == 0


def test_sync_execute_command_uses_rest_for_cpu():
    client = SandboxClient(APIClient(api_key="test-key"))
    client._auth_cache = _FakeCache(is_gpu=False)

    called = {"rest": False}

    def _connect(*_args, **_kwargs):
        raise AssertionError("Connect path should not be used for CPU sandboxes")

    def _rest(*_args, **_kwargs):
        called["rest"] = True
        return CommandResponse(stdout="ok", stderr="", exit_code=0)

    client._execute_command_connect_rpc = _connect
    client._execute_command_rest = _rest

    result = client.execute_command("sbx-cpu", "echo hi")

    assert called["rest"]
    assert result.exit_code == 0


@pytest.mark.asyncio
async def test_async_execute_command_uses_connect_for_gpu():
    client = AsyncSandboxClient(api_key="test-key")
    client._auth_cache = _AsyncFakeCache(is_gpu=True)

    called = {"connect": False}

    async def _connect(*_args, **_kwargs):
        called["connect"] = True
        return CommandResponse(stdout="ok", stderr="", exit_code=0)

    async def _rest(*_args, **_kwargs):
        raise AssertionError("REST path should not be used for GPU sandboxes")

    client._execute_command_connect_rpc = _connect
    client._execute_command_rest = _rest

    try:
        result = await client.execute_command("sbx-gpu", "echo hi")

        assert called["connect"]
        assert result.exit_code == 0
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_async_execute_command_uses_rest_for_cpu():
    client = AsyncSandboxClient(api_key="test-key")
    client._auth_cache = _AsyncFakeCache(is_gpu=False)

    called = {"rest": False}

    async def _connect(*_args, **_kwargs):
        raise AssertionError("Connect path should not be used for CPU sandboxes")

    async def _rest(*_args, **_kwargs):
        called["rest"] = True
        return CommandResponse(stdout="ok", stderr="", exit_code=0)

    client._execute_command_connect_rpc = _connect
    client._execute_command_rest = _rest

    try:
        result = await client.execute_command("sbx-cpu", "echo hi")

        assert called["rest"]
        assert result.exit_code == 0
    finally:
        await client.aclose()


def test_auth_cache_stores_gpu_flag_for_reuse(tmp_path):
    class _FakeAPIClient:
        def __init__(self):
            self.calls = 0

        def request(self, method: str, path: str):
            if method == "GET" and path == "/sandbox/sbx-1":
                self.calls += 1
                return {
                    "id": "sbx-1",
                    "name": "gpu-box",
                    "dockerImage": "img",
                    "startCommand": None,
                    "cpuCores": 1.0,
                    "memoryGB": 2.0,
                    "diskSizeGB": 10.0,
                    "diskMountPath": "/sandbox-workspace",
                    "gpuCount": 1,
                    "gpuType": "RTX_PRO_6000",
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

    assert cache.is_gpu("sbx-1")
    assert cache.is_gpu("sbx-1")
    assert cache.client.calls == 1


def test_sync_connect_execution_collects_stdout_stderr(monkeypatch):
    class _FakeConnectClient:
        def __init__(self, address: str):
            self.address = address

        def execute_server_stream(self, **_kwargs):
            yield process_pb2.StartResponse(
                event=process_pb2.ProcessEvent(
                    data=process_pb2.ProcessEvent.DataEvent(stdout=b"hello\n")
                )
            )
            yield process_pb2.StartResponse(
                event=process_pb2.ProcessEvent(
                    data=process_pb2.ProcessEvent.DataEvent(stderr=b"warn\n")
                )
            )
            yield process_pb2.StartResponse(
                event=process_pb2.ProcessEvent(
                    end=process_pb2.ProcessEvent.EndEvent(exit_code=7, exited=True, status="exit")
                )
            )

        def close(self):
            return None

    monkeypatch.setattr("prime_sandboxes.sandbox.ConnectClientSync", _FakeConnectClient)

    client = SandboxClient(APIClient(api_key="test-key"))
    result = client._execute_command_connect_rpc(
        sandbox_id="sbx-gpu",
        command="echo hi",
        auth=_auth_payload(),
    )

    assert result.stdout == "hello\n"
    assert result.stderr == "warn\n"
    assert result.exit_code == 7


def test_sync_connect_execution_maps_deadline_to_timeout(monkeypatch):
    class _FakeConnectClient:
        def __init__(self, _address: str):
            pass

        def execute_server_stream(self, **_kwargs):
            raise ConnectError(Code.DEADLINE_EXCEEDED, "deadline")

        def close(self):
            return None

    monkeypatch.setattr("prime_sandboxes.sandbox.ConnectClientSync", _FakeConnectClient)

    client = SandboxClient(APIClient(api_key="test-key"))
    client._get_sandbox_error_context = lambda sandbox_id: {
        "status": "RUNNING",
        "error_type": None,
        "error_message": None,
    }

    with pytest.raises(Exception) as exc_info:
        client._execute_command_connect_rpc(
            sandbox_id="sbx-gpu",
            command="sleep 100",
            auth=_auth_payload(),
            timeout=2,
        )

    assert "timed out" in str(exc_info.value).lower()
