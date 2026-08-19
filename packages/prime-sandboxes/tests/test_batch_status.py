"""Focused tests for platform and VM runtime batch status calls."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, cast

import httpx
import pytest

from prime_sandboxes import BatchStatusUnsupportedError
from prime_sandboxes.core.client import APIClient
from prime_sandboxes.models import BackgroundJob, ReadFileResponse
from prime_sandboxes.sandbox import AsyncSandboxClient, SandboxClient


def _auth_payload() -> dict[str, Any]:
    return {
        "gateway_url": "https://gateway.example.com",
        "user_ns": "ns",
        "job_id": "runtime",
        "token": "token",
        "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat(),
    }


def _job(sandbox_id: str, job_id: str) -> BackgroundJob:
    return BackgroundJob(
        job_id=job_id,
        sandbox_id=sandbox_id,
        stdout_log_file=f"/tmp/job_{job_id}.stdout.log",
        stderr_log_file=f"/tmp/job_{job_id}.stderr.log",
        exit_file=f"/tmp/job_{job_id}.exit",
    )


class _SyncPlatformClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((method, path, kwargs))
        sandbox_ids = kwargs["json"]["sandbox_ids"]
        return {
            "statuses": [
                {
                    "sandbox_id": sandbox_id,
                    "status": "RUNNING",
                    "error_type": None,
                    "error_message": None,
                    "pending_image_build_id": None,
                }
                for sandbox_id in sandbox_ids
            ],
            "errors": [],
        }


class _AsyncPlatformClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((method, path, kwargs))
        sandbox_ids = kwargs["json"]["sandbox_ids"]
        return {
            "statuses": [
                {
                    "sandbox_id": sandbox_id,
                    "status": "RUNNING",
                    "error_type": None,
                    "error_message": None,
                    "pending_image_build_id": None,
                }
                for sandbox_id in sandbox_ids
            ],
            "errors": [],
        }

    async def aclose(self) -> None:
        return None


class _SyncAuthCache:
    def __init__(self, is_vm: bool = True) -> None:
        self._is_vm = is_vm

    def get_or_refresh(self, _sandbox_id: str) -> dict[str, Any]:
        return _auth_payload()

    def is_vm(self, _sandbox_id: str) -> bool:
        return self._is_vm

    def invalidate(self, _sandbox_id: str) -> None:
        return None


class _AsyncAuthCache:
    def __init__(self, is_vm: bool = True) -> None:
        self._is_vm = is_vm

    async def get_or_refresh(self, _sandbox_id: str) -> dict[str, Any]:
        return _auth_payload()

    async def is_vm(self, _sandbox_id: str) -> bool:
        return self._is_vm

    async def invalidate(self, _sandbox_id: str) -> None:
        return None


def test_concurrent_sync_creation_waits_share_one_platform_batch() -> None:
    client = SandboxClient(APIClient(api_key="test-key"))
    client.client.client.close()
    platform = _SyncPlatformClient()
    cast(Any, client).client = platform
    cast(Any, client)._is_sandbox_reachable = lambda _sandbox_id: True

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(client.wait_for_creation, sandbox_id)
            for sandbox_id in ["sandbox-a", "sandbox-b"]
        ]
        for future in futures:
            future.result()

    assert len(platform.calls) == 1
    method, path, kwargs = platform.calls[0]
    assert method == "POST"
    assert path == "/sandbox/status:batchGet"
    assert set(kwargs["json"]["sandbox_ids"]) == {"sandbox-a", "sandbox-b"}
    assert kwargs["idempotent_post"] is True


@pytest.mark.asyncio
async def test_concurrent_async_creation_waits_share_one_platform_batch() -> None:
    client = AsyncSandboxClient(api_key="test-key")
    await client.client.aclose()
    platform = _AsyncPlatformClient()
    cast(Any, client).client = platform

    async def reachable(_sandbox_id: str) -> bool:
        return True

    cast(Any, client)._is_sandbox_reachable = reachable
    try:
        await asyncio.gather(
            client.wait_for_creation("sandbox-a"),
            client.wait_for_creation("sandbox-b"),
        )
    finally:
        await client.aclose()

    assert len(platform.calls) == 1
    assert set(platform.calls[0][2]["json"]["sandbox_ids"]) == {"sandbox-a", "sandbox-b"}


def test_sync_get_background_jobs_uses_one_vm_runtime_batch_and_reads_completed_output() -> None:
    client = SandboxClient(APIClient(api_key="test-key"))
    cast(Any, client)._auth_cache = _SyncAuthCache()
    calls: list[dict[str, Any]] = []

    def post(
        url: str,
        headers: dict[str, str],
        timeout: float,
        json: dict[str, Any],
    ) -> httpx.Response:
        calls.append({"url": url, "headers": headers, "timeout": timeout, "json": json})
        return httpx.Response(
            200,
            json={
                "jobs": [
                    {"job_id": "deadbeef", "completed": False},
                    {"job_id": "feedface", "completed": True, "exit_code": 7},
                ]
            },
            request=httpx.Request("POST", url),
        )

    def read_file(
        _sandbox_id: str,
        path: str,
        timeout: Optional[int] = None,
        offset: Optional[int] = None,
        length: Optional[int] = None,
    ) -> ReadFileResponse:
        content = "stdout" if path.endswith("stdout.log") else "stderr"
        return ReadFileResponse(content=content, size=len(content), truncated=False)

    cast(Any, client)._gateway_idempotent_post = post
    cast(Any, client).read_file = read_file
    jobs = [_job("sandbox-vm", "deadbeef"), _job("sandbox-vm", "feedface")]

    statuses = client.get_background_jobs(jobs, timeout=12)

    assert calls[0]["json"] == {"job_ids": ["deadbeef", "feedface"]}
    assert calls[0]["timeout"] == 12
    assert not statuses[0].completed
    assert statuses[1].completed
    assert statuses[1].exit_code == 7
    assert statuses[1].stdout == "stdout"
    assert statuses[1].stderr == "stderr"


def test_sync_get_background_jobs_rejects_container_sandboxes() -> None:
    client = SandboxClient(APIClient(api_key="test-key"))
    cast(Any, client)._auth_cache = _SyncAuthCache(is_vm=False)

    with pytest.raises(BatchStatusUnsupportedError, match="only supported for VM"):
        client.get_background_jobs([_job("sandbox-container", "deadbeef")])


@pytest.mark.asyncio
async def test_async_get_background_jobs_uses_vm_runtime_batch() -> None:
    client = AsyncSandboxClient(api_key="test-key")
    await client.client.aclose()
    cast(Any, client)._auth_cache = _AsyncAuthCache()
    calls: list[dict[str, Any]] = []

    async def post(
        url: str,
        headers: dict[str, str],
        timeout: float,
        json: dict[str, Any],
    ) -> httpx.Response:
        calls.append({"url": url, "headers": headers, "timeout": timeout, "json": json})
        return httpx.Response(
            200,
            json={"jobs": [{"job_id": "deadbeef", "completed": False}]},
            request=httpx.Request("POST", url),
        )

    cast(Any, client)._gateway_idempotent_post = post
    try:
        statuses = await client.get_background_jobs([_job("sandbox-vm", "deadbeef")])
    finally:
        await client.aclose()

    assert calls[0]["json"] == {"job_ids": ["deadbeef"]}
    assert not statuses[0].completed
