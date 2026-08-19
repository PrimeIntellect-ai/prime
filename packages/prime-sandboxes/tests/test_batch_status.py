"""Focused tests for platform lifecycle and VM background-job batch calls."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, cast

import pytest

from prime_sandboxes import BatchStatusUnsupportedError
from prime_sandboxes.core.client import APIClient
from prime_sandboxes.models import BackgroundJob, ReadFileResponse
from prime_sandboxes.sandbox import AsyncSandboxClient, SandboxClient


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


class _SyncBackgroundJobPlatformClient:
    def __init__(self, reject_as_container: bool = False) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.reject_as_container = reject_as_container

    def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((method, path, kwargs))
        jobs = kwargs["json"]["jobs"]
        if self.reject_as_container:
            return {
                "statuses": [],
                "errors": [
                    {
                        **jobs[0],
                        "code": "NOT_VM",
                        "message": (
                            "Batched background job status is only supported for VM sandboxes"
                        ),
                    }
                ],
            }
        return {
            "statuses": [
                {
                    **job,
                    "completed": job["job_id"] == "feedface",
                    "exit_code": 7 if job["job_id"] == "feedface" else None,
                }
                for job in jobs
            ],
            "errors": [],
        }


class _AsyncBackgroundJobPlatformClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((method, path, kwargs))
        return {
            "statuses": [
                {**job, "completed": False, "exit_code": None} for job in kwargs["json"]["jobs"]
            ],
            "errors": [],
        }

    async def aclose(self) -> None:
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


def test_sync_get_background_jobs_uses_one_platform_batch_across_vm_sandboxes() -> None:
    client = SandboxClient(APIClient(api_key="test-key"))
    client.client.client.close()
    platform = _SyncBackgroundJobPlatformClient()
    cast(Any, client).client = platform

    def read_file(
        _sandbox_id: str,
        path: str,
        timeout: Optional[int] = None,
        offset: Optional[int] = None,
        length: Optional[int] = None,
    ) -> ReadFileResponse:
        content = "stdout" if path.endswith("stdout.log") else "stderr"
        return ReadFileResponse(content=content, size=len(content), truncated=False)

    cast(Any, client).read_file = read_file
    jobs = [_job("sandbox-a", "deadbeef"), _job("sandbox-b", "feedface")]

    statuses = client.get_background_jobs(jobs, timeout=12)

    assert len(platform.calls) == 1
    method, path, kwargs = platform.calls[0]
    assert method == "POST"
    assert path == "/sandbox/background-jobs/status:batchGet"
    assert kwargs["json"] == {
        "jobs": [
            {"sandbox_id": "sandbox-a", "job_id": "deadbeef"},
            {"sandbox_id": "sandbox-b", "job_id": "feedface"},
        ]
    }
    assert kwargs["timeout"] == 12
    assert kwargs["idempotent_post"] is True
    assert not statuses[0].completed
    assert statuses[1].completed
    assert statuses[1].exit_code == 7
    assert statuses[1].stdout == "stdout"
    assert statuses[1].stderr == "stderr"


def test_sync_get_background_jobs_rejects_container_sandboxes() -> None:
    client = SandboxClient(APIClient(api_key="test-key"))
    client.client.client.close()
    cast(Any, client).client = _SyncBackgroundJobPlatformClient(reject_as_container=True)

    with pytest.raises(BatchStatusUnsupportedError, match="only supported for VM"):
        client.get_background_jobs([_job("sandbox-container", "deadbeef")])


def test_concurrent_sync_background_waiters_share_one_platform_batch() -> None:
    client = SandboxClient(APIClient(api_key="test-key"))
    client.client.client.close()
    platform = _SyncBackgroundJobPlatformClient()
    cast(Any, client).client = platform

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                cast(Any, client)._background_job_status_batcher.get,
                key,
            )
            for key in [
                ("sandbox-a", "deadbeef"),
                ("sandbox-b", "cafebabe"),
            ]
        ]
        for future in futures:
            assert not future.result().completed

    assert len(platform.calls) == 1
    assert {(job["sandbox_id"], job["job_id"]) for job in platform.calls[0][2]["json"]["jobs"]} == {
        ("sandbox-a", "deadbeef"),
        ("sandbox-b", "cafebabe"),
    }


@pytest.mark.asyncio
async def test_async_get_background_jobs_uses_one_platform_batch() -> None:
    client = AsyncSandboxClient(api_key="test-key")
    await client.client.aclose()
    platform = _AsyncBackgroundJobPlatformClient()
    cast(Any, client).client = platform
    try:
        statuses = await client.get_background_jobs(
            [_job("sandbox-a", "deadbeef"), _job("sandbox-b", "feedface")]
        )
    finally:
        await client.aclose()

    assert len(platform.calls) == 1
    assert platform.calls[0][1] == "/sandbox/background-jobs/status:batchGet"
    assert platform.calls[0][2]["json"] == {
        "jobs": [
            {"sandbox_id": "sandbox-a", "job_id": "deadbeef"},
            {"sandbox_id": "sandbox-b", "job_id": "feedface"},
        ]
    }
    assert not statuses[0].completed
    assert not statuses[1].completed


@pytest.mark.asyncio
async def test_concurrent_async_background_waiters_share_one_platform_batch() -> None:
    client = AsyncSandboxClient(api_key="test-key")
    await client.client.aclose()
    platform = _AsyncBackgroundJobPlatformClient()
    cast(Any, client).client = platform
    try:
        statuses = await asyncio.gather(
            cast(Any, client)._background_job_status_batcher.get(("sandbox-a", "deadbeef")),
            cast(Any, client)._background_job_status_batcher.get(("sandbox-b", "cafebabe")),
        )
    finally:
        await client.aclose()

    assert all(not status.completed for status in statuses)
    assert len(platform.calls) == 1
    assert {(job["sandbox_id"], job["job_id"]) for job in platform.calls[0][2]["json"]["jobs"]} == {
        ("sandbox-a", "deadbeef"),
        ("sandbox-b", "cafebabe"),
    }
