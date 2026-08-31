"""Focused tests for platform lifecycle and VM background-job batch calls."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, Optional, cast

import pytest

from prime_sandboxes import BatchStatusUnsupportedError
from prime_sandboxes.core.client import APIClient, APIError
from prime_sandboxes.models import BackgroundJob, BackgroundJobStatus, ReadFileResponse
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
    def __init__(self, error_sandbox_id: Optional[str] = None) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.error_sandbox_id = error_sandbox_id

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
                if sandbox_id != self.error_sandbox_id
            ],
            "errors": (
                [
                    {
                        "sandbox_id": self.error_sandbox_id,
                        "code": "NOT_FOUND",
                        "message": "Sandbox not found",
                    }
                ]
                if self.error_sandbox_id in sandbox_ids
                else []
            ),
        }


class _AsyncPlatformClient:
    def __init__(self, error_sandbox_id: Optional[str] = None) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.error_sandbox_id = error_sandbox_id

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
                if sandbox_id != self.error_sandbox_id
            ],
            "errors": (
                [
                    {
                        "sandbox_id": self.error_sandbox_id,
                        "code": "NOT_FOUND",
                        "message": "Sandbox not found",
                    }
                ]
                if self.error_sandbox_id in sandbox_ids
                else []
            ),
        }

    async def aclose(self) -> None:
        return None


class _SyncBackgroundJobPlatformClient:
    def __init__(
        self,
        reject_as_container: bool = False,
        error_job_id: Optional[str] = None,
    ) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.reject_as_container = reject_as_container
        self.error_job_id = error_job_id

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
                if job["job_id"] != self.error_job_id
            ],
            "errors": (
                [
                    {
                        **next(job for job in jobs if job["job_id"] == self.error_job_id),
                        "code": "RUNTIME_ERROR",
                        "message": "Runtime lookup failed",
                    }
                ]
                if any(job["job_id"] == self.error_job_id for job in jobs)
                else []
            ),
        }


class _AsyncBackgroundJobPlatformClient:
    def __init__(self, error_job_id: Optional[str] = None) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.error_job_id = error_job_id

    async def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((method, path, kwargs))
        return {
            "statuses": [
                {**job, "completed": False, "exit_code": None}
                for job in kwargs["json"]["jobs"]
                if job["job_id"] != self.error_job_id
            ],
            "errors": (
                [
                    {
                        **next(
                            job
                            for job in kwargs["json"]["jobs"]
                            if job["job_id"] == self.error_job_id
                        ),
                        "code": "RUNTIME_ERROR",
                        "message": "Runtime lookup failed",
                    }
                ]
                if any(job["job_id"] == self.error_job_id for job in kwargs["json"]["jobs"])
                else []
            ),
        }

    async def aclose(self) -> None:
        return None


class _SyncUnsupportedPlatformClient:
    def __init__(self) -> None:
        self.calls = 0

    def request(self, _method: str, _path: str, **_kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        raise APIError("HTTP 405: Method Not Allowed")


class _AsyncUnsupportedPlatformClient:
    def __init__(self) -> None:
        self.calls = 0

    async def request(self, _method: str, _path: str, **_kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        raise APIError("HTTP 405: Method Not Allowed")

    async def aclose(self) -> None:
        return None


class _SyncVMAuthCache:
    def get_or_refresh(self, _sandbox_id: str) -> dict[str, Any]:
        return {}

    def is_vm(self, _sandbox_id: str) -> bool:
        return True


class _AsyncVMAuthCache:
    async def get_or_refresh(self, _sandbox_id: str) -> dict[str, Any]:
        return {}

    async def is_vm(self, _sandbox_id: str) -> bool:
        return True


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


def test_sync_creation_batch_errors_only_fail_the_matching_waiter() -> None:
    client = SandboxClient(APIClient(api_key="test-key"))
    client.client.client.close()
    platform = _SyncPlatformClient(error_sandbox_id="sandbox-b")
    cast(Any, client).client = platform
    cast(Any, client)._is_sandbox_reachable = lambda _sandbox_id: True

    with ThreadPoolExecutor(max_workers=2) as executor:
        running = executor.submit(client.wait_for_creation, "sandbox-a")
        missing = executor.submit(client.wait_for_creation, "sandbox-b")

        assert running.result() is None
        with pytest.raises(APIError, match="sandbox-b: NOT_FOUND"):
            missing.result()

    assert len(platform.calls) == 1


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


@pytest.mark.asyncio
async def test_async_creation_batch_errors_only_fail_the_matching_waiter() -> None:
    client = AsyncSandboxClient(api_key="test-key")
    await client.client.aclose()
    platform = _AsyncPlatformClient(error_sandbox_id="sandbox-b")
    cast(Any, client).client = platform

    async def reachable(_sandbox_id: str) -> bool:
        return True

    cast(Any, client)._is_sandbox_reachable = reachable
    try:
        results = await asyncio.gather(
            client.wait_for_creation("sandbox-a"),
            client.wait_for_creation("sandbox-b"),
            return_exceptions=True,
        )
    finally:
        await client.aclose()

    assert results[0] is None
    assert isinstance(results[1], APIError)
    assert "sandbox-b: NOT_FOUND" in str(results[1])
    assert len(platform.calls) == 1


def test_sync_creation_batch_capability_falls_back_once_per_client() -> None:
    client = SandboxClient(APIClient(api_key="test-key"))
    client.client.client.close()
    platform = _SyncUnsupportedPlatformClient()
    cast(Any, client).client = platform
    cast(Any, client).get = lambda sandbox_id: SimpleNamespace(
        id=sandbox_id,
        status="RUNNING",
        error_type=None,
        error_message=None,
        pending_image_build_id=None,
    )

    for _ in range(2):
        statuses = cast(Any, client)._fetch_sandbox_statuses(["sandbox-a"])
        assert statuses["sandbox-a"].status == "RUNNING"

    assert platform.calls == 1


@pytest.mark.asyncio
async def test_async_creation_batch_capability_falls_back_once_per_client() -> None:
    client = AsyncSandboxClient(api_key="test-key")
    await client.client.aclose()
    platform = _AsyncUnsupportedPlatformClient()
    cast(Any, client).client = platform

    async def get_sandbox(sandbox_id: str):
        return SimpleNamespace(
            id=sandbox_id,
            status="RUNNING",
            error_type=None,
            error_message=None,
            pending_image_build_id=None,
        )

    cast(Any, client).get = get_sandbox
    try:
        for _ in range(2):
            statuses = await cast(Any, client)._fetch_sandbox_statuses(["sandbox-a"])
            assert statuses["sandbox-a"].status == "RUNNING"
    finally:
        await client.aclose()

    assert platform.calls == 1


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


def test_sync_background_batch_capability_falls_back_once_per_client() -> None:
    client = SandboxClient(APIClient(api_key="test-key"))
    client.client.client.close()
    platform = _SyncUnsupportedPlatformClient()
    cast(Any, client).client = platform
    cast(Any, client)._auth_cache = _SyncVMAuthCache()
    cast(Any, client).get_background_job = lambda _sandbox_id, job, timeout=None: (
        BackgroundJobStatus(job_id=job.job_id, completed=False)
    )
    jobs = [_job("sandbox-a", "deadbeef"), _job("sandbox-b", "cafebabe")]

    for _ in range(2):
        statuses = client.get_background_jobs(jobs)
        assert all(not status.completed for status in statuses)

    assert platform.calls == 1


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


def test_sync_background_batch_errors_only_fail_the_matching_waiter() -> None:
    client = SandboxClient(APIClient(api_key="test-key"))
    client.client.client.close()
    platform = _SyncBackgroundJobPlatformClient(error_job_id="cafebabe")
    cast(Any, client).client = platform

    with ThreadPoolExecutor(max_workers=2) as executor:
        running = executor.submit(
            cast(Any, client)._background_job_status_batcher.get,
            ("sandbox-a", "deadbeef"),
        )
        failed = executor.submit(
            cast(Any, client)._background_job_status_batcher.get,
            ("sandbox-b", "cafebabe"),
        )

        assert not running.result().completed
        with pytest.raises(APIError, match="sandbox-b/cafebabe"):
            failed.result()

    assert len(platform.calls) == 1


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


@pytest.mark.asyncio
async def test_async_background_batch_errors_only_fail_the_matching_waiter() -> None:
    client = AsyncSandboxClient(api_key="test-key")
    await client.client.aclose()
    platform = _AsyncBackgroundJobPlatformClient(error_job_id="cafebabe")
    cast(Any, client).client = platform
    try:
        results = await asyncio.gather(
            cast(Any, client)._background_job_status_batcher.get(("sandbox-a", "deadbeef")),
            cast(Any, client)._background_job_status_batcher.get(("sandbox-b", "cafebabe")),
            return_exceptions=True,
        )
    finally:
        await client.aclose()

    assert isinstance(results[0], BackgroundJobStatus)
    assert not results[0].completed
    assert isinstance(results[1], APIError)
    assert "sandbox-b/cafebabe" in str(results[1])
    assert len(platform.calls) == 1


@pytest.mark.asyncio
async def test_async_background_job_cancellation_settles_poll_before_teardown() -> None:
    client = AsyncSandboxClient(api_key="test-key")
    await client.client.aclose()
    cast(Any, client).client = _AsyncUnsupportedPlatformClient()
    cast(Any, client)._auth_cache = _AsyncVMAuthCache()

    poll_started = asyncio.Event()
    release_poll = asyncio.Event()
    sandbox_deleted = False
    polled_after_delete = False

    async def start_background_job(*_args: Any, **_kwargs: Any) -> BackgroundJob:
        return _job("sandbox-a", "deadbeef")

    async def read_file(*_args: Any, **_kwargs: Any) -> ReadFileResponse:
        nonlocal polled_after_delete
        poll_started.set()
        await release_poll.wait()
        polled_after_delete = sandbox_deleted
        return ReadFileResponse(content="", size=0)

    cast(Any, client).start_background_job = start_background_job
    cast(Any, client).read_file = read_file

    run = asyncio.create_task(client.run_background_job("sandbox-a", "sleep 30"))
    await poll_started.wait()

    async def cancel_then_delete() -> None:
        nonlocal sandbox_deleted
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run
        sandbox_deleted = True

    teardown = asyncio.create_task(cancel_then_delete())
    await asyncio.sleep(0)
    assert not sandbox_deleted

    release_poll.set()
    await teardown
    await client.aclose()

    assert sandbox_deleted
    assert not polled_after_delete
