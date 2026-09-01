"""Focused tests for platform lifecycle and VM background-job batch calls."""

import asyncio
import errno
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, Optional, cast

import pytest

from prime_sandboxes import BatchStatusUnsupportedError
from prime_sandboxes.core.client import APIClient, APIError
from prime_sandboxes.models import (
    BackgroundJob,
    BackgroundJobStatus,
    BackgroundJobStatusSnapshot,
    ReadFileResponse,
)
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
    def __init__(
        self,
        error_job_id: Optional[str] = None,
        complete_all: bool = False,
    ) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.error_job_id = error_job_id
        self.complete_all = complete_all

    async def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((method, path, kwargs))
        return {
            "statuses": [
                {
                    **job,
                    "completed": self.complete_all,
                    "exit_code": 0 if self.complete_all else None,
                }
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


class _SyncDeletePlatformClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.delete_called = threading.Event()

    def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((method, path, kwargs))
        if method == "DELETE":
            self.delete_called.set()
        if path == "/sandbox":
            return {
                "succeeded": kwargs["json"].get("sandbox_ids") or [],
                "failed": [],
                "message": "",
            }
        return {}


class _AsyncDeletePlatformClient:
    def __init__(self, close_order: Optional[list[str]] = None) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.delete_called = asyncio.Event()
        self.close_order = close_order

    async def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((method, path, kwargs))
        if method == "DELETE":
            self.delete_called.set()
        if path == "/sandbox":
            return {
                "succeeded": kwargs["json"].get("sandbox_ids") or [],
                "failed": [],
                "message": "",
            }
        return {}

    async def aclose(self) -> None:
        if self.close_order is not None:
            self.close_order.append("platform-close")


class _OrderedAsyncGatewayClient:
    def __init__(self, close_order: list[str]) -> None:
        self.close_order = close_order

    async def aclose(self) -> None:
        self.close_order.append("gateway-close")


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
    cast(Any, client)._get_background_job_status_unleased = (
        lambda sandbox_id, job, timeout=None: BackgroundJobStatusSnapshot(
            sandbox_id=sandbox_id,
            job_id=job.job_id,
            completed=False,
        )
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
async def test_async_completed_output_reads_are_client_bounded_and_sequential() -> None:
    client = AsyncSandboxClient(api_key="test-key")
    await client.client.aclose()
    platform = _AsyncBackgroundJobPlatformClient(complete_all=True)
    cast(Any, client).client = platform

    active_reads = 0
    peak_reads = 0
    paths_by_sandbox: dict[str, list[str]] = {}

    async def read_file(
        sandbox_id: str,
        path: str,
        timeout: Optional[int] = None,
        offset: Optional[int] = None,
        length: Optional[int] = None,
    ) -> ReadFileResponse:
        nonlocal active_reads, peak_reads
        paths_by_sandbox.setdefault(sandbox_id, []).append(path)
        active_reads += 1
        peak_reads = max(peak_reads, active_reads)
        try:
            await asyncio.sleep(0.01)
            return ReadFileResponse(content=path, size=len(path), truncated=False)
        finally:
            active_reads -= 1

    cast(Any, client).read_file = read_file
    jobs = [_job(f"sandbox-{index}", f"{index:08x}") for index in range(100)]
    try:
        statuses = await client.get_background_jobs(jobs)
    finally:
        await client.aclose()

    assert all(status.completed and status.exit_code == 0 for status in statuses)
    assert peak_reads == 20
    assert sum(len(paths) for paths in paths_by_sandbox.values()) == 200
    for job in jobs:
        assert paths_by_sandbox[job.sandbox_id] == [
            job.stdout_log_file,
            job.stderr_log_file,
        ]


@pytest.mark.asyncio
async def test_async_output_error_preserves_completion_and_surfaces_errno() -> None:
    client = AsyncSandboxClient(api_key="test-key")
    await client.client.aclose()
    cast(Any, client).client = _AsyncBackgroundJobPlatformClient(complete_all=True)

    async def read_file(
        _sandbox_id: str,
        path: str,
        timeout: Optional[int] = None,
        offset: Optional[int] = None,
        length: Optional[int] = None,
    ) -> ReadFileResponse:
        if path.endswith("stdout.log"):
            try:
                try:
                    raise OSError(errno.EMFILE, "Too many open files")
                except OSError as os_error:
                    raise RuntimeError("All connection attempts failed") from os_error
            except RuntimeError as transport_error:
                raise APIError("Read file failed: ConnectError") from transport_error
        return ReadFileResponse(content="stderr", size=6, truncated=False)

    cast(Any, client).read_file = read_file
    try:
        status = (await client.get_background_jobs([_job("sandbox-a", "feedface")]))[0]
    finally:
        await client.aclose()

    assert status.completed
    assert status.exit_code == 0
    assert status.stdout is None
    assert status.stdout_error is not None
    assert "errno=24" in status.stdout_error
    assert "Too many open files" in status.stdout_error
    assert status.stderr == "stderr"
    assert status.stderr_error is None


@pytest.mark.asyncio
async def test_async_output_deadline_preserves_completion() -> None:
    client = AsyncSandboxClient(api_key="test-key")
    await client.client.aclose()
    cast(Any, client).client = _AsyncBackgroundJobPlatformClient(complete_all=True)

    async def unexpected_read(*_args: Any, **_kwargs: Any) -> ReadFileResponse:
        raise AssertionError("expired output deadline must not start a read")

    cast(Any, client).read_file = unexpected_read
    try:
        status = (
            await client.get_background_jobs(
                [_job("sandbox-a", "feedface")],
                timeout=0,
            )
        )[0]
    finally:
        await client.aclose()

    assert status.completed
    assert status.exit_code == 0
    assert status.stdout is None
    assert status.stderr is None
    assert status.stdout_error == "Output retrieval deadline exceeded after 0s"
    assert status.stderr_error == "Output retrieval deadline exceeded after 0s"


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

    assert isinstance(results[0], BackgroundJobStatusSnapshot)
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


@pytest.mark.asyncio
async def test_async_cancellation_before_dispatch_removes_lookup() -> None:
    client = AsyncSandboxClient(api_key="test-key")
    fetch_calls = 0

    async def fetch(_keys: list[tuple[str, str]]) -> dict[tuple[str, str], BackgroundJobStatus]:
        nonlocal fetch_calls
        fetch_calls += 1
        return {}

    batcher = cast(Any, client)._background_job_status_batcher
    batcher._fetch = fetch
    lookup = asyncio.create_task(batcher.get(("sandbox-a", "deadbeef")))
    await asyncio.sleep(0)
    lookup.cancel()
    with pytest.raises(asyncio.CancelledError):
        await lookup
    await client.aclose()

    assert fetch_calls == 0
    assert not cast(Any, client)._poll_leases._active


@pytest.mark.asyncio
async def test_async_dispatch_cancellation_settles_pending_lookup() -> None:
    client = AsyncSandboxClient(api_key="test-key")
    batcher = cast(Any, client)._background_job_status_batcher
    lookup = asyncio.create_task(batcher.get(("sandbox-a", "deadbeef")))
    await asyncio.sleep(0)

    batcher._dispatch_task.cancel()
    with pytest.raises(RuntimeError, match="dispatch cancelled"):
        await lookup
    assert not cast(Any, client)._poll_leases._active
    await client.aclose()


@pytest.mark.asyncio
async def test_async_last_waiter_cancels_and_joins_fetch_cleanup() -> None:
    client = AsyncSandboxClient(api_key="test-key")
    fetch_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()

    async def fetch(_keys: list[tuple[str, str]]) -> dict[tuple[str, str], BackgroundJobStatus]:
        fetch_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await release_cleanup.wait()

    batcher = cast(Any, client)._background_job_status_batcher
    batcher._fetch = fetch
    lookup = asyncio.create_task(batcher.get(("sandbox-a", "deadbeef")))
    await fetch_started.wait()

    lookup.cancel()
    await cleanup_started.wait()
    await asyncio.sleep(0)
    assert not lookup.done()

    release_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await lookup
    assert not cast(Any, client)._poll_leases._active
    await client.aclose()


@pytest.mark.asyncio
async def test_async_shared_fetch_survives_one_cancellation_and_blocks_delete() -> None:
    client = AsyncSandboxClient(api_key="test-key")
    await client.client.aclose()
    platform = _AsyncDeletePlatformClient()
    cast(Any, client).client = platform
    fetch_started = asyncio.Event()
    release_fetch = asyncio.Event()
    fetch_cancelled = False

    async def fetch(
        keys: list[tuple[str, str]],
    ) -> dict[tuple[str, str], BackgroundJobStatus]:
        nonlocal fetch_cancelled
        fetch_started.set()
        try:
            await release_fetch.wait()
        except asyncio.CancelledError:
            fetch_cancelled = True
            raise
        return {key: BackgroundJobStatus(job_id=key[1], completed=False) for key in keys}

    batcher = cast(Any, client)._background_job_status_batcher
    batcher._fetch = fetch
    cancelled = asyncio.create_task(batcher.get(("sandbox-a", "deadbeef")))
    surviving = asyncio.create_task(batcher.get(("sandbox-b", "cafebabe")))
    await fetch_started.wait()

    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(cancelled, timeout=1)
    assert not fetch_cancelled

    deleting = asyncio.create_task(client.delete("sandbox-a"))
    while "sandbox-a" not in cast(Any, client)._poll_leases._draining:
        await asyncio.sleep(0)
    with pytest.raises(APIError, match="being deleted"):
        await batcher.get(("sandbox-a", "feedface"))
    assert not platform.delete_called.is_set()

    release_fetch.set()
    status = await surviving
    await deleting
    assert not status.completed
    assert platform.delete_called.is_set()
    await client.aclose()


def test_sync_delete_waits_for_in_flight_poll() -> None:
    client = SandboxClient(APIClient(api_key="test-key"))
    client.client.client.close()
    platform = _SyncDeletePlatformClient()
    cast(Any, client).client = platform
    fetch_started = threading.Event()
    release_fetch = threading.Event()

    def fetch(
        keys: list[tuple[str, str]],
    ) -> dict[tuple[str, str], BackgroundJobStatus]:
        fetch_started.set()
        assert release_fetch.wait(timeout=2)
        return {key: BackgroundJobStatus(job_id=key[1], completed=False) for key in keys}

    cast(Any, client)._background_job_status_batcher._fetch = fetch
    with ThreadPoolExecutor(max_workers=2) as executor:
        lookup = executor.submit(
            cast(Any, client)._background_job_status_batcher.get,
            ("sandbox-a", "deadbeef"),
        )
        assert fetch_started.wait(timeout=2)
        deleting = executor.submit(client.delete, "sandbox-a")
        assert not platform.delete_called.wait(timeout=0.05)
        release_fetch.set()
        assert not lookup.result(timeout=2).completed
        assert deleting.result(timeout=2) == {}

    assert platform.delete_called.is_set()


def test_sync_interrupted_drain_rolls_back_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    client = SandboxClient(APIClient(api_key="test-key"))
    client.client.client.close()
    registry = cast(Any, client)._poll_leases
    active_lease = registry.acquire("sandbox-a")

    def interrupt_wait() -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(registry._condition, "wait", interrupt_wait)
    with pytest.raises(KeyboardInterrupt):
        registry.begin_drain(["sandbox-a"])

    assert "sandbox-a" not in registry._draining
    next_lease = registry.acquire("sandbox-a")
    next_lease.release()
    active_lease.release()

    scopes = registry.begin_drain(["sandbox-a"])
    registry.end_drain(scopes)
    assert "sandbox-a" not in registry._draining


@pytest.mark.asyncio
async def test_async_explicit_bulk_delete_drains_known_sandboxes() -> None:
    client = AsyncSandboxClient(api_key="test-key")
    await client.client.aclose()
    platform = _AsyncDeletePlatformClient()
    cast(Any, client).client = platform
    fetch_started = asyncio.Event()
    release_fetch = asyncio.Event()

    async def fetch(
        keys: list[tuple[str, str]],
    ) -> dict[tuple[str, str], BackgroundJobStatus]:
        fetch_started.set()
        await release_fetch.wait()
        return {key: BackgroundJobStatus(job_id=key[1], completed=False) for key in keys}

    batcher = cast(Any, client)._background_job_status_batcher
    batcher._fetch = fetch
    lookup = asyncio.create_task(batcher.get(("sandbox-a", "deadbeef")))
    await fetch_started.wait()
    deleting = asyncio.create_task(client.bulk_delete(sandbox_ids=["sandbox-a"]))
    await asyncio.sleep(0)
    assert not platform.delete_called.is_set()

    release_fetch.set()
    await lookup
    response = await deleting
    assert response.succeeded == ["sandbox-a"]
    await client.aclose()


@pytest.mark.asyncio
async def test_async_close_joins_fetch_before_transports_even_when_cancelled() -> None:
    client = AsyncSandboxClient(api_key="test-key")
    await client.client.aclose()
    close_order: list[str] = []
    cast(Any, client).client = _AsyncDeletePlatformClient(close_order)
    cast(Any, client)._gateway_client = _OrderedAsyncGatewayClient(close_order)
    fetch_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()

    async def fetch(_keys: list[str]) -> dict[str, Any]:
        fetch_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await release_cleanup.wait()
            close_order.append("fetch-close")

    batcher = cast(Any, client)._sandbox_status_batcher
    batcher._fetch = fetch
    lookup = asyncio.create_task(batcher.get("sandbox-a"))
    await fetch_started.wait()

    closing = asyncio.create_task(client.aclose())
    await cleanup_started.wait()
    closing.cancel()
    await asyncio.sleep(0)
    assert not closing.done()
    assert close_order == []

    release_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await closing
    with pytest.raises(RuntimeError, match="closed"):
        await lookup
    assert close_order == ["fetch-close", "gateway-close", "platform-close"]

    await client.aclose()


def test_sync_completed_output_is_deduplicated_and_cached() -> None:
    client = SandboxClient(APIClient(api_key="test-key"))
    job = _job("sandbox-a", "deadbeef")
    calls: list[str] = []
    first_read_started = threading.Event()
    release_reads = threading.Event()

    def read_file(
        _sandbox_id: str,
        path: str,
        timeout: Optional[int] = None,
        offset: Optional[int] = None,
        length: Optional[int] = None,
    ) -> ReadFileResponse:
        calls.append(path)
        if len(calls) == 1:
            first_read_started.set()
            assert release_reads.wait(timeout=2)
        return ReadFileResponse(content=path, size=len(path), truncated=False)

    cast(Any, client).read_file = read_file
    coordinator = cast(Any, client)._background_job_output_coordinator
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(coordinator.get, job, 0, None)
        assert first_read_started.wait(timeout=2)
        second = executor.submit(coordinator.get, job, 0, None)
        release_reads.set()
        assert first.result(timeout=2).stdout == job.stdout_log_file
        assert second.result(timeout=2).stderr == job.stderr_log_file

    cached = coordinator.get(job, 0, None)
    assert cached.stdout == job.stdout_log_file
    assert calls == [job.stdout_log_file, job.stderr_log_file]


def test_sync_bulk_output_hydration_uses_configured_global_limit() -> None:
    client = SandboxClient(
        APIClient(api_key="test-key"),
        background_job_output_concurrency=2,
    )
    jobs = [_job(f"sandbox-{index}", f"{index:08x}") for index in range(6)]
    snapshots = [
        BackgroundJobStatusSnapshot(
            sandbox_id=job.sandbox_id,
            job_id=job.job_id,
            completed=True,
            exit_code=0,
        )
        for job in jobs
    ]
    cast(Any, client).get_background_job_statuses = lambda _jobs, timeout=None: snapshots
    lock = threading.Lock()
    active = 0
    peak = 0

    def read_file(
        _sandbox_id: str,
        path: str,
        timeout: Optional[int] = None,
        offset: Optional[int] = None,
        length: Optional[int] = None,
    ) -> ReadFileResponse:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.01)
            return ReadFileResponse(content=path, size=len(path), truncated=False)
        finally:
            with lock:
                active -= 1

    cast(Any, client).read_file = read_file
    statuses = client.get_background_jobs(jobs)
    assert all(status.completed for status in statuses)
    assert peak == 2


@pytest.mark.asyncio
async def test_async_status_only_batch_never_downloads_output() -> None:
    client = AsyncSandboxClient(api_key="test-key")
    await client.client.aclose()
    cast(Any, client).client = _AsyncBackgroundJobPlatformClient(complete_all=True)
    reads = 0

    async def read_file(*_args: Any, **_kwargs: Any) -> ReadFileResponse:
        nonlocal reads
        reads += 1
        raise AssertionError("status-only lookup must not download output")

    cast(Any, client).read_file = read_file
    snapshots = await client.get_background_job_statuses(
        [_job("sandbox-a", "deadbeef"), _job("sandbox-b", "cafebabe")]
    )
    assert all(snapshot.completed and snapshot.exit_code == 0 for snapshot in snapshots)
    assert reads == 0
    await client.aclose()


@pytest.mark.asyncio
async def test_async_output_hydration_does_not_block_later_status_batches() -> None:
    client = AsyncSandboxClient(api_key="test-key")
    await client.client.aclose()
    cast(Any, client).client = _AsyncBackgroundJobPlatformClient(complete_all=True)
    output_started = asyncio.Event()
    release_output = asyncio.Event()

    async def read_file(
        _sandbox_id: str,
        path: str,
        timeout: Optional[int] = None,
        offset: Optional[int] = None,
        length: Optional[int] = None,
    ) -> ReadFileResponse:
        output_started.set()
        await release_output.wait()
        return ReadFileResponse(content=path, size=len(path), truncated=False)

    cast(Any, client).read_file = read_file
    hydration = asyncio.create_task(client.get_background_jobs([_job("sandbox-a", "deadbeef")]))
    await output_started.wait()

    snapshot = await asyncio.wait_for(
        cast(Any, client)._background_job_status_batcher.get(("sandbox-b", "cafebabe")),
        timeout=0.5,
    )
    assert snapshot.completed
    release_output.set()
    assert (await hydration)[0].completed
    await client.aclose()


@pytest.mark.asyncio
async def test_async_completed_output_is_deduplicated_and_partially_cached() -> None:
    client = AsyncSandboxClient(api_key="test-key")
    await client.client.aclose()
    cast(Any, client).client = _AsyncBackgroundJobPlatformClient()
    job = _job("sandbox-a", "deadbeef")
    calls: list[str] = []
    stderr_attempts = 0

    async def read_file(
        _sandbox_id: str,
        path: str,
        timeout: Optional[int] = None,
        offset: Optional[int] = None,
        length: Optional[int] = None,
    ) -> ReadFileResponse:
        nonlocal stderr_attempts
        calls.append(path)
        await asyncio.sleep(0.01)
        if path == job.stderr_log_file:
            stderr_attempts += 1
            if stderr_attempts == 1:
                raise APIError("temporary stderr failure")
        return ReadFileResponse(content=path, size=len(path), truncated=False)

    cast(Any, client).read_file = read_file
    coordinator = cast(Any, client)._background_job_output_coordinator
    first, shared = await asyncio.gather(
        coordinator.get(job, 0, None),
        coordinator.get(job, 0, None),
    )
    assert first.stdout == job.stdout_log_file
    assert shared.stderr_error == "APIError: temporary stderr failure"

    retried = await coordinator.get(job, 0, None)
    assert retried.stdout == job.stdout_log_file
    assert retried.stderr == job.stderr_log_file
    assert calls == [job.stdout_log_file, job.stderr_log_file, job.stderr_log_file]
    await client.aclose()


@pytest.mark.asyncio
async def test_async_sole_output_waiter_cancels_and_joins_fetch() -> None:
    client = AsyncSandboxClient(api_key="test-key")
    await client.client.aclose()
    cast(Any, client).client = _AsyncBackgroundJobPlatformClient()
    fetch_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()

    async def read_file(*_args: Any, **_kwargs: Any) -> ReadFileResponse:
        fetch_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await release_cleanup.wait()

    cast(Any, client).read_file = read_file
    task = asyncio.create_task(
        cast(Any, client)._background_job_output_coordinator.get(
            _job("sandbox-a", "deadbeef"), 0, None
        )
    )
    await fetch_started.wait()
    task.cancel()
    await cleanup_started.wait()
    await asyncio.sleep(0)
    assert not task.done()

    release_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not cast(Any, client)._operation_leases._active
    await client.aclose()


@pytest.mark.asyncio
async def test_async_shared_output_survives_one_waiter_cancellation() -> None:
    client = AsyncSandboxClient(api_key="test-key")
    await client.client.aclose()
    cast(Any, client).client = _AsyncBackgroundJobPlatformClient()
    fetch_started = asyncio.Event()
    release_fetch = asyncio.Event()
    fetch_cancelled = False
    job = _job("sandbox-a", "deadbeef")

    async def read_file(
        _sandbox_id: str,
        path: str,
        timeout: Optional[int] = None,
        offset: Optional[int] = None,
        length: Optional[int] = None,
    ) -> ReadFileResponse:
        nonlocal fetch_cancelled
        fetch_started.set()
        try:
            await release_fetch.wait()
        except asyncio.CancelledError:
            fetch_cancelled = True
            raise
        return ReadFileResponse(content=path, size=len(path), truncated=False)

    cast(Any, client).read_file = read_file
    coordinator = cast(Any, client)._background_job_output_coordinator
    cancelled = asyncio.create_task(coordinator.get(job, 0, None))
    surviving = asyncio.create_task(coordinator.get(job, 0, None))
    await fetch_started.wait()
    while next(iter(coordinator._inflight.values())).waiters != 2:
        await asyncio.sleep(0)

    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    assert not fetch_cancelled

    release_fetch.set()
    status = await surviving
    assert status.stdout == job.stdout_log_file
    assert not fetch_cancelled
    await client.aclose()


@pytest.mark.asyncio
async def test_async_delete_cancels_queued_output_and_waits_active_output() -> None:
    client = AsyncSandboxClient(
        api_key="test-key",
        background_job_output_concurrency=1,
    )
    await client.client.aclose()
    platform = _AsyncDeletePlatformClient()
    cast(Any, client).client = platform
    active_started = asyncio.Event()
    release_active = asyncio.Event()
    active_job = _job("sandbox-a", "deadbeef")
    queued_job = _job("sandbox-a", "cafebabe")

    async def read_file(
        _sandbox_id: str,
        path: str,
        timeout: Optional[int] = None,
        offset: Optional[int] = None,
        length: Optional[int] = None,
    ) -> ReadFileResponse:
        if path == active_job.stdout_log_file:
            active_started.set()
            await release_active.wait()
        return ReadFileResponse(content=path, size=len(path), truncated=False)

    cast(Any, client).read_file = read_file
    coordinator = cast(Any, client)._background_job_output_coordinator
    active = asyncio.create_task(coordinator.get(active_job, 0, None))
    await active_started.wait()
    queued = asyncio.create_task(coordinator.get(queued_job, 0, None))
    while coordinator._pending_count != 1:
        await asyncio.sleep(0)

    deleting = asyncio.create_task(client.delete("sandbox-a"))
    queued_status = await asyncio.wait_for(queued, timeout=0.5)
    assert queued_status.completed
    assert queued_status.stdout_error is not None
    assert "being deleted" in queued_status.stdout_error
    assert not platform.delete_called.is_set()

    release_active.set()
    assert (await active).completed
    await deleting
    assert platform.delete_called.is_set()
    await client.aclose()


@pytest.mark.asyncio
async def test_async_output_queue_schedules_sandboxes_round_robin() -> None:
    client = AsyncSandboxClient(
        api_key="test-key",
        background_job_output_concurrency=1,
    )
    await client.client.aclose()
    cast(Any, client).client = _AsyncBackgroundJobPlatformClient()
    release_first = asyncio.Event()
    first_started = asyncio.Event()
    stdout_order: list[str] = []

    async def read_file(
        _sandbox_id: str,
        path: str,
        timeout: Optional[int] = None,
        offset: Optional[int] = None,
        length: Optional[int] = None,
    ) -> ReadFileResponse:
        if path.endswith("stdout.log"):
            stdout_order.append(path)
            if len(stdout_order) == 1:
                first_started.set()
                await release_first.wait()
        return ReadFileResponse(content=path, size=len(path), truncated=False)

    cast(Any, client).read_file = read_file
    coordinator = cast(Any, client)._background_job_output_coordinator
    a1 = _job("sandbox-a", "00000001")
    a2 = _job("sandbox-a", "00000002")
    b1 = _job("sandbox-b", "00000003")
    a3 = _job("sandbox-a", "00000004")
    tasks = [asyncio.create_task(coordinator.get(a1, 0, None))]
    await first_started.wait()
    tasks.extend(asyncio.create_task(coordinator.get(job, 0, None)) for job in (a2, b1, a3))
    while coordinator._pending_count != 3:
        await asyncio.sleep(0)
    release_first.set()
    await asyncio.gather(*tasks)

    assert stdout_order == [
        a1.stdout_log_file,
        a2.stdout_log_file,
        b1.stdout_log_file,
        a3.stdout_log_file,
    ]
    await client.aclose()


@pytest.mark.asyncio
async def test_async_close_joins_output_before_transports() -> None:
    client = AsyncSandboxClient(api_key="test-key")
    await client.client.aclose()
    close_order: list[str] = []
    cast(Any, client).client = _AsyncDeletePlatformClient(close_order)
    cast(Any, client)._gateway_client = _OrderedAsyncGatewayClient(close_order)
    fetch_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()

    async def read_file(*_args: Any, **_kwargs: Any) -> ReadFileResponse:
        fetch_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await release_cleanup.wait()
            close_order.append("output-close")

    cast(Any, client).read_file = read_file
    lookup = asyncio.create_task(
        cast(Any, client)._background_job_output_coordinator.get(
            _job("sandbox-a", "deadbeef"), 0, None
        )
    )
    await fetch_started.wait()
    closing = asyncio.create_task(client.aclose())
    await cleanup_started.wait()
    assert close_order == []

    release_cleanup.set()
    await closing
    with pytest.raises(RuntimeError, match="closed"):
        await lookup
    assert close_order == ["output-close", "gateway-close", "platform-close"]
