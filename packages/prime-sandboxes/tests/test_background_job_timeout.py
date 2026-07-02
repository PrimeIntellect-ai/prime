"""Unit tests verifying that get_background_job forwards the timeout kwarg."""

from typing import Any, List, Optional, cast

import pytest

from prime_sandboxes.core.client import APIClient
from prime_sandboxes.models import BackgroundJob, ReadFileResponse
from prime_sandboxes.sandbox import AsyncSandboxClient, SandboxClient


def _make_job() -> BackgroundJob:
    return BackgroundJob(
        job_id="job-123",
        sandbox_id="sbx-123",
        stdout_log_file="/tmp/job_abc.stdout",
        stderr_log_file="/tmp/job_abc.stderr",
        exit_file="/tmp/job_abc.exit",
    )


def _whole_file(content: str) -> ReadFileResponse:
    size = len(content.encode())
    return ReadFileResponse(
        content=content, size=size, total_size=size, offset=0, truncated=False
    )


def test_sync_get_background_job_forwards_timeout_to_read_file():
    client = SandboxClient(APIClient(api_key="test-key"))
    client_any = cast(Any, client)

    seen_timeouts: List[Optional[int]] = []

    def fake_read_file(
        sandbox_id: str,
        file_path: str,
        timeout: Optional[int] = None,
        offset: Optional[int] = None,
        length: Optional[int] = None,
    ) -> ReadFileResponse:
        seen_timeouts.append(timeout)
        # Empty content => job not completed; single read_file invocation is enough.
        return _whole_file("")

    client_any.read_file = fake_read_file

    job = _make_job()
    status = client.get_background_job("sbx-123", job, timeout=60)

    assert not status.completed
    assert seen_timeouts == [60]


def test_sync_get_background_job_defaults_timeout_to_none():
    client = SandboxClient(APIClient(api_key="test-key"))
    client_any = cast(Any, client)

    seen_timeouts: List[Optional[int]] = []

    def fake_read_file(
        sandbox_id: str,
        file_path: str,
        timeout: Optional[int] = None,
        offset: Optional[int] = None,
        length: Optional[int] = None,
    ) -> ReadFileResponse:
        seen_timeouts.append(timeout)
        return _whole_file("")

    client_any.read_file = fake_read_file

    job = _make_job()
    client.get_background_job("sbx-123", job)

    assert seen_timeouts == [None]


def test_sync_get_background_job_forwards_timeout_on_completed_reads():
    """When the exit file has content, stdout and stderr are also read - verify
    all three read_file calls receive the same timeout."""

    client = SandboxClient(APIClient(api_key="test-key"))
    client_any = cast(Any, client)

    seen_timeouts: List[Optional[int]] = []

    def fake_read_file(
        sandbox_id: str,
        file_path: str,
        timeout: Optional[int] = None,
        offset: Optional[int] = None,
        length: Optional[int] = None,
    ) -> ReadFileResponse:
        seen_timeouts.append(timeout)
        if file_path.endswith(".exit"):
            return _whole_file("0\n")
        return _whole_file("out")

    client_any.read_file = fake_read_file

    job = _make_job()
    status = client.get_background_job("sbx-123", job, timeout=45)

    assert status.completed
    assert status.exit_code == 0
    # Exit file read, then stdout, then stderr.
    assert seen_timeouts == [45, 45, 45]


@pytest.mark.asyncio
async def test_async_get_background_job_forwards_timeout_to_read_file():
    client = AsyncSandboxClient(api_key="test-key")
    client_any = cast(Any, client)

    seen_timeouts: List[Optional[int]] = []

    async def fake_read_file(
        sandbox_id: str,
        file_path: str,
        timeout: Optional[int] = None,
        offset: Optional[int] = None,
        length: Optional[int] = None,
    ) -> ReadFileResponse:
        seen_timeouts.append(timeout)
        return _whole_file("")

    client_any.read_file = fake_read_file

    job = _make_job()
    status = await client.get_background_job("sbx-123", job, timeout=60)

    assert not status.completed
    assert seen_timeouts == [60]


@pytest.mark.asyncio
async def test_async_get_background_job_defaults_timeout_to_none():
    client = AsyncSandboxClient(api_key="test-key")
    client_any = cast(Any, client)

    seen_timeouts: List[Optional[int]] = []

    async def fake_read_file(
        sandbox_id: str,
        file_path: str,
        timeout: Optional[int] = None,
        offset: Optional[int] = None,
        length: Optional[int] = None,
    ) -> ReadFileResponse:
        seen_timeouts.append(timeout)
        return _whole_file("")

    client_any.read_file = fake_read_file

    job = _make_job()
    await client.get_background_job("sbx-123", job)

    assert seen_timeouts == [None]
