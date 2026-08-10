"""Unit tests for background job output retrieval."""

from pathlib import Path
from typing import Any, List, Optional, cast

import pytest

from prime_sandboxes.core.client import APIClient
from prime_sandboxes.exceptions import SandboxFileTooLargeError
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
    return ReadFileResponse(content=content, size=size, total_size=size, offset=0, truncated=False)


def _legacy_whole_file(content: str) -> ReadFileResponse:
    """Response shape from servers without windowed-read support (VM sandboxes)."""
    return ReadFileResponse.model_validate({"content": content, "size": len(content.encode())})


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


def test_sync_get_background_job_handles_legacy_read_file_response():
    """VM sandboxes ignore offset/length and omit the window metadata fields;
    the truncated flags must default to False rather than fail validation."""

    client = SandboxClient(APIClient(api_key="test-key"))
    client_any = cast(Any, client)

    def fake_read_file(
        sandbox_id: str,
        file_path: str,
        timeout: Optional[int] = None,
        offset: Optional[int] = None,
        length: Optional[int] = None,
    ) -> ReadFileResponse:
        if file_path.endswith(".exit"):
            return _legacy_whole_file("0\n")
        return _legacy_whole_file("out")

    client_any.read_file = fake_read_file

    status = client.get_background_job("sbx-123", _make_job())

    assert status.completed
    assert status.exit_code == 0
    assert status.stdout == "out"
    assert status.stderr == "out"
    assert status.stdout_truncated is False
    assert status.stderr_truncated is False


def test_sync_get_background_job_downloads_large_output_tail(monkeypatch):
    client = SandboxClient(APIClient(api_key="test-key"))
    client_any = cast(Any, client)
    monkeypatch.setattr("prime_sandboxes.sandbox.JOB_OUTPUT_TAIL_BYTES", 6)

    def fake_read_file(
        sandbox_id: str,
        file_path: str,
        timeout: Optional[int] = None,
        offset: Optional[int] = None,
        length: Optional[int] = None,
    ) -> ReadFileResponse:
        if file_path.endswith(".exit"):
            return _whole_file("0\n")
        raise SandboxFileTooLargeError("output too large")

    downloads = []

    def fake_download_file(
        sandbox_id: str,
        file_path: str,
        local_file_path: str,
        timeout: Optional[int] = None,
    ) -> None:
        downloads.append((file_path, timeout))
        content = b"prefixstdout" if file_path.endswith(".stdout") else b"prefixstderr"
        Path(local_file_path).write_bytes(content)

    client_any.read_file = fake_read_file
    client_any.download_file = fake_download_file

    status = client.get_background_job("sbx-123", _make_job(), timeout=45)

    assert status.stdout == "stdout"
    assert status.stderr == "stderr"
    assert status.stdout_truncated is True
    assert status.stderr_truncated is True
    assert downloads == [("/tmp/job_abc.stdout", 45), ("/tmp/job_abc.stderr", 45)]


@pytest.mark.asyncio
async def test_async_get_background_job_handles_legacy_read_file_response():
    client = AsyncSandboxClient(api_key="test-key")
    client_any = cast(Any, client)

    async def fake_read_file(
        sandbox_id: str,
        file_path: str,
        timeout: Optional[int] = None,
        offset: Optional[int] = None,
        length: Optional[int] = None,
    ) -> ReadFileResponse:
        if file_path.endswith(".exit"):
            return _legacy_whole_file("0\n")
        return _legacy_whole_file("out")

    client_any.read_file = fake_read_file

    status = await client.get_background_job("sbx-123", _make_job())

    assert status.completed
    assert status.exit_code == 0
    assert status.stdout == "out"
    assert status.stdout_truncated is False
    assert status.stderr_truncated is False


@pytest.mark.asyncio
async def test_async_get_background_job_downloads_large_output_tail(monkeypatch):
    client = AsyncSandboxClient(api_key="test-key")
    client_any = cast(Any, client)
    monkeypatch.setattr("prime_sandboxes.sandbox.JOB_OUTPUT_TAIL_BYTES", 6)

    async def fake_read_file(
        sandbox_id: str,
        file_path: str,
        timeout: Optional[int] = None,
        offset: Optional[int] = None,
        length: Optional[int] = None,
    ) -> ReadFileResponse:
        if file_path.endswith(".exit"):
            return _whole_file("0\n")
        raise SandboxFileTooLargeError("output too large")

    downloads = []

    async def fake_download_file(
        sandbox_id: str,
        file_path: str,
        local_file_path: str,
        timeout: Optional[int] = None,
    ) -> None:
        downloads.append((file_path, timeout))
        content = b"prefixstdout" if file_path.endswith(".stdout") else b"prefixstderr"
        Path(local_file_path).write_bytes(content)

    client_any.read_file = fake_read_file
    client_any.download_file = fake_download_file

    status = await client.get_background_job("sbx-123", _make_job(), timeout=45)

    assert status.stdout == "stdout"
    assert status.stderr == "stderr"
    assert status.stdout_truncated is True
    assert status.stderr_truncated is True
    assert downloads == [("/tmp/job_abc.stdout", 45), ("/tmp/job_abc.stderr", 45)]


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
