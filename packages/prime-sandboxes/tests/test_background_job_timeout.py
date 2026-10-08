"""Unit tests verifying that get_background_job forwards the timeout kwarg."""

from typing import Any, List, Optional, cast

import pytest

from prime_sandboxes.core.client import APIClient, APIError
from prime_sandboxes.models import BackgroundJob, CommandResponse, ReadFileResponse
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


def _probe(stdout: str, seen_timeouts: Optional[List[Optional[int]]] = None):
    """Fake execute_command answering the background-job status probe."""

    def execute_command(_sandbox_id: str, _command: str, timeout: Optional[int] = None, **_kw):
        if seen_timeouts is not None:
            seen_timeouts.append(timeout)
        return CommandResponse(stdout=stdout, stderr="", exit_code=0)

    return execute_command


def _async_probe(stdout: str, seen_timeouts: Optional[List[Optional[int]]] = None):
    probe = _probe(stdout, seen_timeouts)

    async def execute_command(*args: Any, **kwargs: Any) -> CommandResponse:
        return probe(*args, **kwargs)

    return execute_command


def test_sync_get_background_job_forwards_timeout_to_status_probe():
    client = SandboxClient(APIClient(api_key="test-key"))
    client_any = cast(Any, client)

    seen_timeouts: List[Optional[int]] = []

    client_any.execute_command = _probe("", seen_timeouts)

    job = _make_job()
    status = client.get_background_job("sbx-123", job, timeout=60)

    assert not status.completed
    assert seen_timeouts == [60]


def test_sync_get_background_job_defaults_timeout_to_none():
    client = SandboxClient(APIClient(api_key="test-key"))
    client_any = cast(Any, client)

    seen_timeouts: List[Optional[int]] = []

    client_any.execute_command = _probe("", seen_timeouts)

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
        return _whole_file("out")

    client_any.read_file = fake_read_file
    client_any.execute_command = _probe("0\n", seen_timeouts)

    job = _make_job()
    status = client.get_background_job("sbx-123", job, timeout=45)

    assert status.completed
    assert status.exit_code == 0
    # Status probe, then stdout, then stderr.
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
        return _legacy_whole_file("out")

    client_any.read_file = fake_read_file
    client_any.execute_command = _probe("0\n")

    status = client.get_background_job("sbx-123", _make_job())

    assert status.completed
    assert status.exit_code == 0
    assert status.stdout == "out"
    assert status.stderr == "out"
    assert status.stdout_truncated is False
    assert status.stderr_truncated is False


def test_sync_output_error_preserves_completed_exit_code():
    client = SandboxClient(APIClient(api_key="test-key"))
    client_any = cast(Any, client)

    def fake_read_file(
        sandbox_id: str,
        file_path: str,
        timeout: Optional[int] = None,
        offset: Optional[int] = None,
        length: Optional[int] = None,
    ) -> ReadFileResponse:
        if file_path.endswith(".stdout"):
            raise APIError("Read file failed: ConnectError")
        return _whole_file("stderr")

    client_any.read_file = fake_read_file
    client_any.execute_command = _probe("7\n")

    status = client.get_background_job("sbx-123", _make_job())

    assert status.completed
    assert status.exit_code == 7
    assert status.stdout is None
    assert status.stdout_error == "APIError: Read file failed: ConnectError"
    assert status.stderr == "stderr"
    assert status.stderr_error is None


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
        return _legacy_whole_file("out")

    client_any.read_file = fake_read_file
    client_any.execute_command = _async_probe("0\n")

    status = await client.get_background_job("sbx-123", _make_job())

    assert status.completed
    assert status.exit_code == 0
    assert status.stdout == "out"
    assert status.stdout_truncated is False
    assert status.stderr_truncated is False


@pytest.mark.asyncio
async def test_async_get_background_job_forwards_timeout_to_status_probe():
    client = AsyncSandboxClient(api_key="test-key")
    client_any = cast(Any, client)

    seen_timeouts: List[Optional[int]] = []

    client_any.execute_command = _async_probe("", seen_timeouts)

    job = _make_job()
    status = await client.get_background_job("sbx-123", job, timeout=60)

    assert not status.completed
    assert seen_timeouts == [60]


@pytest.mark.asyncio
async def test_async_get_background_job_defaults_timeout_to_none():
    client = AsyncSandboxClient(api_key="test-key")
    client_any = cast(Any, client)

    seen_timeouts: List[Optional[int]] = []

    client_any.execute_command = _async_probe("", seen_timeouts)

    job = _make_job()
    await client.get_background_job("sbx-123", job)

    assert seen_timeouts == [None]


def test_sync_get_background_job_raises_when_job_died_without_exit_code():
    client = SandboxClient(APIClient(api_key="test-key"))
    cast(Any, client).execute_command = _probe("lost\n")

    with pytest.raises(APIError, match="exited without recording an exit code"):
        client.get_background_job("sbx-123", _make_job())


@pytest.mark.asyncio
async def test_async_get_background_job_raises_when_job_died_without_exit_code():
    client = AsyncSandboxClient(api_key="test-key")
    cast(Any, client).execute_command = _async_probe("lost\n")

    with pytest.raises(APIError, match="exited without recording an exit code"):
        await client.get_background_job("sbx-123", _make_job())
