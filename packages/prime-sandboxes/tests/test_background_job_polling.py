"""Adaptive pacing tests for background-job completion polling."""

from typing import Any, cast

import pytest

from prime_sandboxes.core.client import APIClient
from prime_sandboxes.models import BackgroundJob, BackgroundJobStatus
from prime_sandboxes.sandbox import (
    BACKGROUND_JOB_POLL_MAX_DELAY,
    AsyncSandboxClient,
    SandboxClient,
    _next_background_job_poll_delay,
)


def _poll_delays(initial_interval: float, count: int) -> list[float]:
    delay = min(initial_interval, BACKGROUND_JOB_POLL_MAX_DELAY)
    delays = []
    for _ in range(count):
        delays.append(delay)
        delay = _next_background_job_poll_delay(delay)
    return delays


def _job() -> BackgroundJob:
    return BackgroundJob(
        job_id="job",
        sandbox_id="sandbox",
        stdout_log_file="/tmp/job.stdout",
        stderr_log_file="/tmp/job.stderr",
        exit_file="/tmp/job.exit",
    )


def test_background_job_polling_backs_off_from_default_interval() -> None:
    delays = _poll_delays(3, 8)

    assert delays == pytest.approx([3, 4.5, 6.75, 10.125, 15.1875, 20, 20, 20])


def test_background_job_poll_interval_override_sets_initial_delay_and_still_backs_off() -> None:
    delays = _poll_delays(5, 6)

    assert delays == pytest.approx([5, 7.5, 11.25, 16.875, 20, 20])


def test_background_job_polling_stays_capped_for_very_old_jobs() -> None:
    delays = _poll_delays(3, 1_752)

    assert delays[-2:] == [20, 20]


def test_sync_background_job_polling_checks_status_at_timeout(monkeypatch) -> None:
    client = SandboxClient(APIClient(api_key="test-key"))
    clock = 0.0
    sleeps = []
    snapshots = iter(
        [
            BackgroundJobStatus(job_id="job", completed=False),
            BackgroundJobStatus(job_id="job", completed=False),
            BackgroundJobStatus(job_id="job", completed=False),
            BackgroundJobStatus(job_id="job", completed=True, exit_code=0),
        ]
    )
    result = BackgroundJobStatus(job_id="job", completed=True, exit_code=0, stdout="done")

    def sleep(delay: float) -> None:
        nonlocal clock
        sleeps.append(delay)
        clock += delay

    monkeypatch.setattr("prime_sandboxes.sandbox.time.monotonic", lambda: clock)
    monkeypatch.setattr("prime_sandboxes.sandbox.time.sleep", sleep)
    monkeypatch.setattr(client, "start_background_job", lambda *_args, **_kwargs: _job())
    monkeypatch.setattr(cast(Any, client)._auth_cache, "is_vm", lambda _sandbox_id: False)
    monkeypatch.setattr(
        client,
        "get_background_job_status",
        lambda *_args, **_kwargs: next(snapshots),
    )
    monkeypatch.setattr(
        cast(Any, client)._background_job_output_coordinator,
        "get",
        lambda *_args, **_kwargs: result,
    )

    assert client.run_background_job("sandbox", "command", timeout=10) is result
    assert sleeps == pytest.approx([3, 4.5, 2.5])


@pytest.mark.asyncio
async def test_async_background_job_polling_checks_status_at_timeout(monkeypatch) -> None:
    client = AsyncSandboxClient(api_key="test-key")
    await client.client.aclose()
    clock = 0.0
    sleeps = []
    snapshots = iter(
        [
            BackgroundJobStatus(job_id="job", completed=False),
            BackgroundJobStatus(job_id="job", completed=False),
            BackgroundJobStatus(job_id="job", completed=False),
            BackgroundJobStatus(job_id="job", completed=True, exit_code=0),
        ]
    )
    result = BackgroundJobStatus(job_id="job", completed=True, exit_code=0, stdout="done")

    async def sleep(delay: float) -> None:
        nonlocal clock
        sleeps.append(delay)
        clock += delay

    async def start_background_job(*_args, **_kwargs) -> BackgroundJob:
        return _job()

    async def is_vm(_sandbox_id: str) -> bool:
        return False

    async def get_status(*_args, **_kwargs) -> BackgroundJobStatus:
        return next(snapshots)

    async def get_output(*_args, **_kwargs) -> BackgroundJobStatus:
        return result

    monkeypatch.setattr("prime_sandboxes.sandbox.time.monotonic", lambda: clock)
    monkeypatch.setattr("prime_sandboxes.sandbox.asyncio.sleep", sleep)
    monkeypatch.setattr(client, "start_background_job", start_background_job)
    monkeypatch.setattr(cast(Any, client)._auth_cache, "is_vm", is_vm)
    monkeypatch.setattr(client, "get_background_job_status", get_status)
    monkeypatch.setattr(cast(Any, client)._background_job_output_coordinator, "get", get_output)

    assert await client.run_background_job("sandbox", "command", timeout=10) is result
    assert sleeps == pytest.approx([3, 4.5, 2.5])
