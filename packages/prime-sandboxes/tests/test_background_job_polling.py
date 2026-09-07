"""Adaptive pacing tests for background-job completion polling."""

import pytest

from prime_sandboxes.sandbox import (
    BACKGROUND_JOB_POLL_MAX_DELAY,
    _next_background_job_poll_delay,
)


def _poll_delays(initial_interval: float, count: int) -> list[float]:
    delay = min(initial_interval, BACKGROUND_JOB_POLL_MAX_DELAY)
    delays = []
    for _ in range(count):
        delays.append(delay)
        delay = _next_background_job_poll_delay(delay)
    return delays


def test_background_job_polling_backs_off_from_default_interval() -> None:
    delays = _poll_delays(3, 8)

    assert delays == pytest.approx([3, 4.5, 6.75, 10.125, 15.1875, 20, 20, 20])


def test_background_job_poll_interval_override_sets_initial_delay_and_still_backs_off() -> None:
    delays = _poll_delays(5, 6)

    assert delays == pytest.approx([5, 7.5, 11.25, 16.875, 20, 20])


def test_background_job_polling_stays_capped_for_very_old_jobs() -> None:
    delays = _poll_delays(3, 1_752)

    assert delays[-2:] == [20, 20]
