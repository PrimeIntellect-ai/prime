"""Adaptive pacing tests for background-job completion polling."""

import pytest

from prime_sandboxes.sandbox import _background_job_poll_delay


def test_background_job_polling_backs_off_from_default_interval() -> None:
    delays = [_background_job_poll_delay(3, poll_index) for poll_index in range(8)]

    assert delays == pytest.approx([3, 4.5, 6.75, 10.125, 15.1875, 20, 20, 20])


def test_background_job_poll_interval_override_sets_initial_delay_and_still_backs_off() -> None:
    delays = [_background_job_poll_delay(5, poll_index) for poll_index in range(6)]

    assert delays == pytest.approx([5, 7.5, 11.25, 16.875, 20, 20])
