"""Tests for RL logs wait/poll helpers."""

from unittest.mock import MagicMock

import pytest
from prime_cli.commands.rl import _fetch_rl_logs_until_content, _rl_logs_have_displayable_content


def test_rl_logs_have_displayable_content_raw() -> None:
    assert _rl_logs_have_displayable_content("", True) is False
    assert _rl_logs_have_displayable_content("  \n\t", True) is False
    assert _rl_logs_have_displayable_content("hello", True) is True


def test_rl_logs_have_displayable_content_formatted_skips_progress_only() -> None:
    progress_only = (
        '{"timestamp":"2025-01-01T00:00:00Z","level":"INFO",'
        '"type":"progress","message":"x"}'
    )
    assert _rl_logs_have_displayable_content(progress_only, False) is False
    real_line = (
        '{"timestamp":"2025-01-01T00:00:00Z","level":"INFO","message":"started"}'
    )
    assert _rl_logs_have_displayable_content(real_line, False) is True


def test_fetch_rl_logs_until_content_returns_as_soon_as_logs_exist() -> None:
    mock_client = MagicMock()
    mock_client.get_logs.side_effect = ["", "", "ok\n"]
    result = _fetch_rl_logs_until_content(mock_client, "run-1", 1000, raw=True)
    assert result == "ok\n"
    assert mock_client.get_logs.call_count == 3


def test_fetch_rl_logs_until_content_stops_after_wait_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_client = MagicMock()
    mock_client.get_logs.return_value = ""
    monkeypatch.setattr(
        "prime_cli.commands.rl.RL_LOGS_WAIT_FOR_CONTENT_SECONDS",
        0.02,
    )
    monkeypatch.setattr(
        "prime_cli.commands.rl.RL_LOGS_WAIT_POLL_INTERVAL_SECONDS",
        0.001,
    )
    result = _fetch_rl_logs_until_content(mock_client, "run-1", 1000, raw=True)
    assert result == ""
    assert mock_client.get_logs.call_count >= 2
