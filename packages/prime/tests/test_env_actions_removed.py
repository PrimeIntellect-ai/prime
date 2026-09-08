"""Environment Actions were removed from the Hub (platform #4872, ENG-5766).

The CLI keeps the old entry points so scripts do not break with a usage error,
but they explain the removal instead of calling endpoints that now 404.
"""

from typing import Any, Dict, Optional

import pytest
from prime_cli.api.client import APIClient
from prime_cli.main import app
from prime_cli.utils import strip_ansi
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture
def no_api_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_API_KEY", "test-key")

    def _refuse(self: Any, endpoint: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise AssertionError(f"unexpected API call: {endpoint}")

    monkeypatch.setattr(APIClient, "get", _refuse)
    monkeypatch.setattr(APIClient, "post", _refuse)


@pytest.mark.parametrize(
    "args",
    [
        ["list", "owner/env"],
        ["list", "owner/env", "--num", "5", "--output", "json"],
        ["logs", "owner/env", "job-1", "--tail", "50", "--follow"],
        ["retry", "owner/env"],
        ["retry", "owner/env", "job-1", "--output", "json"],
    ],
)
def test_env_action_commands_explain_the_removal(no_api_calls: None, args: list[str]) -> None:
    result = runner.invoke(app, ["env", "action", *args])

    assert result.exit_code == 1
    assert "Environment Actions were removed" in strip_ansi(result.output)


def test_env_action_group_is_hidden_from_help() -> None:
    result = runner.invoke(app, ["env", "--help"])

    assert result.exit_code == 0
    assert "Environment Actions" not in strip_ansi(result.output)


def test_env_list_ignores_the_removed_action_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_API_KEY", "test-key")
    captured: Dict[str, Any] = {}

    def _get(self: Any, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        captured["endpoint"] = endpoint
        captured["params"] = dict(params or {})
        return {"data": [], "total_count": 0, "offset": 0, "limit": 20}

    monkeypatch.setattr(APIClient, "get", _get)

    result = runner.invoke(app, ["env", "list", "--show-actions", "--action-status", "FAILED"])

    assert result.exit_code == 0, result.output
    assert captured["endpoint"].startswith("/environmentshub")
    assert "ci_status" not in captured["params"]
    assert "include_ci_status" not in captured["params"]
    assert "Environment Actions were removed" in strip_ansi(result.output)


def test_env_list_json_output_stays_machine_readable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_API_KEY", "test-key")

    def _get(self: Any, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {"data": [], "total_count": 0, "offset": 0, "limit": 20}

    monkeypatch.setattr(APIClient, "get", _get)

    result = runner.invoke(app, ["env", "list", "--show-actions", "--output", "json"])

    assert result.exit_code == 0, result.output
    assert "Environment Actions were removed" not in result.output


def test_env_status_help_no_longer_describes_action_status() -> None:
    result = runner.invoke(app, ["env", "status", "--help"])

    assert result.exit_code == 0
    assert "action status" not in strip_ansi(result.output).lower()


def test_train_run_help_hides_the_action_preflight_flag() -> None:
    result = runner.invoke(app, ["train", "run", "--help"])

    assert result.exit_code == 0
    assert "--skip-action-check" not in strip_ansi(result.output)
