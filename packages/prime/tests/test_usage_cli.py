"""Tests for `prime train usage` (per-run token + price view)."""

import json
from typing import Any, Dict, List, Optional

import pytest
from prime_cli.main import app
from prime_cli.utils.formatters import strip_ansi
from typer.testing import CliRunner


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_API_KEY", "dummy")
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")


def _run_usage_payload() -> Dict[str, Any]:
    return {
        "run_id": "rft_abc",
        "run_name": "my-run",
        "base_model": "qwen2.5-7b",
        "status": "RUNNING",
        "training": {
            "tokens": 5_000_000,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 12.5,
        },
        "inference": {
            "tokens": 4_000_000,
            "input_tokens": 1_000_000,
            "output_tokens": 3_000_000,
            "cost_usd": 7.0,
        },
        "total_tokens": 9_000_000,
        "total_cost_usd": 19.5,
        "pricing": {
            "training_per_mtok": 2.5,
            "inference_input_per_mtok": 0.5,
            "inference_output_per_mtok": 1.5,
        },
        "record_count": 12,
    }


def _make_get_mock(routes: Dict[str, Dict[str, Any]], calls: List[Dict[str, Any]]):
    def mock_get(
        self: Any, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        calls.append({"endpoint": endpoint, "params": params})
        if endpoint in routes:
            return routes[endpoint]
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    return mock_get


def test_train_usage_table_renders_tokens_and_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get",
        _make_get_mock({"/billing/runs/rft_abc/usage": _run_usage_payload()}, calls),
    )

    result = CliRunner().invoke(app, ["train", "usage", "rft_abc"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    assert "Run Usage" in plain
    assert "my-run" in plain
    assert "RUNNING" in plain
    assert "5.00M" in plain
    assert "1.00M" in plain
    assert "3.00M" in plain
    assert "$12.50" in plain
    assert "$7.00" in plain
    assert "$19.50" in plain
    assert "$2.5" in plain or "$2.50" in plain
    assert calls == [{"endpoint": "/billing/runs/rft_abc/usage", "params": None}]


def test_train_usage_json_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get",
        _make_get_mock({"/billing/runs/rft_abc/usage": _run_usage_payload()}, []),
    )

    result = CliRunner().invoke(app, ["train", "usage", "rft_abc", "--output", "json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["run_id"] == "rft_abc"
    assert data["total_tokens"] == 9_000_000
    assert data["total_cost_usd"] == 19.5
    assert data["pricing"]["training_per_mtok"] == 2.5


def test_train_usage_appears_in_train_help() -> None:
    result = CliRunner().invoke(app, ["train", "--help"])

    assert result.exit_code == 0, result.output
    assert "usage" in result.output


def test_train_usage_escapes_rich_markup_in_run_name_and_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """API-supplied run_name/status must not be parsed as Rich markup."""
    payload = _run_usage_payload()
    payload["run_name"] = "evil[bold red]name[/bold red]"
    payload["status"] = "RUN[blink]"

    monkeypatch.setattr(
        "prime_cli.core.APIClient.get",
        _make_get_mock({"/billing/runs/rft_abc/usage": payload}, []),
    )

    result = CliRunner().invoke(app, ["train", "usage", "rft_abc"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    # Bracketed text is rendered literally, not consumed as markup.
    assert "evil[bold red]name[/bold red]" in plain
    assert "RUN[blink]" in plain


def test_train_usage_propagates_typed_api_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Subclassed APIErrors must propagate intact, not be re-wrapped."""
    from prime_cli.core import UnauthorizedError

    def raise_unauthorized(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise UnauthorizedError("token expired")

    monkeypatch.setattr("prime_cli.core.APIClient.get", raise_unauthorized)

    result = CliRunner().invoke(app, ["train", "usage", "rft_abc"])

    assert result.exit_code == 1
    # The original message should appear, not "Failed to get run usage: …".
    assert "token expired" in result.output
    assert "Failed to get run usage" not in result.output
