"""Tests for `prime train usage` (per-run token + price view)."""

import json
from typing import Any, Dict, List, Optional

import pytest
from click.testing import CliRunner
from prime_cli.main import app
from prime_cli.utils.formatters import strip_ansi


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_API_KEY", "dummy")
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")


def _run_usage_payload() -> Dict[str, Any]:
    # Cost numbers are self-consistent with the rates × tokens math the CLI
    # does for the per-row inference breakdown:
    #   training:           5M × $2.5/Mtok = $12.50
    #   inference (input):  1M × $0.5/Mtok = $0.50
    #   inference (output): 3M × $1.5/Mtok = $4.50
    #   inference combined: $5.00; total: $17.50
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
            "cost_usd": 5.0,
        },
        "total_tokens": 9_000_000,
        "total_cost_usd": 17.5,
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
    # Per-row inference costs derived from rates × tokens:
    assert "$0.50" in plain  # 1M × $0.5
    assert "$4.50" in plain  # 3M × $1.5
    assert "$17.50" in plain  # total
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
    assert data["total_cost_usd"] == 17.5
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


def test_format_tokens_promotes_to_M_at_rounding_boundary() -> None:
    """A value that would round to '1000.00K' must render as '1.00M' instead."""
    from prime_cli.commands.usage import _format_tokens

    # 999_995 rounds to 1.0M at 2dp; without the fix it would print "1000.00K"
    assert _format_tokens(999_995) == "1.00M"
    # Comfortably within the K range
    assert _format_tokens(900_000) == "900.00K"
    assert _format_tokens(1234) == "1.23K"
    # Comfortably within the M range
    assert _format_tokens(1_000_000) == "1.00M"
    assert _format_tokens(1_500_000) == "1.50M"
    assert _format_tokens(0) == "0"


def test_train_usage_wraps_response_shape_drift_as_api_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the backend response is missing required fields, the CLI should
    surface a clean error rather than an unhandled Pydantic traceback.
    """
    bad_payload = {"run_id": "rft_abc"}  # missing training/inference/pricing
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get",
        _make_get_mock({"/billing/runs/rft_abc/usage": bad_payload}, []),
    )

    result = CliRunner().invoke(app, ["train", "usage", "rft_abc"])

    assert result.exit_code == 1
    assert "Unexpected" in result.output
