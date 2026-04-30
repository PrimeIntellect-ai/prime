"""Tests for `prime usage` — RFT run usage and aggregated billing summary."""

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


def _summary_payload() -> Dict[str, Any]:
    return {
        "period": "this_month",
        "start_date": "2026-04-01",
        "end_date": "2026-04-30",
        "wallet_id": "w-1",
        "team_id": None,
        "total_cost_usd": 612.5,
        "areas": [
            {
                "area": "training",
                "total_cost_usd": 600.0,
                "training_tokens": 20_000_000,
                "inference_tokens": 4_000_000,
                "inference_requests": 0,
            },
            {
                "area": "inference",
                "total_cost_usd": 8.0,
                "training_tokens": 0,
                "inference_tokens": 0,
                "inference_requests": 20,
            },
            {
                "area": "compute",
                "total_cost_usd": 2.5,
                "training_tokens": 0,
                "inference_tokens": 0,
                "inference_requests": 0,
            },
            {
                "area": "disks",
                "total_cost_usd": 1.5,
                "training_tokens": 0,
                "inference_tokens": 0,
                "inference_requests": 0,
            },
            {
                "area": "images",
                "total_cost_usd": 0.5,
                "training_tokens": 0,
                "inference_tokens": 0,
                "inference_requests": 0,
            },
        ],
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


def test_run_usage_table_renders_tokens_and_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get",
        _make_get_mock({"/billing/runs/rft_abc/usage": _run_usage_payload()}, calls),
    )

    result = CliRunner().invoke(app, ["usage", "run", "rft_abc"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    # Table header
    assert "Run Usage" in plain
    assert "my-run" in plain
    assert "RUNNING" in plain
    # Token formatting (5M, 1M, 3M)
    assert "5.00M" in plain
    assert "1.00M" in plain
    assert "3.00M" in plain
    # Costs
    assert "$12.50" in plain
    assert "$7.00" in plain
    assert "$19.50" in plain
    # Pricing per Mtok
    assert "$2.5" in plain or "$2.50" in plain
    # Single GET to the right endpoint
    assert calls == [{"endpoint": "/billing/runs/rft_abc/usage", "params": None}]


def test_run_usage_json_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get",
        _make_get_mock({"/billing/runs/rft_abc/usage": _run_usage_payload()}, []),
    )

    result = CliRunner().invoke(app, ["usage", "run", "rft_abc", "--output", "json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["run_id"] == "rft_abc"
    assert data["total_tokens"] == 9_000_000
    assert data["total_cost_usd"] == 19.5
    assert data["pricing"]["training_per_mtok"] == 2.5


def test_summary_table_renders_all_areas_excludes_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get",
        _make_get_mock({"/billing/usage": _summary_payload()}, calls),
    )

    result = CliRunner().invoke(app, ["usage", "summary"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    assert "Training" in plain
    assert "Inference" in plain
    assert "Compute" in plain
    assert "Disks" in plain
    assert "Images" in plain
    assert "Sandbox" not in plain
    assert "$600.00" in plain
    assert "$612.50" in plain
    assert "20 req" in plain
    # Default period sent as query param
    assert calls == [{"endpoint": "/billing/usage", "params": {"period": "this_month"}}]


def test_summary_passes_team_and_period_query_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get",
        _make_get_mock({"/billing/usage": _summary_payload()}, calls),
    )

    result = CliRunner().invoke(
        app,
        [
            "usage",
            "summary",
            "--period",
            "7_days",
            "--team",
            "team-1",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["period"] == "this_month"  # echoed from payload
    assert calls == [
        {
            "endpoint": "/billing/usage",
            "params": {"period": "7_days", "teamId": "team-1"},
        }
    ]


def test_summary_rejects_invalid_period(monkeypatch: pytest.MonkeyPatch) -> None:
    # No HTTP call expected — should fail before that.
    def fail_get(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise AssertionError("Should not call API on invalid period")

    monkeypatch.setattr("prime_cli.core.APIClient.get", fail_get)

    result = CliRunner().invoke(app, ["usage", "summary", "--period", "forever"])

    assert result.exit_code == 1, result.output
    assert "invalid --period" in result.output


def test_usage_help_lists_subcommands(monkeypatch: pytest.MonkeyPatch) -> None:
    result = CliRunner().invoke(app, ["usage", "--help"])

    assert result.exit_code == 0, result.output
    assert "run" in result.output
    assert "summary" in result.output
    assert "View token usage and price" in result.output
