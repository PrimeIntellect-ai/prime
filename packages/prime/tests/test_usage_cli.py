"""Tests for `prime train usage` (per-run token + price view)."""

import io
import json
from typing import Any, Dict, List, Optional

import pytest
from prime_cli.api.billing import RunUsage
from prime_cli.commands.usage import _build_run_usage_table
from prime_cli.main import app
from prime_cli.utils.formatters import strip_ansi
from rich.console import Console
from typer.testing import CliRunner


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


def test_train_usage_style_snapshot_preserves_column_and_total_styles() -> None:
    stream = io.StringIO()
    console = Console(
        file=stream,
        record=True,
        force_terminal=True,
        color_system="standard",
        width=100,
    )

    console.print(_build_run_usage_table(RunUsage.model_validate(_run_usage_payload())))
    snapshot = console.export_text(styles=True)

    assert "\x1b[36mTraining          \x1b[0m" in snapshot
    assert "\x1b[32m$12.50\x1b[0m" in snapshot
    assert "\x1b[1;36mTotal             \x1b[0m" in snapshot
    assert "\x1b[1;32m$17.50\x1b[0m" in snapshot
    assert "\x1b[2m                   model: qwen2.5-7b                   \x1b[0m" in snapshot


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


@pytest.mark.parametrize("plain", [False, True])
def test_train_usage_escapes_rich_markup_in_run_name_and_status(
    monkeypatch: pytest.MonkeyPatch,
    plain: bool,
) -> None:
    """API-supplied run_name/status must not be parsed as Rich markup."""
    payload = _run_usage_payload()
    payload["run_name"] = "evil[bold red]name[/bold red] [/INFO]"
    payload["status"] = "RUN[blink] [/INFO]"
    payload["run_id"] = "run[/INFO]"
    payload["base_model"] = "model[/INFO]"

    monkeypatch.setattr(
        "prime_cli.core.APIClient.get",
        _make_get_mock({"/billing/runs/rft_abc/usage": payload}, []),
    )

    result = CliRunner().invoke(
        app,
        [*(["--plain"] if plain else []), "train", "usage", "rft_abc"],
        env={"COLUMNS": "200"},
    )

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    # Bracketed text is rendered literally, not consumed as markup.
    assert "evil[bold red]name[/bold red]" in plain
    assert "[/INFO]" in plain
    assert "RUN[blink] [/INFO]" in plain
    assert "run[/INFO]" in plain
    assert "model[/INFO]" in plain


def test_train_usage_wraps_long_metadata_without_corrupting_table(monkeypatch) -> None:
    payload = _run_usage_payload()
    payload["run_name"] = "[bold]" + ("🚀" * 120) + "[/bold]"
    payload["base_model"] = "model[red" + ("x" * 280)
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get",
        _make_get_mock({"/billing/runs/rft_abc/usage": payload}, []),
    )

    result = CliRunner().invoke(
        app,
        ["train", "usage", "rft_abc"],
        env={"COLUMNS": "60"},
    )

    assert result.exit_code == 0, result.output
    output = strip_ansi(result.output)
    assert "[bold]" in output
    assert "model[red" in output
    assert max(map(len, output.splitlines())) <= 60


@pytest.mark.parametrize("plain", [False, True])
@pytest.mark.parametrize("output", ["table", "json"])
@pytest.mark.parametrize("watch", [False, True])
def test_train_usage_backend_errors_are_literal(monkeypatch, plain, output, watch):
    from prime_cli.core import APIError

    def fail(*args, **kwargs):
        raise APIError("billing failed [/INFO] [bold]data[/bold]")

    monkeypatch.setattr("prime_cli.commands.usage.BillingClient.get_run_usage", fail)
    args = ["--plain"] if plain else []
    args += ["train", "usage", "rft_abc", "--output", output]
    if watch:
        args.append("--watch")
    result = CliRunner().invoke(app, args)

    assert result.exit_code == 1
    diagnostic = result.stderr if output == "json" else result.stdout
    assert "billing failed [/INFO] [bold]data[/bold]" in strip_ansi(diagnostic)
    if output == "json":
        assert result.stdout == ""


def test_train_usage_json_preserves_markup_values(monkeypatch):
    payload = _run_usage_payload()
    payload.update(run_id="run[/INFO]", run_name="[bold]name[/bold]", status="[/INFO]")
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get",
        _make_get_mock({"/billing/runs/rft_abc/usage": payload}, []),
    )

    result = CliRunner().invoke(app, ["--plain", "train", "usage", "rft_abc", "-o", "json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == payload


def test_train_usage_propagates_typed_api_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Subclassed APIErrors must propagate intact, not be re-wrapped."""
    from prime_cli.core import UnauthorizedError

    def raise_unauthorized(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise UnauthorizedError("token expired [/INFO]")

    monkeypatch.setattr("prime_cli.core.APIClient.get", raise_unauthorized)

    result = CliRunner().invoke(app, ["train", "usage", "rft_abc"])

    assert result.exit_code == 1
    # The original message should appear, not "Failed to get run usage: …".
    assert "token expired [/INFO]" in result.output
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
