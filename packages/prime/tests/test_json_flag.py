"""`--json` replaces `--output json`; the old option stays as a hidden alias."""

import json
from typing import Any, Dict, Optional

import pytest
import typer
from prime_cli.main import app
from prime_cli.utils import get_console, resolve_output_format
from prime_cli.utils.formatters import strip_ansi
from typer.testing import CliRunner

WALLET = {
    "wallet_id": "wal_abc",
    "team_id": None,
    "balance_usd": 1.5,
    "currency": "USD",
    "total_billings": 0,
    "recent_billings": [],
}


@pytest.fixture(autouse=True)
def _wallet_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_API_KEY", "dummy")
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)
    monkeypatch.setattr("prime_cli.core.Config.team_id", None)

    def _get(self: Any, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return WALLET

    monkeypatch.setattr("prime_cli.core.APIClient.get", _get)


@pytest.mark.parametrize("flag", [["--json"], ["--output", "json"], ["-o", "json"]])
def test_json_flag_and_legacy_alias_print_the_same_json(flag: list[str]) -> None:
    result = CliRunner().invoke(app, ["wallet", *flag])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == WALLET


def test_legacy_output_table_still_prints_the_table() -> None:
    result = CliRunner().invoke(app, ["wallet", "--output", "table"])

    assert result.exit_code == 0, result.output
    assert "$1.50" in strip_ansi(result.output)


def test_json_flag_wins_over_legacy_output() -> None:
    result = CliRunner().invoke(app, ["wallet", "--json", "--output", "table"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == WALLET


def test_legacy_output_rejects_unknown_format() -> None:
    result = CliRunner().invoke(app, ["wallet", "--output", "yaml"])

    assert result.exit_code == 1
    assert "Invalid output format 'yaml'" in strip_ansi(result.output)


def test_help_lists_json_and_hides_legacy_output() -> None:
    result = CliRunner().invoke(app, ["wallet", "--help"])

    assert result.exit_code == 0
    output = strip_ansi(result.output)
    assert "--json" in output
    assert "--output" not in output


def test_resolve_output_format_uses_the_command_default() -> None:
    console = get_console()

    assert resolve_output_format(False, None, console) == "table"
    assert resolve_output_format(False, None, console, default="text") == "text"
    assert resolve_output_format(False, "text", console, default="text") == "text"
    assert resolve_output_format(True, None, console, default="text") == "json"
    with pytest.raises(typer.Exit):
        resolve_output_format(False, "table", console, default="text")


class _Evals:
    def __init__(self, _api_client: Any) -> None:
        pass

    def get_evaluation(self, eval_id: str) -> Dict[str, Any]:
        return {"evaluation_id": eval_id, "status": "COMPLETED"}


@pytest.mark.parametrize("flag", [[], ["--json"], ["--output", "json"], ["-o", "pretty"]])
def test_eval_get_keeps_output_and_accepts_json(
    monkeypatch: pytest.MonkeyPatch, flag: list[str]
) -> None:
    monkeypatch.setattr("prime_cli.commands.evals.EvalsClient", _Evals)

    result = CliRunner().invoke(app, ["eval", "get", "eval-1", *flag])

    assert result.exit_code == 0, result.output
    assert json.loads(strip_ansi(result.output)) == {
        "evaluation_id": "eval-1",
        "status": "COMPLETED",
    }
