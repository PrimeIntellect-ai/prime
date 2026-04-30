"""Tests for `prime wallet` — balance + recent billings snapshot."""

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
    # Default to no team so tests don't depend on the developer's local
    # `~/.prime/config.json`. Tests that exercise the team path override
    # `Config.team_id` explicitly.
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)
    monkeypatch.setattr("prime_cli.core.Config.team_id", None)


def _wallet_payload() -> Dict[str, Any]:
    return {
        "wallet_id": "wal_abc",
        "team_id": None,
        "balance_usd": 123.45,
        "currency": "USD",
        "total_billings": 3,
        "recent_billings": [
            {
                "id": "bill_1",
                "created_at": "2026-04-30T10:00:00Z",
                "updated_at": "2026-04-30T10:00:00Z",
                "last_billed_at": "2026-04-30T10:00:00Z",
                "amount_usd": 12.5,
                "currency": "USD",
                "resource_type": "training",
                "resource_id": "rft_xyz",
            },
            {
                "id": "bill_2",
                "created_at": "2026-04-30T09:00:00Z",
                "updated_at": "2026-04-30T09:30:00Z",
                "last_billed_at": "2026-04-30T09:30:00Z",
                "amount_usd": 1.25,
                "currency": "USD",
                "resource_type": "compute",
                "resource_id": "pod_def",
            },
            {
                "id": "bill_3",
                "created_at": "2026-04-30T08:00:00Z",
                "updated_at": "2026-04-30T08:00:00Z",
                "last_billed_at": None,
                "amount_usd": 0.05,
                "currency": "USD",
                "resource_type": "disks",
                "resource_id": "disk_ghi",
            },
        ],
    }


def _empty_wallet_payload() -> Dict[str, Any]:
    return {
        "wallet_id": "wal_abc",
        "team_id": None,
        "balance_usd": 0.0,
        "currency": "USD",
        "total_billings": 0,
        "recent_billings": [],
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


def test_wallet_renders_balance_and_billings_table(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get",
        _make_get_mock({"/billing/wallet": _wallet_payload()}, calls),
    )

    result = CliRunner().invoke(app, ["wallet"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    # Balance
    assert "$123.45" in plain
    # Total count
    assert "3" in plain
    # Each billing row's resource type + id
    assert "training" in plain
    assert "rft_xyz" in plain
    assert "compute" in plain
    assert "pod_def" in plain
    assert "disks" in plain
    # Each billing row's amount
    assert "$12.50" in plain
    assert "$1.25" in plain
    assert "$0.05" in plain or "$0.0500" in plain
    # Default query params
    assert calls == [{"endpoint": "/billing/wallet", "params": {"limit": 20, "offset": 0}}]


def test_wallet_json_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get",
        _make_get_mock({"/billing/wallet": _wallet_payload()}, []),
    )

    result = CliRunner().invoke(app, ["wallet", "--output", "json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["wallet_id"] == "wal_abc"
    assert data["balance_usd"] == 123.45
    assert data["total_billings"] == 3
    assert len(data["recent_billings"]) == 3
    assert data["recent_billings"][0]["resource_type"] == "training"


def test_wallet_passes_limit_and_uses_configured_team(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """team_id comes from Config (set by `prime switch`), never a CLI flag."""
    calls: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get",
        _make_get_mock({"/billing/wallet": _wallet_payload()}, calls),
    )
    monkeypatch.setattr("prime_cli.core.Config.team_id", "team-cfg")

    result = CliRunner().invoke(app, ["wallet", "--limit", "5", "--output", "json"])

    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "endpoint": "/billing/wallet",
            "params": {"limit": 5, "offset": 0, "teamId": "team-cfg"},
        }
    ]


def test_wallet_does_not_accept_team_flag() -> None:
    """`--team` is intentionally not supported — must follow the configured team."""
    result = CliRunner().invoke(app, ["wallet", "--team", "team-1"])

    assert result.exit_code != 0
    assert "No such option: --team" in result.output or "no such option" in result.output.lower()


def test_wallet_handles_empty_billings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get",
        _make_get_mock({"/billing/wallet": _empty_wallet_payload()}, []),
    )

    result = CliRunner().invoke(app, ["wallet"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    assert "$0.00" in plain
    assert "No billing rows yet" in plain


def test_wallet_appears_in_root_help() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0, result.output
    assert "wallet" in result.output


def test_wallet_propagates_typed_api_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Subclassed APIErrors must propagate intact (e.g. 401 → UnauthorizedError)."""
    from prime_cli.core import UnauthorizedError

    def raise_unauthorized(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise UnauthorizedError("token expired")

    monkeypatch.setattr("prime_cli.core.APIClient.get", raise_unauthorized)

    result = CliRunner().invoke(app, ["wallet"])

    assert result.exit_code == 1
    assert "token expired" in result.output
    assert "Failed to get wallet" not in result.output
