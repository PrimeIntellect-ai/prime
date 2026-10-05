"""Tests for the whoami command's output rendering."""

from pathlib import Path
from typing import Any

import pytest
from prime_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    for variable in ("PRIME_CONTEXT", "PRIME_TEAM_ID", "PRIME_USER_ID"):
        monkeypatch.delenv(variable, raising=False)


def _mock_whoami(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_get(self: Any, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        assert endpoint == "/user/whoami"
        return {
            "data": {
                "id": "user-1",
                "email": "user@example.com",
                "name": "User One",
                "slug": None,  # missing username renders "Not set"
                "scope": {},
            }
        }

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)


@pytest.mark.parametrize("plain", [False, True])
def test_missing_username_renders_without_markup_leak(monkeypatch, plain):
    _mock_whoami(monkeypatch)

    args = ["whoami", "--plain"] if plain else ["whoami"]
    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    assert "Not set" in result.output
    assert "[dim]" not in result.output


def _mock_whoami_with_limits(
    monkeypatch: pytest.MonkeyPatch, account_limits: Any, calls: list[Any]
) -> None:
    def mock_get(self: Any, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs.get("params"))
        return {
            "data": {
                "id": "user-1",
                "email": "user@example.com",
                "name": "User One",
                "slug": "user-one",
                "scope": {},
                "key_limits": {
                    "max_concurrent_sandboxes": 10,
                    "max_sandbox_creations_per_hour": None,
                    "max_sandbox_cpu_cores": 1_000_000,
                    "max_sandbox_gpu_count": None,
                    "max_concurrent_tunnels": 2,
                    "max_tunnel_creations_per_hour": None,
                    "max_tunnel_ttl_hours": None,
                },
                "account_limits": account_limits,
            }
        }

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)


ACCOUNT_LIMITS = {
    "sandbox_total_cpu_limit": 4096,
    "vm_sandbox_limit": 1024,
    "vm_sandbox_gpu_limit": 0,
    "tunnel_limit": 32,
    "tunnel_creations_per_hour_limit": 256,
    "tunnel_ttl_hours": 168,
}


def _limit_row(output: str, label: str) -> list[str]:
    line = next(line for line in output.splitlines() if label in line)
    return [cell.strip() for cell in line.split("│")[2:5]]


def test_limits_table_shows_account_key_and_effective(monkeypatch):
    calls: list[Any] = []
    _mock_whoami_with_limits(monkeypatch, ACCOUNT_LIMITS, calls)

    result = runner.invoke(app, ["whoami"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    assert calls == [None]
    # A key limit above the account's never raises the effective limit.
    assert _limit_row(result.output, "Sandbox CPU cores") == ["4096", "1000000", "4096"]
    assert _limit_row(result.output, "Concurrent tunnels") == ["32", "2", "2"]
    assert _limit_row(result.output, "Concurrent sandboxes") == ["-", "10", "10"]
    assert _limit_row(result.output, "Tunnel TTL (hours)") == ["168", "-", "168"]
    assert _limit_row(result.output, "Sandbox creations / hour") == ["-", "-", "-"]


def test_limits_table_requests_the_active_teams_limits(monkeypatch):
    calls: list[Any] = []
    _mock_whoami_with_limits(monkeypatch, None, calls)
    monkeypatch.setenv("PRIME_TEAM_ID", "team-1")

    result = runner.invoke(app, ["whoami"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    assert calls == [{"teamId": "team-1"}]
    assert "Account limits are unavailable for this account." in result.output
    assert _limit_row(result.output, "Concurrent tunnels") == ["-", "2", "2"]
