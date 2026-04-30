from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from prime_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


def test_tunnel_list_passes_label_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    captured: dict[str, Any] = {}

    class FakeTunnelClient:
        async def list_tunnels(self, **kwargs: Any) -> list[Any]:
            captured.update(kwargs)
            return [
                SimpleNamespace(
                    tunnel_id="t-test123",
                    name="api",
                    url="https://t-test123.example.com",
                    hostname="t-test123.example.com",
                    status="CONNECTED",
                    labels=["dev"],
                    local_port=8765,
                    user_id="user-1",
                    team_id=None,
                    created_at=datetime.now(timezone.utc),
                    expires_at=datetime.now(timezone.utc),
                )
            ]

        async def close(self) -> None:
            return None

    monkeypatch.setattr("prime_cli.commands.tunnel.TunnelClient", FakeTunnelClient)

    result = runner.invoke(
        app,
        ["tunnel", "list", "--label", "dev", "--sort-by", "name", "--output", "json"],
    )

    assert result.exit_code == 0, result.output
    assert captured["labels"] == ["dev"]
    assert captured["sort_by"] == "name"


def test_tunnel_stop_by_label_uses_bulk_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    captured: dict[str, Any] = {}

    class FakeTunnelClient:
        async def bulk_delete_tunnels(self, **kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return {"succeeded": ["t-test123"], "failed": [], "message": "ok"}

        async def close(self) -> None:
            return None

    monkeypatch.setattr("prime_cli.commands.tunnel.TunnelClient", FakeTunnelClient)

    result = runner.invoke(app, ["tunnel", "stop", "--label", "dev", "--yes"])

    assert result.exit_code == 0, result.output
    assert captured["labels"] == ["dev"]
    assert captured["all_users"] is False


def test_tunnel_stop_all_uses_scoped_bulk_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    captured: dict[str, Any] = {}

    class FakeTunnelClient:
        async def list_tunnels(self, **kwargs: Any) -> list[Any]:
            raise AssertionError("stop --all should not list tunnels before deletion")

        async def bulk_delete_tunnels(self, **kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return {"succeeded": ["t-test123"], "failed": [], "message": "ok"}

        async def close(self) -> None:
            return None

    monkeypatch.setattr("prime_cli.commands.tunnel.TunnelClient", FakeTunnelClient)

    result = runner.invoke(app, ["tunnel", "stop", "--all", "--yes"])

    assert result.exit_code == 0, result.output
    assert captured["team_id"] is None
    assert captured["all_users"] is False


def test_tunnel_stop_all_users_requires_team_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")

    result = runner.invoke(app, ["tunnel", "stop", "--all", "--all-users", "--yes"])

    assert result.exit_code == 1
    assert "--all-users requires --team-id" in result.output
