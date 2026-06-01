import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from prime_cli.commands.tunnel import _format_tunnel_for_output
from prime_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


def test_format_tunnel_does_not_derive_created_from_expiration() -> None:
    tunnel_data = _format_tunnel_for_output(
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
            created_at=None,
            expires_at=datetime.now(timezone.utc),
        )
    )

    assert tunnel_data["created_at"] is None
    assert tunnel_data["created"] is None
    assert tunnel_data["expires_at"] is not None


def test_tunnel_list_passes_label_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    captured: dict[str, Any] = {}

    class FakeTunnelClient:
        async def list_tunnels_page(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return SimpleNamespace(
                tunnels=[
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
                ],
                total=1,
                page=1,
                per_page=50,
                has_next=False,
            )

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


def test_tunnel_list_json_outputs_empty_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")

    class FakeTunnelClient:
        async def list_tunnels_page(self, **kwargs: Any) -> Any:
            return SimpleNamespace(tunnels=[], total=0, page=1, per_page=50, has_next=False)

        async def close(self) -> None:
            return None

    monkeypatch.setattr("prime_cli.commands.tunnel.TunnelClient", FakeTunnelClient)

    result = runner.invoke(app, ["tunnel", "list", "--output", "json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "tunnels": [],
        "total": 0,
        "page": 1,
        "per_page": 50,
        "has_next": False,
    }


def test_tunnel_stop_by_label_uses_bulk_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    captured: dict[str, Any] = {}

    class FakeTunnelClient:
        config = SimpleNamespace(user_id="user-1", team_id=None)

        async def bulk_delete_tunnels(self, **kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return {"succeeded": ["t-test123"], "failed": [], "message": "ok"}

        async def close(self) -> None:
            return None

    monkeypatch.setattr("prime_cli.commands.tunnel.TunnelClient", FakeTunnelClient)

    result = runner.invoke(app, ["tunnel", "stop", "--label", "dev", "--yes"])

    assert result.exit_code == 0, result.output
    assert captured["labels"] == ["dev"]
    assert captured["user_id"] == "user-1"
    assert captured["all_users"] is False


def test_tunnel_stop_all_lists_then_bulk_deletes_explicit_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    captured: dict[str, Any] = {}

    class FakeTunnelClient:
        config = SimpleNamespace(user_id="user-1", team_id=None)

        async def list_tunnels(self, **kwargs: Any) -> list[Any]:
            captured["list_kwargs"] = kwargs
            return [
                SimpleNamespace(tunnel_id="t-owned", user_id="user-1"),
                SimpleNamespace(tunnel_id="t-other", user_id="user-2"),
            ]

        async def bulk_delete_tunnels(self, tunnel_ids: list[str]) -> dict[str, Any]:
            captured["tunnel_ids"] = tunnel_ids
            return {"succeeded": tunnel_ids, "failed": [], "message": "ok"}

        async def close(self) -> None:
            return None

    monkeypatch.setattr("prime_cli.commands.tunnel.TunnelClient", FakeTunnelClient)

    result = runner.invoke(app, ["tunnel", "stop", "--all", "--yes"])

    assert result.exit_code == 0, result.output
    assert captured["list_kwargs"]["team_id"] is None
    assert captured["list_kwargs"]["page"] == 1
    assert captured["list_kwargs"]["per_page"] == 1000
    assert captured["tunnel_ids"] == ["t-owned"]


def test_tunnel_stop_all_users_uses_configured_team_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    captured: dict[str, Any] = {}

    class FakeTunnelClient:
        config = SimpleNamespace(user_id=None, team_id="team-1")

        async def list_tunnels(self, **kwargs: Any) -> list[Any]:
            captured["list_kwargs"] = kwargs
            return [
                SimpleNamespace(tunnel_id="t-owned", user_id="user-1"),
                SimpleNamespace(tunnel_id="t-other", user_id="user-2"),
            ]

        async def bulk_delete_tunnels(self, tunnel_ids: list[str]) -> dict[str, Any]:
            captured["tunnel_ids"] = tunnel_ids
            return {"succeeded": tunnel_ids, "failed": [], "message": "ok"}

        async def close(self) -> None:
            return None

    monkeypatch.setattr("prime_cli.commands.tunnel.TunnelClient", FakeTunnelClient)

    result = runner.invoke(app, ["tunnel", "stop", "--all", "--all-users", "--yes"])

    assert result.exit_code == 0, result.output
    assert captured["list_kwargs"]["team_id"] == "team-1"
    assert captured["tunnel_ids"] == ["t-owned", "t-other"]


def test_tunnel_stop_all_paginates_before_bulk_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    captured: dict[str, Any] = {"pages": []}

    page_1 = [SimpleNamespace(tunnel_id=f"t-{i}", user_id="user-1") for i in range(1, 1001)]
    page_2 = [SimpleNamespace(tunnel_id="t-1001", user_id="user-1")]

    class FakeTunnelClient:
        config = SimpleNamespace(user_id="user-1", team_id=None)

        async def list_tunnels(self, **kwargs: Any) -> list[Any]:
            captured["pages"].append(kwargs["page"])
            if kwargs["page"] == 1:
                return page_1
            if kwargs["page"] == 2:
                return page_2
            return []

        async def bulk_delete_tunnels(self, tunnel_ids: list[str]) -> dict[str, Any]:
            captured["tunnel_ids"] = tunnel_ids
            return {"succeeded": tunnel_ids, "failed": [], "message": "ok"}

        async def close(self) -> None:
            return None

    monkeypatch.setattr("prime_cli.commands.tunnel.TunnelClient", FakeTunnelClient)

    result = runner.invoke(app, ["tunnel", "stop", "--all", "--yes"])

    assert result.exit_code == 0, result.output
    assert captured["pages"] == [1, 2]
    assert captured["tunnel_ids"] == [f"t-{i}" for i in range(1, 1002)]


def test_tunnel_stop_all_noops_when_no_active_tunnels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")

    class FakeTunnelClient:
        config = SimpleNamespace(user_id="user-1", team_id=None)

        async def list_tunnels(self, **kwargs: Any) -> list[Any]:
            return []

        async def bulk_delete_tunnels(self, tunnel_ids: list[str]) -> dict[str, Any]:
            raise AssertionError("bulk delete should not be called for an empty --all result")

        async def close(self) -> None:
            return None

    monkeypatch.setattr("prime_cli.commands.tunnel.TunnelClient", FakeTunnelClient)

    result = runner.invoke(app, ["tunnel", "stop", "--all", "--yes"])

    assert result.exit_code == 0, result.output
    assert "No active tunnels to stop" in result.output
    assert "--all-users" in result.output


def test_tunnel_stop_all_requires_current_user_for_only_mine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")

    class FakeTunnelClient:
        config = SimpleNamespace(user_id=None, team_id=None)

        async def bulk_delete_tunnels(self, **kwargs: Any) -> dict[str, Any]:
            raise AssertionError("bulk delete should not be called without user_id")

        async def close(self) -> None:
            return None

    monkeypatch.setattr("prime_cli.commands.tunnel.TunnelClient", FakeTunnelClient)

    result = runner.invoke(app, ["tunnel", "stop", "--all", "--yes"])

    assert result.exit_code == 1
    assert "Cannot resolve current user ID" in result.output
