import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

import pytest
from click.testing import CliRunner
from prime_cli.commands.tunnel import _format_tunnel_for_output
from prime_cli.main import app

runner = CliRunner()


def test_prime_requires_runtime_sdks_with_cli_feature_support() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    assert "prime-evals>=0.2.3" in pyproject["project"]["dependencies"]
    assert "prime-tunnel>=0.1.9" in pyproject["project"]["dependencies"]


def test_tunnel_start_cli_passes_labels_to_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    captured: dict[str, Any] = {}

    class DoneEvent:
        def is_set(self) -> bool:
            return True

        def set(self) -> None:
            return None

        async def wait(self) -> None:
            return None

    class FakeTunnel:
        tunnel_id = "t-test123"
        is_running = True
        recent_output: list[str] = []

        def __init__(
            self,
            *,
            local_port: int,
            name: str | None,
            team_id: str | None,
            labels: list[str] | None,
            http_user: str | None,
        ) -> None:
            captured.update(
                {
                    "local_port": local_port,
                    "name": name,
                    "team_id": team_id,
                    "labels": labels,
                    "http_user": http_user,
                }
            )

        async def start(self) -> str:
            return "https://t-test123.example.com"

        async def stop(self) -> None:
            captured["stopped"] = True

    monkeypatch.setattr("prime_cli.commands.tunnel.asyncio.Event", DoneEvent)
    monkeypatch.setattr("prime_tunnel.Tunnel", FakeTunnel)

    result = runner.invoke(
        app,
        ["tunnel", "start", "--port", "8765", "--labels", "dev", "preview"],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "local_port": 8765,
        "name": None,
        "team_id": None,
        "labels": ["dev", "preview"],
        "http_user": None,
        "stopped": True,
    }


def test_tunnel_start_cli_passes_auth_to_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    captured: dict[str, Any] = {}

    class DoneEvent:
        def is_set(self) -> bool:
            return True

        def set(self) -> None:
            return None

        async def wait(self) -> None:
            return None

    class FakeTunnel:
        tunnel_id = "t-test123"
        is_running = True
        recent_output: list[str] = []
        http_password = "generated-by-backend"

        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        async def start(self) -> str:
            return "https://t-test123.example.com"

        async def stop(self) -> None:
            return None

    monkeypatch.setattr("prime_cli.commands.tunnel.asyncio.Event", DoneEvent)
    monkeypatch.setattr("prime_tunnel.Tunnel", FakeTunnel)

    result = runner.invoke(
        app,
        ["tunnel", "start", "--port", "8765", "--auth", "alice"],
    )

    assert result.exit_code == 0, result.output
    assert captured["http_user"] == "alice"
    assert "http_password" not in captured
    assert "Basic auth user:" in result.output
    assert "generated-by-backend" in result.output
    assert "shown only once" in result.output


def test_tunnel_start_cli_rejects_invalid_auth_username(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")

    for bad_auth in ("alice:s3cret", "alice bob", " "):
        result = runner.invoke(
            app,
            ["tunnel", "start", "--port", "8765", "--auth", bad_auth],
        )
        assert result.exit_code == 1, result.output
        assert "Invalid --auth username" in result.output


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


def test_tunnel_start_import_failure_exits_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    monkeypatch.setitem(sys.modules, "prime_tunnel", None)

    result = runner.invoke(app, ["tunnel", "start"])

    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "Traceback" not in result.output


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

    monkeypatch.setattr("prime_tunnel.core.client.TunnelClient", FakeTunnelClient)

    result = runner.invoke(
        app,
        ["tunnel", "list", "--labels", "dev", "--sort-by", "name", "--output", "json"],
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

    monkeypatch.setattr("prime_tunnel.core.client.TunnelClient", FakeTunnelClient)

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

    monkeypatch.setattr("prime_tunnel.core.client.TunnelClient", FakeTunnelClient)

    result = runner.invoke(app, ["tunnel", "stop", "--labels", "dev", "--yes"])

    assert result.exit_code == 0, result.output
    assert captured["labels"] == ["dev"]
    assert captured["user_id"] == "user-1"
    assert captured["all_users"] is False


def test_tunnel_stop_by_label_validates_scope_before_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")

    class FakeTunnelClient:
        config = SimpleNamespace(user_id=None, team_id=None)

        async def bulk_delete_tunnels(self, **kwargs: Any) -> dict[str, Any]:
            raise AssertionError("bulk delete should not be called without user_id")

        async def close(self) -> None:
            return None

    monkeypatch.setattr("prime_tunnel.core.client.TunnelClient", FakeTunnelClient)

    # No --yes: if the scope check ran after the prompt we'd see the confirmation
    # text; instead it must fail fast with a clean error and never prompt.
    result = runner.invoke(app, ["tunnel", "stop", "--labels", "dev"])

    assert result.exit_code == 1
    assert "Cannot resolve current user ID" in result.output
    assert "This action cannot be undone" not in result.output
    assert "Failed to delete" not in result.output


def test_tunnel_stop_all_uses_scoped_bulk_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    captured: dict[str, Any] = {}

    class FakeTunnelClient:
        config = SimpleNamespace(user_id="user-1", team_id=None)

        async def list_tunnels_page(self, **kwargs: Any) -> Any:
            raise AssertionError("--all should not pre-list tunnels; it must scope the delete")

        async def bulk_delete_tunnels(self, **kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return {"succeeded": ["t-owned"], "failed": [], "message": "ok"}

        async def close(self) -> None:
            return None

    monkeypatch.setattr("prime_tunnel.core.client.TunnelClient", FakeTunnelClient)

    result = runner.invoke(app, ["tunnel", "stop", "--all", "--yes"])

    assert result.exit_code == 0, result.output
    assert "tunnel_ids" not in captured
    assert captured["user_id"] == "user-1"
    assert captured["all_users"] is False
    assert captured.get("team_id") is None


def test_tunnel_stop_all_users_uses_configured_team_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    captured: dict[str, Any] = {}

    class FakeTunnelClient:
        config = SimpleNamespace(user_id=None, team_id="team-1")

        async def list_tunnels_page(self, **kwargs: Any) -> Any:
            raise AssertionError("--all should not pre-list tunnels; it must scope the delete")

        async def bulk_delete_tunnels(self, **kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return {"succeeded": ["t-owned", "t-other"], "failed": [], "message": "ok"}

        async def close(self) -> None:
            return None

    monkeypatch.setattr("prime_tunnel.core.client.TunnelClient", FakeTunnelClient)

    result = runner.invoke(app, ["tunnel", "stop", "--all", "--all-users", "--yes"])

    assert result.exit_code == 0, result.output
    assert "tunnel_ids" not in captured
    assert captured["team_id"] == "team-1"
    assert captured["all_users"] is True
    assert captured.get("user_id") is None


def test_tunnel_stop_by_status_uses_scoped_bulk_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    captured: dict[str, Any] = {}

    class FakeTunnelClient:
        config = SimpleNamespace(user_id="user-1", team_id=None)

        async def list_tunnels_page(self, **kwargs: Any) -> Any:
            raise AssertionError("--status must scope the delete, not pre-list tunnels")

        async def bulk_delete_tunnels(self, **kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return {"succeeded": ["t-disc"], "failed": [], "message": "ok"}

        async def close(self) -> None:
            return None

    monkeypatch.setattr("prime_tunnel.core.client.TunnelClient", FakeTunnelClient)

    # Mixed case is accepted and normalized to the lowercase API enum value.
    result = runner.invoke(app, ["tunnel", "stop", "--status", "DISCONNECTED", "--yes"])

    assert result.exit_code == 0, result.output
    assert captured["status"] == "disconnected"
    assert captured["user_id"] == "user-1"
    assert captured["all_users"] is False
    assert "tunnel_ids" not in captured


def test_tunnel_stop_status_combines_with_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    captured: dict[str, Any] = {}

    class FakeTunnelClient:
        config = SimpleNamespace(user_id="user-1", team_id=None)

        async def bulk_delete_tunnels(self, **kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return {"succeeded": ["t-disc"], "failed": [], "message": "ok"}

        async def close(self) -> None:
            return None

    monkeypatch.setattr("prime_tunnel.core.client.TunnelClient", FakeTunnelClient)

    result = runner.invoke(
        app,
        ["tunnel", "stop", "--labels", "dev", "--status", "disconnected", "--yes"],
    )

    assert result.exit_code == 0, result.output
    assert captured["labels"] == ["dev"]
    assert captured["status"] == "disconnected"
    assert captured["user_id"] == "user-1"


def test_tunnel_stop_invalid_status_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")

    result = runner.invoke(app, ["tunnel", "stop", "--status", "expired", "--yes"])

    assert result.exit_code == 1
    assert "--status must be one of" in result.output


def test_tunnel_stop_status_with_ids_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")

    result = runner.invoke(app, ["tunnel", "stop", "t-123", "--status", "disconnected", "--yes"])

    assert result.exit_code == 1
    assert "Tunnel IDs cannot be combined with --all, --label, or --status" in result.output


def test_tunnel_stop_all_exits_cleanly_when_nothing_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    captured: dict[str, Any] = {}

    class FakeTunnelClient:
        config = SimpleNamespace(user_id="user-1", team_id=None)

        async def bulk_delete_tunnels(self, **kwargs: Any) -> dict[str, Any]:
            captured["called"] = True
            return {"succeeded": [], "failed": [], "message": "ok"}

        async def close(self) -> None:
            return None

    monkeypatch.setattr("prime_tunnel.core.client.TunnelClient", FakeTunnelClient)

    result = runner.invoke(app, ["tunnel", "stop", "--all", "--yes"])

    assert result.exit_code == 0, result.output
    assert captured.get("called") is True
    assert "Processed 0 tunnel(s)" in result.output


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

    monkeypatch.setattr("prime_tunnel.core.client.TunnelClient", FakeTunnelClient)

    result = runner.invoke(app, ["tunnel", "stop", "--all", "--yes"])

    assert result.exit_code == 1
    assert "Cannot resolve current user ID" in result.output
