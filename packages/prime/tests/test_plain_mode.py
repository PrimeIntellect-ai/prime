import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from prime_cli.commands.login import fetch_and_select_team
from prime_cli.main import app
from prime_cli.utils import strip_ansi
from prime_cli.utils.plain import set_plain_mode
from typer.testing import CliRunner

runner = CliRunner()


def _assert_plain_output(output: str) -> None:
    stripped = strip_ansi(output)
    assert "\x1b" not in stripped
    assert not any(ch in stripped for ch in "╭╰│─")


def _availability_response() -> dict[str, Any]:
    data_path = Path(__file__).parent / "data" / "availability_response.json"
    return json.loads(data_path.read_text())


@pytest.fixture
def mock_availability_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_API_KEY", "dummy")
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    response = _availability_response()

    def mock_get(self: Any, endpoint: str, params: Any = None) -> dict[str, Any]:
        if endpoint in {"/availability", "availability"}:
            return response
        return response

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)


def test_plain_root_help_has_no_rich_borders() -> None:
    result = runner.invoke(app, ["--plain", "--help"])

    assert result.exit_code == 0
    _assert_plain_output(result.output)
    assert "--plain" in result.output
    assert "Commands:" in result.output


def test_plain_eval_help_has_no_rich_borders() -> None:
    result = runner.invoke(app, ["eval", "--plain", "--help"])

    assert result.exit_code == 0
    _assert_plain_output(result.output)
    assert "Usage: prime eval" in result.output


def test_plain_availability_list_renders_dense_text(mock_availability_api: None) -> None:
    result = runner.invoke(
        app,
        ["availability", "list", "--plain"],
        env={"COLUMNS": "200", "LINES": "50"},
    )

    assert result.exit_code == 0, result.output
    _assert_plain_output(result.output)
    assert "Available GPU Resources" in result.output
    assert "ID" in result.output
    assert "GPU Type" in result.output
    assert "To deploy a pod with one of these configurations:" in result.output


def test_plain_json_output_is_unchanged(mock_availability_api: None) -> None:
    plain_result = runner.invoke(
        app,
        ["availability", "list", "--plain", "--output", "json"],
        env={"COLUMNS": "200", "LINES": "50"},
    )
    regular_result = runner.invoke(
        app,
        ["availability", "list", "--output", "json"],
        env={"COLUMNS": "200", "LINES": "50"},
    )

    assert plain_result.exit_code == 0, plain_result.output
    assert regular_result.exit_code == 0, regular_result.output
    assert json.loads(plain_result.output) == json.loads(regular_result.output)


def test_plain_whoami_renders_compact_details(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_API_KEY", "dummy")
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")

    def mock_get(self: Any, endpoint: str, params: Any = None) -> dict[str, Any]:
        assert endpoint == "/user/whoami"
        return {
            "data": {
                "id": "user-123",
                "email": "user@example.com",
                "name": "Prime User",
                "slug": "prime-user",
                "scope": {"pods": {"read": True, "write": False}},
            }
        }

    class FakeConfig:
        team_id = None
        team_name = None
        team_role = None

        def set_user_id(self, user_id: str) -> None:
            self.user_id = user_id

        def update_current_environment_file(self) -> None:
            return None

    monkeypatch.setattr("prime_cli.commands.whoami.APIClient.get", mock_get)
    monkeypatch.setattr("prime_cli.commands.whoami.Config", FakeConfig)

    result = runner.invoke(app, ["whoami", "--plain"])

    assert result.exit_code == 0, result.output
    _assert_plain_output(result.output)
    assert "Account" in result.output
    assert "Type" in result.output
    assert "User ID" in result.output
    assert "Token Permissions" in result.output


def test_plain_login_team_selection_is_unstyled(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class FakeConfig:
        def __init__(self) -> None:
            self.selection: tuple[str, str, str] | None = None

        def set_team(
            self,
            team_id: str | None,
            team_name: str | None = None,
            team_role: str | None = None,
        ) -> None:
            self.selection = (team_id or "", team_name or "", team_role or "")

        def update_current_environment_file(self) -> None:
            return None

    monkeypatch.setattr(
        "prime_cli.commands.login.fetch_teams",
        lambda client: [{"teamId": "team-1", "name": "Core Team", "role": "admin"}],
    )
    monkeypatch.setattr("prime_cli.commands.login.typer.prompt", lambda *args, **kwargs: 2)

    set_plain_mode(True)
    try:
        config = FakeConfig()
        fetch_and_select_team(SimpleNamespace(), config)
    finally:
        set_plain_mode(False)

    output = capsys.readouterr().out
    _assert_plain_output(output)
    assert "Select:" in output
    assert "(1) Personal" in output
    assert "(2) Core Team (role: admin)" in output
    assert config.selection == ("team-1", "Core Team", "admin")


def test_plain_watch_appends_snapshots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_API_KEY", "dummy")
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")

    class FakePod:
        def __init__(self, pod_id: str, name: str) -> None:
            self.id = pod_id
            self.name = name
            self.gpu_type = "H100_80GB"
            self.gpu_count = 1
            self.status = "ACTIVE"
            self.installation_status = "FINISHED"
            self.created_at = "2026-03-19T10:00:00Z"

        def model_dump(self) -> dict[str, Any]:
            return {"id": self.id, "name": self.name, "status": self.status}

    snapshots = [
        SimpleNamespace(data=[FakePod("pod-1", "first")], total_count=1),
        SimpleNamespace(data=[FakePod("pod-2", "second")], total_count=1),
    ]

    def mock_list(self: Any, offset: int = 0, limit: int = 100) -> Any:
        return snapshots.pop(0) if snapshots else SimpleNamespace(data=[], total_count=0)

    sleep_calls = {"count": 0}

    def mock_sleep(seconds: int) -> None:
        sleep_calls["count"] += 1
        if sleep_calls["count"] >= 2:
            raise KeyboardInterrupt

    def forbid_clear(command: str) -> None:
        raise AssertionError(command)

    monkeypatch.setattr("prime_cli.commands.pods.PodsClient.list", mock_list)
    monkeypatch.setattr("prime_cli.commands.pods.os.system", forbid_clear)
    monkeypatch.setattr("time.sleep", mock_sleep)

    result = runner.invoke(app, ["pods", "list", "--watch", "--plain"])

    assert result.exit_code == 0, result.output
    _assert_plain_output(result.output)
    assert result.output.count("Compute Pods (Total: 1)") == 2
    assert "Press Ctrl+C to exit watch mode" in result.output


def test_plain_sandbox_create_uses_plain_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_API_KEY", "dummy")
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")

    def mock_create(self: Any, request: Any) -> Any:
        return SimpleNamespace(id="sbx-plain-123")

    monkeypatch.setattr("prime_cli.commands.sandbox.SandboxClient.create", mock_create)

    result = runner.invoke(
        app,
        [
            "sandbox",
            "create",
            "--gpu-count",
            "1",
            "--gpu-type",
            "H100_80GB",
            "--yes",
            "--plain",
        ],
    )

    assert result.exit_code == 0, result.output
    _assert_plain_output(result.output)
    assert "Creating sandbox..." in result.output
    assert "Successfully created sandbox sbx-plain-123" in result.output
