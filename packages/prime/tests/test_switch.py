from typing import Any, Dict, Optional

import pytest
from prime_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()

TEST_ENV = {
    "COLUMNS": "200",
    "LINES": "50",
    "PRIME_DISABLE_VERSION_CHECK": "1",
    "PRIME_TEAM_ID": "",
}


@pytest.fixture
def temp_home(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.fixture
def mock_teams_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_API_KEY", "test-key")

    teams = [
        {
            "teamId": "cmf0ohr9s0026ilerf3w68s6n",
            "name": "Prime Team",
            "slug": "prime",
            "role": "admin",
            "createdAt": "2026-01-15T10:00:00Z",
        },
        {
            "teamId": "cmf0ohr9s0026ilerf3w68s6m",
            "name": "Research Team",
            "slug": "research",
            "role": "member",
            "createdAt": "2026-01-15T10:00:00Z",
        },
    ]

    def mock_get(
        self: Any, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if endpoint == "/user/teams":
            return {"data": teams}
        if endpoint == "/user/whoami":
            return {"data": {"id": "user-123"}}
        return {"data": []}

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))


class TestSwitchCommand:
    def test_switch_to_personal(self, temp_home: None, mock_teams_api: None) -> None:
        result = runner.invoke(app, ["switch", "personal"], env=TEST_ENV)

        assert result.exit_code == 0, result.output
        assert "Switched to personal account." in result.output

    def test_switch_to_personal_skips_team_fetch(
        self, temp_home: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("PRIME_API_KEY", raising=False)
        monkeypatch.setattr(
            "prime_cli.commands.switch.fetch_teams",
            lambda _client: pytest.fail("fetch_teams should not be called for personal switch"),
        )
        monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

        result = runner.invoke(app, ["switch", "personal"], env=TEST_ENV)

        assert result.exit_code == 0, result.output
        assert "Switched to personal account." in result.output

    def test_switch_to_team_slug(self, temp_home: None, mock_teams_api: None) -> None:
        result = runner.invoke(app, ["switch", "prime"], env=TEST_ENV)

        assert result.exit_code == 0, result.output
        assert "Switched to team 'Prime Team'." in result.output

    def test_switch_to_team_id_fallback(self, temp_home: None, mock_teams_api: None) -> None:
        result = runner.invoke(app, ["switch", "cmf0ohr9s0026ilerf3w68s6n"], env=TEST_ENV)

        assert result.exit_code == 0, result.output
        assert "Switched to team 'Prime Team'." in result.output

    def test_switch_to_unknown_team_slug_shows_available_teams(
        self, temp_home: None, mock_teams_api: None
    ) -> None:
        result = runner.invoke(app, ["switch", "unknown"], env=TEST_ENV)

        assert result.exit_code == 1, result.output
        assert "Team 'unknown' not found." in result.output
        assert "Available teams: prime, research" in result.output

    def test_switch_does_not_match_none_slug(
        self, temp_home: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PRIME_API_KEY", "test-key")

        def mock_get(
            self: Any, endpoint: str, params: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            if endpoint == "/user/teams":
                return {
                    "data": [
                        {
                            "teamId": "cmf0ohr9s0026ilerf3w68s6n",
                            "name": "New team",
                            "slug": None,
                            "role": "ADMIN",
                            "createdAt": "2026-01-15T10:00:00Z",
                        }
                    ]
                }
            if endpoint == "/user/whoami":
                return {"data": {"id": "user-123"}}
            return {"data": []}

        monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)
        monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

        result = runner.invoke(app, ["switch", "none"], env=TEST_ENV)

        assert result.exit_code == 1, result.output
        assert "Team 'none' not found." in result.output

    def test_switch_interactive_selection(self, temp_home: None, mock_teams_api: None) -> None:
        result = runner.invoke(app, ["switch"], input="2\n", env=TEST_ENV)

        assert result.exit_code == 0, result.output
        assert "Switch account:" in result.output
        assert "Prime Team (slug: prime, role: admin)" in result.output
        assert "Switched to team 'Prime Team'." in result.output

    def test_switch_interactive_handles_missing_slug_without_double_current(
        self, temp_home: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        monkeypatch.setenv("PRIME_API_KEY", "test-key")

        config_dir = tmp_path / ".prime"
        config_dir.mkdir()
        (config_dir / "environments").mkdir()
        (config_dir / "config.json").write_text(
            '{"api_key":"","team_id":"cmf0ohr9s0026ilerf3w68s6n","team_name":"New team","team_role":"ADMIN","user_id":null,"base_url":"https://api.primeintellect.ai","frontend_url":"https://app.primeintellect.ai","inference_url":"https://api.pinference.ai/api/v1","ssh_key_path":"~/.ssh/id_rsa","current_environment":"production"}'
        )

        def mock_get(
            self: Any, endpoint: str, params: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            if endpoint == "/user/teams":
                return {
                    "data": [
                        {
                            "teamId": "cmf0ohr9s0026ilerf3w68s6n",
                            "name": "New team",
                            "slug": None,
                            "role": "ADMIN",
                            "createdAt": "2026-01-15T10:00:00Z",
                        }
                    ]
                }
            return {"data": []}

        monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)
        monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

        result = runner.invoke(app, ["switch"], input="1\n", env=TEST_ENV)

        assert result.exit_code == 0, result.output
        assert "Personal (current)" not in result.output
        assert "New team (role: admin) (current)" in result.output
        assert "slug:" not in result.output

    def test_switch_fails_when_team_is_forced_by_environment(
        self, temp_home: None, mock_teams_api: None
    ) -> None:
        result = runner.invoke(
            app,
            ["switch", "personal"],
            env={
                "COLUMNS": "200",
                "LINES": "50",
                "PRIME_DISABLE_VERSION_CHECK": "1",
                "PRIME_TEAM_ID": "cmf0ohr9s0026ilerf3w68s6n",
            },
        )

        assert result.exit_code == 1, result.output
        assert "PRIME_TEAM_ID is set in your environment" in result.output
