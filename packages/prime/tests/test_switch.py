import io
from typing import Any, Dict, Optional

import pytest
from prime_cli.main import app
from prime_cli.utils.plain import get_console
from typer.testing import CliRunner

runner = CliRunner()

TEST_ENV = {
    "COLUMNS": "200",
    "LINES": "50",
    "PRIME_DISABLE_VERSION_CHECK": "1",
    "PRIME_TEAM_ID": "",
}


def test_get_console_accepts_explicit_plain_rendering_kwargs() -> None:
    console = get_console(
        file=io.StringIO(),
        markup=False,
        highlight=False,
        no_color=True,
        emoji=False,
    )

    assert console is not None


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

    def test_switch_accepts_plain_flag(self, temp_home: None, mock_teams_api: None) -> None:
        result = runner.invoke(app, ["switch", "--plain", "personal"], env=TEST_ENV)

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

    def test_switch_interactive_selection(
        self, temp_home: None, mock_teams_api: None, keys: Any
    ) -> None:
        keys.select(1)
        result = runner.invoke(app, ["switch"], env=TEST_ENV)

        assert result.exit_code == 0, result.output
        assert "Switched to team 'Prime Team'." in result.output

    def test_account_choices_marks_current_team_without_slug(self) -> None:
        from prime_cli.commands.switch import _account_choices

        teams = [{"teamId": "team-1", "name": "New team", "slug": None, "role": "ADMIN"}]
        labels = [choice.title for choice in _account_choices(teams, "team-1")]

        assert "Personal" in labels
        assert "Personal (current)" not in labels
        assert "New team (role: admin) (current)" in labels
        assert not any("slug:" in label for label in labels)

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
