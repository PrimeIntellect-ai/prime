from typing import Any, Dict, Optional

import pytest
from prime_cli.client import APIError
from prime_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()

TEST_ENV = {
    "COLUMNS": "200",
    "LINES": "50",
    "PRIME_DISABLE_VERSION_CHECK": "1",
}


@pytest.fixture
def temp_home(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))


class TestConfigSetTeamId:
    def test_set_team_id_requires_argument(self, temp_home: None) -> None:
        result = runner.invoke(app, ["config", "set-team-id"], env=TEST_ENV)

        assert result.exit_code != 0
        assert "Missing argument 'TEAM_ID'" in result.output

    def test_set_team_id_populates_name_from_paginated_team_lookup(
        self, temp_home: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PRIME_API_KEY", "test-key")
        monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

        target_team = {
            "teamId": "cmf0ohr9s0026ilerf3w68s6z",
            "name": "Page Two Team",
            "slug": "page-two",
            "role": "admin",
            "createdAt": "2026-01-15T10:00:00Z",
        }
        first_page = [
            {
                "teamId": f"cmf0ohr9s0026ilerf3w68{idx:02d}",
                "name": f"Team {idx}",
                "slug": f"team-{idx}",
                "role": "member",
                "createdAt": "2026-01-15T10:00:00Z",
            }
            for idx in range(100)
        ]

        def mock_get(
            self: Any, endpoint: str, params: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            if endpoint == "/user/teams":
                offset = (params or {}).get("offset", 0)
                limit = (params or {}).get("limit", 100)
                if offset == 0:
                    return {"data": first_page[:limit], "total_count": 101}
                if offset == 100:
                    return {"data": [target_team], "total_count": 101}
            return {"data": []}

        monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

        result = runner.invoke(
            app,
            ["config", "set-team-id", target_team["teamId"]],
            env=TEST_ENV,
        )

        assert result.exit_code == 0, result.output
        expected_message = (
            f"Team '{target_team['name']}' ({target_team['teamId']}) configured successfully!"
        )
        assert expected_message in result.output

    def test_set_team_id_rejects_team_the_user_is_not_a_member_of(
        self, temp_home: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PRIME_API_KEY", "test-key")
        monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

        def mock_get(
            self: Any, endpoint: str, params: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            if endpoint == "/user/teams":
                return {
                    "data": [
                        {
                            "teamId": "cmf0ohr9s0026ilerf3w68s6z",
                            "name": "My Team",
                            "slug": "my-team",
                            "role": "member",
                            "createdAt": "2026-01-15T10:00:00Z",
                        }
                    ],
                    "total_count": 1,
                }
            return {"data": []}

        monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

        stranger = "cmupu62cp000irv83cjhqiqwn"
        result = runner.invoke(app, ["config", "set-team-id", stranger], env=TEST_ENV)

        # Saving a team you don't belong to silently bills the personal account
        # at dispatch, so this has to fail loudly instead of succeeding.
        assert result.exit_code == 1, result.output
        assert f"not a member of team '{stranger}'" in result.output

        view = runner.invoke(app, ["config", "view"], env=TEST_ENV)
        assert stranger not in view.output

    def test_set_team_id_saves_with_a_warning_when_lookup_is_unavailable(
        self, temp_home: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PRIME_API_KEY", "test-key")
        monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

        def mock_get(
            self: Any, endpoint: str, params: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            raise APIError("connection refused")

        monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

        team_id = "cmf0ohr9s0026ilerf3w68s6z"
        result = runner.invoke(app, ["config", "set-team-id", team_id], env=TEST_ENV)

        # An unreachable API is not evidence of a bad team id, so don't block.
        assert result.exit_code == 0, result.output
        assert f"Team ID '{team_id}' configured successfully!" in result.output
        assert "couldn't reach the API" in result.output
