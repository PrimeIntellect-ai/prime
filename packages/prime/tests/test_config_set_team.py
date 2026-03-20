from typing import Any, Dict, Optional

import pytest
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
