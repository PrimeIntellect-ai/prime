import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
from prime_cli.core import Config
from prime_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()

TEST_ENV = {
    "COLUMNS": "200",
    "LINES": "50",
    "PRIME_DISABLE_VERSION_CHECK": "1",
    "PRIME_TEAM_ID": "",
}

USER_ID = "cmakj7hyo002rz091pdjngniy"
USER_NAME = "Ada Lovelace"


@pytest.fixture
def temp_home(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("PRIME_USER_ID", raising=False)
    monkeypatch.delenv("PRIME_API_KEY", raising=False)
    monkeypatch.delenv("PRIME_CONTEXT", raising=False)
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    return tmp_path


def _user_row(output: str) -> str:
    match = re.search(r"│\s*User\s*│\s*(.*?)\s*│", output)
    assert match, output
    return match.group(1)


class TestConfigViewUser:
    def test_view_shows_name_and_id(self, temp_home: Path) -> None:
        Config().set_user_id(USER_ID, user_name=USER_NAME)

        result = runner.invoke(app, ["config", "view"], env=TEST_ENV)

        assert result.exit_code == 0, result.output
        assert _user_row(result.output) == f"{USER_NAME} ({USER_ID})"

    def test_view_falls_back_to_id_without_name(self, temp_home: Path) -> None:
        Config().set_user_id(USER_ID)

        result = runner.invoke(app, ["config", "view"], env=TEST_ENV)

        assert result.exit_code == 0, result.output
        assert _user_row(result.output) == USER_ID

    def test_view_shows_not_set_without_user(self, temp_home: Path) -> None:
        result = runner.invoke(app, ["config", "view"], env=TEST_ENV)

        assert result.exit_code == 0, result.output
        assert _user_row(result.output) == "Not set"

    def test_view_ignores_stored_name_when_user_id_from_env(
        self, temp_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        Config().set_user_id(USER_ID, user_name=USER_NAME)
        monkeypatch.setenv("PRIME_USER_ID", "someone-else")

        result = runner.invoke(app, ["config", "view"], env=TEST_ENV)

        assert result.exit_code == 0, result.output
        assert _user_row(result.output) == "someone-else (from env var)"
        assert USER_NAME not in result.output


class TestUserNamePersistence:
    def test_whoami_stores_user_name(
        self, temp_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PRIME_API_KEY", "test-key")

        def mock_get(
            self: Any, endpoint: str, params: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            if endpoint == "/user/whoami":
                return {
                    "data": {
                        "id": USER_ID,
                        "name": USER_NAME,
                        "email": "ada@example.com",
                        "scope": {},
                    }
                }
            return {"data": []}

        monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

        result = runner.invoke(app, ["whoami"], env=TEST_ENV)

        assert result.exit_code == 0, result.output
        stored = json.loads((temp_home / ".prime" / "config.json").read_text())
        assert stored["user_id"] == USER_ID
        assert stored["user_name"] == USER_NAME

    def test_clearing_user_id_clears_name(self, temp_home: Path) -> None:
        config = Config()
        config.set_user_id(USER_ID, user_name=USER_NAME)
        config.set_user_id(None)

        assert config.user_id is None
        assert config.user_name is None

    def test_user_name_round_trips_through_saved_environment(self, temp_home: Path) -> None:
        config = Config()
        config.set_user_id(USER_ID, user_name=USER_NAME)
        config.save_environment("staging")
        config.set_user_id(None)

        assert config.load_environment("staging")
        assert config.user_id == USER_ID
        assert config.user_name == USER_NAME

    def test_saved_environment_omits_name_when_user_id_from_env(
        self, temp_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = Config()
        config.set_user_id(USER_ID, user_name=USER_NAME)
        monkeypatch.setenv("PRIME_USER_ID", "someone-else")

        config.save_environment("staging")

        saved = json.loads((temp_home / ".prime" / "environments" / "staging.json").read_text())
        assert saved["user_id"] == "someone-else"
        assert saved["user_name"] is None

    def test_logout_clears_user_name(self, temp_home: Path) -> None:
        config = Config()
        config.set_api_key("pit_test_key")
        config.set_user_id(USER_ID, user_name=USER_NAME)

        result = runner.invoke(app, ["logout", "--yes"], env=TEST_ENV)

        assert result.exit_code == 0, result.output
        assert Config().user_name is None
