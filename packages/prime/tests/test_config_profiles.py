import json
import os
from pathlib import Path
from typing import Any

import pytest
from prime_cli.core import Config
from prime_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()

TEST_ENV = {
    "COLUMNS": "200",
    "LINES": "50",
    "PRIME_DISABLE_VERSION_CHECK": "1",
}


@pytest.fixture
def temp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    Config.set_context_from_cli_option(False)
    for name in (
        "PRIME_API_KEY",
        "PRIME_TEAM_ID",
        "PRIME_USER_ID",
        "PRIME_API_BASE_URL",
        "PRIME_BASE_URL",
        "PRIME_FRONTEND_URL",
        "PRIME_INFERENCE_URL",
        "PRIME_CONTEXT",
    ):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def _saved_profile(home: Path, name: str) -> dict[str, Any]:
    return json.loads((home / ".prime" / "environments" / f"{name}.json").read_text())


def test_save_use_restores_saved_team(temp_home: Path) -> None:
    config = Config()
    config.set_api_key("key-team-one")
    config.set_team("team-one", team_name="Team One", team_role="ADMIN")
    config.save_environment("team1")

    config.set_api_key("key-team-two")
    config.set_team("team-two", team_name="Team Two", team_role="MEMBER")
    config.save_environment("team2")

    assert config.load_environment("team1")

    reloaded = Config()
    assert reloaded.api_key == "key-team-one"
    assert reloaded.team_id == "team-one"
    assert reloaded.team_name == "Team One"
    assert reloaded.current_environment == "team1"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("PRIME_API_KEY", "env-key"),
        ("PRIME_API_KEY", "   "),
        ("PRIME_USER_ID", ""),
        ("PRIME_CONTEXT", "team1"),
    ],
)
def test_config_use_fails_when_env_override_masks_profile(
    temp_home: Path, name: str, value: str
) -> None:
    config = Config()
    config.save_environment("team1")
    config.save_environment("staging")

    result = runner.invoke(
        app,
        ["config", "use", "staging"],
        env={**TEST_ENV, name: value},
    )

    assert result.exit_code == 1, result.output
    assert f"{name} is set in your environment" in result.output
    assert "prime config use staging" in result.output


def test_context_option_does_not_block_config_use(temp_home: Path) -> None:
    config = Config()
    config.set_api_key("key-team-one")
    config.save_environment("team1")
    config.set_api_key("key-staging")
    config.save_environment("staging")

    result = runner.invoke(app, ["--context", "team1", "config", "use", "staging"], env=TEST_ENV)

    assert result.exit_code == 0, result.output

    reloaded = Config()
    assert reloaded.current_environment == "staging"
    assert reloaded.api_key == "key-staging"
    assert os.getenv("PRIME_CONTEXT") is None
    assert not Config.context_from_cli_option()


def test_active_profile_update_uses_persisted_values_not_env_overrides(
    temp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = Config()
    config.set_api_key("profile-key")
    config.set_team("team-one", team_name="Team One", team_role="ADMIN")
    config.save_environment("profile")
    assert config.load_environment("profile")

    monkeypatch.setenv("PRIME_API_KEY", "env-key")

    config.set_team("team-two", team_name="Team Two", team_role="MEMBER")
    config.update_current_environment_file()

    saved = _saved_profile(temp_home, "profile")
    assert saved["api_key"] == "profile-key"
    assert saved["team_id"] == "team-two"


def test_set_api_key_syncs_active_saved_profile_after_whoami(
    temp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeAPIClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def get(self, endpoint: str) -> dict[str, dict[str, str]]:
            assert endpoint == "/user/whoami"
            assert self.api_key == "new-key"
            return {"data": {"id": "user-new"}}

    monkeypatch.setattr("prime_cli.commands.config.APIClient", FakeAPIClient)

    config = Config()
    config.set_api_key("old-key")
    config.set_user_id("user-old")
    config.save_environment("profile")
    assert config.load_environment("profile")

    result = runner.invoke(app, ["config", "set-api-key", "new-key"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    saved = _saved_profile(temp_home, "profile")
    assert saved["api_key"] == "new-key"
    assert saved["user_id"] == "user-new"
