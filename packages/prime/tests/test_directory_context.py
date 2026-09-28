import json
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
from prime_cli.core import Config
from prime_cli.core.config import find_local_context_file
from prime_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()

TEST_ENV = {"COLUMNS": "200", "PRIME_DISABLE_VERSION_CHECK": "1"}
EDISON = "cmf0ohr9s0026ilerf3w68s6n"
ACME = "cmf0ohr9s0026ilerf3w68s6m"
GLOBAL_TEAM = "cmf0ohr9s0026ilerf3w68s6g"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    for name in ("PRIME_CONTEXT", "PRIME_API_KEY", "PRIME_TEAM_ID", "PRIME_USER_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    config = Config(use_context=False)
    config.set_api_key("global-key")
    config.set_team(GLOBAL_TEAM, team_name="Global Team", team_role="admin")
    config.set_user_id("global-user", user_name="Global User")
    return home


@pytest.fixture
def repo(home: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = home / "code" / "edison"
    (repo / "src" / "pkg").mkdir(parents=True)
    monkeypatch.chdir(repo)
    return repo


@pytest.fixture
def teams_api(monkeypatch: pytest.MonkeyPatch) -> None:
    teams = [
        {"teamId": EDISON, "name": "Edison", "slug": "edison", "role": "member"},
        {"teamId": ACME, "name": "Acme", "slug": "acme", "role": "admin"},
    ]

    def mock_get(
        self: Any, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if endpoint == "/user/teams":
            return {"data": teams}
        return {"data": []}

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)


def _pin(directory: Path, data: dict) -> Path:
    path = directory / ".prime" / "context.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def _global(home: Path) -> dict:
    return json.loads((home / ".prime" / "config.json").read_text())


def _save_env(name: str, **fields: Any) -> Path:
    config = Config(use_context=False)
    config.save_environment(name)
    path = config.environments_dir / f"{name}.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), **fields}))
    return path


def test_without_directory_context_uses_global_config(repo: Path) -> None:
    config = Config()

    assert config.local_context_file is None
    assert config.team_id == GLOBAL_TEAM
    assert config.api_key == "global-key"


def test_team_pin_applies_to_subdirectories(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pin = _pin(repo, {"team_id": EDISON, "team_name": "Edison"})
    monkeypatch.chdir(repo / "src" / "pkg")

    config = Config()

    assert config.local_context_file == pin.resolve()
    assert config.team_id == EDISON
    assert config.team_name == "Edison"
    assert config.api_key == "global-key"
    assert config.user_id == "global-user"


def test_null_team_pin_selects_personal_account(repo: Path) -> None:
    _pin(repo, {"team_id": None})

    config = Config()

    assert config.team_id is None
    assert config.team_name is None


def test_nearest_directory_context_wins(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _pin(repo, {"team_id": EDISON})
    _pin(repo / "src", {"team_id": ACME})
    monkeypatch.chdir(repo / "src" / "pkg")

    assert Config().team_id == ACME


def test_env_vars_and_temporary_context_outrank_directory_context(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pin(repo, {"team_id": EDISON})
    _save_env("dev", team_id=ACME, api_key="dev-key")

    monkeypatch.setenv("PRIME_CONTEXT", "dev")
    config = Config()
    assert config.local_context_file is None
    assert config.team_id == ACME

    monkeypatch.delenv("PRIME_CONTEXT")
    monkeypatch.setenv("PRIME_TEAM_ID", GLOBAL_TEAM)
    assert Config().team_id == GLOBAL_TEAM


def test_home_prime_directory_is_never_a_directory_context(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pin(home, {"team_id": EDISON})
    monkeypatch.chdir(home)

    assert find_local_context_file() is None
    assert Config().team_id == GLOBAL_TEAM


def test_writes_under_team_pin_do_not_leak_the_pinned_team(repo: Path, home: Path) -> None:
    _pin(repo, {"team_id": EDISON, "team_name": "Edison"})
    env_file = _save_env("dev")
    Config(use_context=False).set_current_environment("dev")

    config = Config()
    config.set_api_key("rotated-key")
    config.update_current_environment_file()

    assert _global(home)["api_key"] == "rotated-key"
    assert _global(home)["team_id"] == GLOBAL_TEAM
    assert json.loads(env_file.read_text())["team_id"] == GLOBAL_TEAM
    assert config.team_id == EDISON


def test_context_pin_reads_and_writes_that_context(repo: Path, home: Path) -> None:
    env_file = _save_env(
        "customer", api_key="customer-key", team_id=ACME, base_url="https://api.customer.example"
    )
    _pin(repo, {"context": "customer"})
    global_before = _global(home)

    config = Config()
    assert config.api_key == "customer-key"
    assert config.team_id == ACME
    assert config.base_url == "https://api.customer.example"
    assert config.current_environment == "customer"

    config.set_api_key("customer-key-2")
    config.set_user_id("customer-user", user_name="Customer")
    config.update_current_environment_file()

    saved = json.loads(env_file.read_text())
    assert saved["api_key"] == "customer-key-2"
    assert saved["user_id"] == "customer-user"
    assert _global(home) == global_before
    assert Config().api_key == "customer-key-2"


def test_context_and_team_pin_combine(repo: Path) -> None:
    _save_env("customer", api_key="customer-key", team_id=ACME)
    _pin(repo, {"context": "customer", "team_id": EDISON})

    config = Config()

    assert config.api_key == "customer-key"
    assert config.team_id == EDISON


def test_unknown_pinned_context_is_an_error(repo: Path) -> None:
    pin = _pin(repo, {"context": "missing"})

    with pytest.raises(ValueError, match="Unknown context 'missing'"):
        Config()

    result = runner.invoke(app, ["config", "view"], env=TEST_ENV)
    assert result.exit_code == 1
    assert "Unknown context 'missing'" in result.output
    assert str(pin) in result.output.replace("\n", "")


@pytest.mark.parametrize(
    "content",
    ["[]", "{not json", json.dumps({"context": "../x"}), json.dumps({"team_id": 3})],
)
def test_malformed_directory_context_is_an_error(repo: Path, content: str) -> None:
    path = repo / ".prime" / "context.json"
    path.parent.mkdir()
    path.write_text(content)

    with pytest.raises(ValueError, match="context.json"):
        Config()


def test_switch_local_pins_team_without_touching_global(
    repo: Path, home: Path, teams_api: None
) -> None:
    result = runner.invoke(app, ["switch", "edison", "--local"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert "Switched to team 'Edison' for" in result.output
    pin = json.loads((repo / ".prime" / "context.json").read_text())
    assert pin == {"team_id": EDISON, "team_name": "Edison"}
    assert _global(home)["team_id"] == GLOBAL_TEAM


def test_switch_inside_pinned_directory_updates_the_pin(
    repo: Path, home: Path, teams_api: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    pin = _pin(repo, {"team_id": EDISON, "team_name": "Edison"})
    monkeypatch.chdir(repo / "src")

    result = runner.invoke(app, ["switch", "personal"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert json.loads(pin.read_text()) == {"team_id": None}
    assert not (repo / "src" / ".prime").exists()
    assert _global(home)["team_id"] == GLOBAL_TEAM
    assert Config().team_id is None


def test_switch_global_inside_pinned_directory_changes_global(
    repo: Path, home: Path, teams_api: None
) -> None:
    pin = _pin(repo, {"team_id": EDISON})

    result = runner.invoke(app, ["switch", "acme", "--global"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert "still selects this directory's account" in result.output
    assert _global(home)["team_id"] == ACME
    assert json.loads(pin.read_text()) == {"team_id": EDISON}


def test_switch_rejects_local_and_global_together(repo: Path, teams_api: None) -> None:
    result = runner.invoke(app, ["switch", "edison", "--local", "--global"], env=TEST_ENV)

    assert result.exit_code == 1
    assert not (repo / ".prime").exists()


def test_set_team_id_inside_pinned_directory_updates_the_pin(
    repo: Path, home: Path, teams_api: None
) -> None:
    pin = _pin(repo, {"team_id": EDISON})

    result = runner.invoke(app, ["config", "set-team-id", ACME], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert json.loads(pin.read_text()) == {"team_id": ACME, "team_name": "Acme"}
    assert _global(home)["team_id"] == GLOBAL_TEAM


def test_config_use_local_pins_context_and_drops_pinned_team(repo: Path, home: Path) -> None:
    _save_env("customer", api_key="customer-key")
    pin = _pin(repo, {"team_id": EDISON})

    result = runner.invoke(app, ["config", "use", "customer", "--local"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert json.loads(pin.read_text()) == {"context": "customer"}
    assert _global(home)["current_environment"] == "production"
    assert Config().api_key == "customer-key"


def test_config_use_local_rejects_unknown_environment(repo: Path) -> None:
    result = runner.invoke(app, ["config", "use", "nope", "--local"], env=TEST_ENV)

    assert result.exit_code == 1
    assert "Unknown environment: nope" in result.output
    assert not (repo / ".prime").exists()


def test_config_use_global_inside_pinned_directory_changes_global(repo: Path, home: Path) -> None:
    _save_env("customer", api_key="customer-key")
    pin = _pin(repo, {"context": "customer"})

    result = runner.invoke(app, ["config", "use", "production", "--global"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert json.loads(pin.read_text()) == {"context": "customer"}
    assert _global(home)["api_key"] == "global-key"
    assert _global(home)["current_environment"] == "production"


def test_unpin_removes_the_directory_context(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pin = _pin(repo, {"team_id": EDISON})
    monkeypatch.chdir(repo / "src" / "pkg")

    result = runner.invoke(app, ["config", "unpin"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert not pin.exists()
    assert not pin.parent.exists()
    assert Config().team_id == GLOBAL_TEAM


def test_unpin_keeps_other_files_in_prime_directory(repo: Path) -> None:
    pin = _pin(repo, {"team_id": EDISON})
    (repo / ".prime" / "lab.json").write_text("{}")

    result = runner.invoke(app, ["config", "unpin"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert not pin.exists()
    assert (repo / ".prime" / "lab.json").exists()


def test_config_view_shows_directory_context(repo: Path) -> None:
    pin = _pin(repo, {"team_id": EDISON, "team_name": "Edison"})

    result = runner.invoke(app, ["config", "view"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert "Directory Context" in result.output
    assert str(pin) in result.output
    assert "Edison" in result.output
    assert "(directory context)" in result.output


def test_cannot_delete_the_pinned_context(repo: Path) -> None:
    env_file = _save_env("customer")
    _pin(repo, {"context": "customer"})

    result = runner.invoke(app, ["config", "delete", "customer"], env=TEST_ENV)

    assert result.exit_code == 1
    assert env_file.exists()


def test_logout_in_pinned_context_only_clears_that_context(repo: Path, home: Path) -> None:
    env_file = _save_env("customer", api_key="customer-key")
    _pin(repo, {"context": "customer"})

    result = runner.invoke(app, ["logout", "--yes"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert json.loads(env_file.read_text())["api_key"] == ""
    assert _global(home)["api_key"] == "global-key"
    assert _global(home)["team_id"] == GLOBAL_TEAM


def test_symlinked_directory_context_is_ignored(repo: Path, home: Path) -> None:
    (repo / ".prime").mkdir()
    (repo / ".prime" / "context.json").symlink_to(home / ".prime" / "config.json")

    assert find_local_context_file() is None
    assert Config().local_context_file is None


def test_symlinked_prime_directory_is_ignored(repo: Path, tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    _pin(elsewhere, {"team_id": EDISON})
    (repo / ".prime").symlink_to(elsewhere / ".prime")

    assert find_local_context_file() is None


def test_switch_local_refuses_to_write_through_a_symlink(
    repo: Path, home: Path, teams_api: None
) -> None:
    global_before = _global(home)
    (repo / ".prime").symlink_to(home / ".prime")

    result = runner.invoke(app, ["switch", "edison", "--local"], env=TEST_ENV)

    assert result.exit_code == 1
    assert "symlink" in result.output
    assert _global(home) == global_before
    assert not (home / ".prime" / "context.json").exists()


@pytest.mark.parametrize("content", ['{"context": "missing"}', "{not json"])
def test_unpin_removes_a_broken_directory_context(repo: Path, content: str) -> None:
    path = repo / ".prime" / "context.json"
    path.parent.mkdir()
    path.write_text(content)

    blocked = runner.invoke(app, ["config", "view"], env=TEST_ENV)
    assert blocked.exit_code == 1
    assert "context.json" in blocked.output.replace("\n", "")

    result = runner.invoke(app, ["config", "unpin"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert not path.exists()
    assert runner.invoke(app, ["config", "view"], env=TEST_ENV).exit_code == 0


def test_other_commands_still_report_a_broken_directory_context(repo: Path) -> None:
    _pin(repo, {"context": "missing"})

    result = runner.invoke(app, ["whoami"], env=TEST_ENV)

    assert result.exit_code == 1
    assert "Unknown context 'missing'" in result.output


def test_malformed_pinned_saved_context_is_a_readable_error(repo: Path) -> None:
    _save_env("customer", frontend_url=None)
    _pin(repo, {"context": "customer"})

    with pytest.raises(ValueError, match="Invalid context 'customer'"):
        Config()

    result = runner.invoke(app, ["whoami"], env=TEST_ENV)
    assert result.exit_code == 1
    assert "Invalid context 'customer'" in result.output
    assert "Traceback" not in result.output
    assert runner.invoke(app, ["config", "unpin"], env=TEST_ENV).exit_code == 0


def test_persistent_environment_switch_cannot_overwrite_the_pinned_context(
    repo: Path, home: Path
) -> None:
    pinned = _save_env("customer", api_key="customer-key")
    _save_env("other", api_key="other-key")
    _pin(repo, {"context": "customer"})
    pinned_before = pinned.read_text()
    global_before = _global(home)

    with pytest.raises(ValueError, match="selects context 'customer'"):
        Config().load_environment("other")

    assert pinned.read_text() == pinned_before
    assert _global(home) == global_before
