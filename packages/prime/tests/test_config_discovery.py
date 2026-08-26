"""Project-local config discovery: PRIME_CONFIG_DIR > nearest .prime/ > ~/.prime."""

import json
import os
import stat
from pathlib import Path

import pytest
from prime_cli.core.config import Config, find_local_config_dir
from prime_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()

TEST_ENV = {"COLUMNS": "200", "LINES": "50", "PRIME_DISABLE_VERSION_CHECK": "1"}


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake home *inside* tmp_path so there is room above it for tests."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.delenv("PRIME_API_KEY", raising=False)
    monkeypatch.delenv("PRIME_BASE_URL", raising=False)
    monkeypatch.delenv("PRIME_API_BASE_URL", raising=False)
    return home


@pytest.fixture
def project(home: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A project dir under home; the working directory is a subdirectory of it."""
    project = home / "code" / "prime"
    workdir = project / "envs" / "my-env"
    workdir.mkdir(parents=True)
    monkeypatch.chdir(workdir)
    return project


def write_config(directory: Path, **values: object) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.json"
    path.write_text(json.dumps(values))
    return path


def test_falls_back_to_global_without_local_config(home: Path, project: Path) -> None:
    write_config(home / ".prime", api_key="global-key")

    config = Config()

    assert config.config_source == "global"
    assert config.config_dir == home / ".prime"
    assert config.is_global
    assert config.api_key == "global-key"


def test_discovers_local_config_in_ancestor_of_cwd(home: Path, project: Path) -> None:
    write_config(home / ".prime", api_key="global-key")
    write_config(project / ".prime", api_key="local-key")

    config = Config()

    assert config.config_source == "local"
    assert config.config_dir == project / ".prime"
    assert not config.is_global
    assert config.api_key == "local-key"


def test_nearest_local_config_wins(home: Path, project: Path) -> None:
    write_config(project / ".prime", api_key="outer-key")
    write_config(project / "envs" / ".prime", api_key="inner-key")

    assert Config().api_key == "inner-key"


def test_local_config_replaces_global_rather_than_merging(home: Path, project: Path) -> None:
    """A planted local file must never combine its base_url with the global API key."""
    write_config(home / ".prime", api_key="global-key")
    write_config(project / ".prime", base_url="https://local.example")

    config = Config()

    assert config.base_url == "https://local.example"
    assert config.api_key == ""


def test_env_vars_still_override_local_config(
    home: Path, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_config(project / ".prime", api_key="local-key")
    monkeypatch.setenv("PRIME_API_KEY", "env-key")

    assert Config().api_key == "env-key"


def test_prime_config_dir_env_var_beats_local_discovery(
    home: Path, project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_config(project / ".prime", api_key="local-key")
    explicit = tmp_path / "elsewhere"
    write_config(explicit, api_key="explicit-key")
    monkeypatch.setenv("PRIME_CONFIG_DIR", str(explicit))

    config = Config()

    assert config.config_source == "env"
    assert config.config_dir == explicit
    assert config.api_key == "explicit-key"


def test_discovery_stops_at_home(home: Path, project: Path, tmp_path: Path) -> None:
    """A .prime/ above the home directory is not the user's project."""
    write_config(tmp_path / ".prime", api_key="above-home-key")

    config = Config()

    assert config.config_source == "global"
    assert config.config_dir == home / ".prime"
    assert config.api_key == ""


def test_home_itself_is_global_not_local(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_config(home / ".prime", api_key="global-key")
    monkeypatch.chdir(home)

    config = Config()

    assert config.config_source == "global"
    assert config.api_key == "global-key"


def test_ignores_local_config_owned_by_another_user(
    home: Path, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_config(project / ".prime", api_key="planted-key")
    real_uid = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: real_uid + 1)

    assert find_local_config_dir() is None
    assert Config().config_source == "global"


def test_writes_go_to_the_discovered_local_file(home: Path, project: Path) -> None:
    global_file = write_config(home / ".prime", api_key="global-key")
    local_file = write_config(project / ".prime", api_key="local-key")

    Config().set_api_key("rotated-key")

    assert json.loads(local_file.read_text())["api_key"] == "rotated-key"
    assert json.loads(global_file.read_text())["api_key"] == "global-key"
    assert stat.S_IMODE(local_file.stat().st_mode) == 0o600


def test_config_local_creates_file_that_later_calls_discover(
    home: Path, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_config(home / ".prime", api_key="global-key")
    monkeypatch.chdir(project)

    created = Config.local()
    created.set_api_key("local-key")

    local_file = project / ".prime" / "config.json"
    assert created.config_file == local_file
    assert stat.S_IMODE(local_file.stat().st_mode) == 0o600

    monkeypatch.chdir(project / "envs" / "my-env")
    assert Config().api_key == "local-key"
    assert json.loads((home / ".prime" / "config.json").read_text())["api_key"] == "global-key"


def test_global_config_file_is_created_private(home: Path) -> None:
    config = Config()

    assert config.config_file == home / ".prime" / "config.json"
    assert stat.S_IMODE(config.config_file.stat().st_mode) == 0o600


class TestCli:
    def test_view_shows_active_config_file(self, home: Path, project: Path) -> None:
        write_config(project / ".prime", api_key="local-key")

        result = runner.invoke(app, ["config", "view"], env=TEST_ENV)

        assert result.exit_code == 0, result.output
        assert "Config File" in result.output
        assert "project-local" in result.output

    def test_view_labels_global_config(self, home: Path, project: Path) -> None:
        result = runner.invoke(app, ["config", "view"], env=TEST_ENV)

        assert result.exit_code == 0, result.output
        assert "(global)" in result.output

    def test_set_api_key_local_creates_project_config(
        self, home: Path, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        global_file = write_config(home / ".prime", api_key="global-key")
        monkeypatch.chdir(project)
        (project / ".git").mkdir()
        monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

        def no_network(*args: object, **kwargs: object) -> None:
            raise AssertionError("must not call the API in this test")

        monkeypatch.setattr("prime_cli.commands.config.APIClient", no_network)

        result = runner.invoke(
            app, ["config", "set-api-key", "--local", "local-key-1234567"], env=TEST_ENV
        )

        assert result.exit_code == 0, result.output
        local_file = project / ".prime" / "config.json"
        assert json.loads(local_file.read_text())["api_key"] == "local-key-1234567"
        assert json.loads(global_file.read_text())["api_key"] == "global-key"
        assert "project-local config" in result.output
        assert ".gitignore" in result.output  # .git exists but .prime/ is not ignored

    def test_set_api_key_local_is_quiet_when_gitignored(
        self, home: Path, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(project)
        (project / ".git").mkdir()
        (project / ".gitignore").write_text("outputs/\n/.prime/\n")
        monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

        def no_network(*args: object, **kwargs: object) -> None:
            raise AssertionError("must not call the API in this test")

        monkeypatch.setattr("prime_cli.commands.config.APIClient", no_network)

        result = runner.invoke(
            app, ["config", "set-api-key", "--local", "local-key-1234567"], env=TEST_ENV
        )

        assert result.exit_code == 0, result.output
        assert ".gitignore" not in result.output
