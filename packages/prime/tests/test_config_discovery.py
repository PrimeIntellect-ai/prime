"""Project-local config discovery: PRIME_CONFIG_DIR > nearest trusted .prime/ > ~/.prime."""

import json
import os
import stat
import subprocess
from pathlib import Path

import httpx
import pytest
from prime_cli.core import config as config_module
from prime_cli.core.config import (
    Config,
    find_local_config_dir,
    is_trusted_local_config,
    load_trusted_configs,
    trust_local_config,
)
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
    monkeypatch.setattr(config_module, "_warned_untrusted", set())
    for var in ("PRIME_API_KEY", "PRIME_BASE_URL", "PRIME_API_BASE_URL", "PRIME_CONTEXT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    return home


@pytest.fixture
def project(home: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A project dir under home; the working directory is a subdirectory of it."""
    project = home / "code" / "prime"
    workdir = project / "envs" / "my-env"
    workdir.mkdir(parents=True)
    monkeypatch.chdir(workdir)
    return project


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("must not call the API in this test")

    monkeypatch.setattr("prime_cli.commands.config.APIClient", fail)


def write_config(directory: Path, *, trusted: bool = True, **values: object) -> Path:
    """Write `<directory>/config.json`; local ones are trusted unless told otherwise."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.json"
    path.write_text(json.dumps(values))
    if trusted and directory != Path.home() / ".prime":
        trust_local_config(path)
    return path


def git_init(directory: Path) -> None:
    subprocess.run(["git", "init", "-q", str(directory)], check=True)


def test_falls_back_to_global_without_local_config(home: Path, project: Path) -> None:
    write_config(home / ".prime", api_key="global-key")

    config = Config()

    assert config.config_source == "global"
    assert config.config_dir == home / ".prime"
    assert config.is_global
    assert config.api_key == "global-key"


def test_discovers_trusted_local_config_in_ancestor_of_cwd(home: Path, project: Path) -> None:
    write_config(home / ".prime", api_key="global-key")
    write_config(project / ".prime", api_key="local-key")

    config = Config()

    assert config.config_source == "local"
    assert config.config_dir == project / ".prime"
    assert not config.is_global
    assert config.api_key == "local-key"
    assert config.untrusted_local_configs == []


def test_nearest_trusted_local_config_wins(home: Path, project: Path) -> None:
    write_config(project / ".prime", api_key="outer-key")
    write_config(project / "envs" / ".prime", api_key="inner-key")

    assert Config().api_key == "inner-key"


def test_untrusted_local_config_is_ignored_with_a_warning(
    home: Path, project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A .prime/config.json that came with a cloned repo is owned by the user but
    must not be honored until they explicitly trust it."""
    write_config(home / ".prime", api_key="global-key")
    planted = write_config(project / ".prime", trusted=False, base_url="https://evil.example")

    config = Config()

    assert config.config_source == "global"
    assert config.base_url == Config.DEFAULT_BASE_URL
    assert config.untrusted_local_configs == [planted]
    err = capsys.readouterr().err
    assert "Ignoring untrusted project config" in err
    assert f"prime config trust {project}" in err

    # Warned once per process, not once per Config() call.
    Config()
    assert "Ignoring untrusted" not in capsys.readouterr().err


def test_untrusted_nested_config_does_not_mask_trusted_outer_one(home: Path, project: Path) -> None:
    write_config(project / ".prime", api_key="outer-key")
    write_config(project / "envs" / ".prime", trusted=False, api_key="planted-key")

    config = Config()

    assert config.api_key == "outer-key"
    assert config.untrusted_local_configs == [project / "envs" / ".prime" / "config.json"]


def test_trust_is_bound_to_file_content(home: Path, project: Path) -> None:
    """If the file changes under the user (e.g. git pull), it needs trusting again."""
    local = write_config(project / ".prime", api_key="local-key")
    assert Config().api_key == "local-key"

    local.write_text(json.dumps({"api_key": "local-key", "base_url": "https://evil.example"}))

    assert Config().config_source == "global"
    trust_local_config(local)
    assert Config().base_url == "https://evil.example"


def test_local_config_replaces_global_rather_than_merging(home: Path, project: Path) -> None:
    """A local file must never combine its base_url with the global API key."""
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
    write_config(explicit, trusted=False, api_key="explicit-key")
    monkeypatch.setenv("PRIME_CONFIG_DIR", str(explicit))

    config = Config()

    assert config.config_source == "env"
    assert config.config_dir == explicit
    assert config.api_key == "explicit-key"
    assert config.untrusted_local_configs == []


def test_discovery_stops_at_home(home: Path, project: Path, tmp_path: Path) -> None:
    """A .prime/ above the home directory is not the user's project, even if trusted."""
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
    config = Config()
    assert config.config_source == "global"
    assert config.untrusted_local_configs == []  # not even a candidate


def test_writes_go_to_the_discovered_local_file_and_keep_it_trusted(
    home: Path, project: Path
) -> None:
    global_file = write_config(home / ".prime", api_key="global-key")
    local_file = write_config(project / ".prime", api_key="local-key")

    Config().set_api_key("rotated-key")

    assert json.loads(local_file.read_text())["api_key"] == "rotated-key"
    assert json.loads(global_file.read_text())["api_key"] == "global-key"
    assert stat.S_IMODE(local_file.stat().st_mode) == 0o600
    # The CLI's own write refreshed the trust digest.
    assert Config().api_key == "rotated-key"


def test_config_local_creates_trusted_file_that_later_calls_discover(
    home: Path, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_config(home / ".prime", api_key="global-key")
    monkeypatch.chdir(project)

    created = Config.local()
    created.set_api_key("local-key")

    local_file = project / ".prime" / "config.json"
    assert created.config_file == local_file
    assert stat.S_IMODE(local_file.stat().st_mode) == 0o600
    assert str(local_file) in load_trusted_configs()

    monkeypatch.chdir(project / "envs" / "my-env")
    assert Config().api_key == "local-key"
    assert json.loads((home / ".prime" / "config.json").read_text())["api_key"] == "global-key"


def test_config_local_without_create_writes_nothing_until_first_set(
    home: Path, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(project)

    config = Config.local(create=False)
    assert not (project / ".prime").exists()
    assert config.api_key == ""

    config.set_api_key("local-key")
    assert json.loads((project / ".prime" / "config.json").read_text())["api_key"] == "local-key"
    assert Config().config_source == "local"


def test_adopt_urls_copies_service_urls_but_not_credentials(home: Path, project: Path) -> None:
    write_config(
        home / ".prime",
        api_key="global-key",
        base_url="https://api.dev.example",
        frontend_url="https://app.dev.example",
    )

    config = Config.local(project, create=False)
    config.adopt_urls(Config())

    assert config.base_url == "https://api.dev.example"
    assert config.frontend_url == "https://app.dev.example"
    assert config.api_key == ""


def test_trust_requires_ownership_even_with_matching_digest(
    home: Path, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A trusted path replaced by someone else's file with the same bytes is not trusted."""
    write_config(project / ".prime", api_key="local-key")
    real_uid = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: real_uid + 1)

    assert not is_trusted_local_config(project / ".prime" / "config.json")


def test_writes_refuse_files_owned_by_another_user(
    home: Path, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = write_config(project / ".prime", api_key="local-key")
    before = local.read_text()
    config = Config(config_dir=project / ".prime")
    real_uid = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: real_uid + 1)

    with pytest.raises(PermissionError, match="not owned"):
        config.set_api_key("stolen")

    assert local.read_text() == before


def test_writes_refuse_symlinks_owned_by_another_user(
    home: Path, project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A planted symlink must not turn O_TRUNC into a write to wherever it points."""
    (project / ".prime").mkdir(parents=True)
    target = tmp_path / "victim.json"
    target.write_text(json.dumps({"api_key": "precious"}))
    (project / ".prime" / "config.json").symlink_to(target)
    config = Config(config_dir=project / ".prime")
    real_uid = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: real_uid + 1)

    with pytest.raises(PermissionError, match="symlink"):
        config.set_api_key("x")

    assert json.loads(target.read_text()) == {"api_key": "precious"}


def test_global_config_writes_follow_symlinks_the_user_owns(home: Path, tmp_path: Path) -> None:
    """Dotfile-style symlinked ~/.prime/config.json keeps working."""
    (home / ".prime").mkdir()
    target = tmp_path / "dotfiles" / "prime.json"
    target.parent.mkdir()
    target.write_text(json.dumps({"api_key": "old"}))
    link = home / ".prime" / "config.json"
    link.symlink_to(target)

    Config().set_api_key("new")

    assert link.is_symlink()
    assert json.loads(target.read_text())["api_key"] == "new"


def test_project_config_writes_never_follow_symlinks(
    home: Path, project: Path, tmp_path: Path
) -> None:
    """Even a user-owned symlink: a cloned repo's links are user-owned too."""
    (project / ".prime").mkdir(parents=True)
    target = tmp_path / "elsewhere.json"
    target.write_text(json.dumps({"api_key": "precious"}))
    (project / ".prime" / "config.json").symlink_to(target)

    with pytest.raises(PermissionError, match="symlink"):
        Config(config_dir=project / ".prime").set_api_key("x")

    assert json.loads(target.read_text()) == {"api_key": "precious"}


def test_global_config_writes_refuse_dangling_symlinks(home: Path, tmp_path: Path) -> None:
    (home / ".prime").mkdir()
    (home / ".prime" / "config.json").symlink_to(tmp_path / "nowhere.json")

    with pytest.raises(PermissionError, match="dangling"):
        Config(config_dir=home / ".prime", create=False).set_api_key("x")

    assert not (tmp_path / "nowhere.json").exists()


class TestLocalEnvironments:
    """Named environment files next to a local config are inside its trust boundary."""

    @staticmethod
    def write_env(project: Path, name: str, **values: object) -> Path:
        env_dir = project / ".prime" / "environments"
        env_dir.mkdir(parents=True, exist_ok=True)
        env_file = env_dir / f"{name}.json"
        env_file.write_text(json.dumps(values))
        return env_file

    def test_prime_context_ignores_untrusted_environment_file(
        self,
        home: Path,
        project: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A repo update can rewrite environments/<name>.json while config.json stays trusted."""
        write_config(project / ".prime", api_key="local-key", base_url="https://api.dev.example")
        self.write_env(project, "dev", base_url="https://evil.example")
        monkeypatch.setenv("PRIME_CONTEXT", "dev")

        config = Config()

        assert config.config_source == "local"
        assert config.base_url == "https://api.dev.example"
        assert config.current_environment == "production"
        assert "Ignoring untrusted environment file" in capsys.readouterr().err

    def test_use_refuses_untrusted_environment_file(self, home: Path, project: Path) -> None:
        write_config(project / ".prime", api_key="local-key")
        self.write_env(project, "dev", base_url="https://evil.example")

        result = runner.invoke(app, ["config", "use", "dev"], env=TEST_ENV)

        assert result.exit_code == 1, result.output
        assert "Error: Environment file" in result.output  # long path gets ellipsized
        assert Config().base_url == Config.DEFAULT_BASE_URL

    def test_trust_command_covers_environment_files(
        self, home: Path, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_config(project / ".prime", trusted=False, api_key="local-key")
        env_file = self.write_env(project, "dev", base_url="https://api.dev.example")

        result = runner.invoke(app, ["config", "trust", str(project)], env=TEST_ENV)
        assert result.exit_code == 0, result.output
        assert str(env_file) in load_trusted_configs()

        monkeypatch.setenv("PRIME_CONTEXT", "dev")
        assert Config().base_url == "https://api.dev.example"

        # Untrust drops the environment entries too.
        result = runner.invoke(app, ["config", "untrust", str(project)], env=TEST_ENV)
        assert result.exit_code == 0, result.output
        assert load_trusted_configs() == {}

    def test_environment_saved_by_the_cli_is_trusted(
        self, home: Path, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_config(project / ".prime", api_key="local-key")
        monkeypatch.setenv("PRIME_BASE_URL", "https://api.dev.example")
        assert runner.invoke(app, ["config", "save", "dev"], env=TEST_ENV).exit_code == 0
        monkeypatch.delenv("PRIME_BASE_URL")

        assert runner.invoke(app, ["config", "use", "dev"], env=TEST_ENV).exit_code == 0
        assert Config().base_url == "https://api.dev.example"

        # ...until someone rewrites it behind the CLI's back.
        env_file = project / ".prime" / "environments" / "dev.json"
        env_file.write_text(json.dumps({"base_url": "https://evil.example"}))
        result = runner.invoke(app, ["config", "use", "dev"], env=TEST_ENV)
        assert result.exit_code == 1, result.output
        assert "Error: Environment file" in result.output

    def test_login_local_ignores_planted_environment_file(
        self, home: Path, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Config.local() is 'explicit', which must gate environment files like 'local'."""
        self.write_env(project, "dev", base_url="https://evil.example")
        monkeypatch.chdir(project)
        monkeypatch.setenv("PRIME_CONTEXT", "dev")
        posted: list[str] = []

        def record(url: str, *args: object, **kwargs: object) -> None:
            posted.append(url)
            raise httpx.ConnectError("offline")

        monkeypatch.setattr("prime_cli.commands.login.httpx.post", record)

        result = runner.invoke(app, ["login", "--local", "--headless"], env=TEST_ENV)

        assert result.exit_code == 1
        assert posted and all("evil.example" not in url for url in posted), posted
        assert posted[0].startswith(Config.DEFAULT_BASE_URL)

    def test_set_traces_url_refuses_to_launder_untrusted_environment_file(
        self, home: Path, project: Path
    ) -> None:
        root = write_config(project / ".prime", api_key="local-key", current_environment="dev")
        env_file = self.write_env(project, "dev", base_url="https://evil.example")
        before_env, before_root = env_file.read_text(), root.read_text()

        result = runner.invoke(
            app, ["config", "set-traces-url", "https://traces.example"], env=TEST_ENV
        )

        assert result.exit_code != 0, result.output
        assert env_file.read_text() == before_env
        assert root.read_text() == before_root  # refused before any write, not half-applied
        assert str(env_file) not in load_trusted_configs()

    def test_save_refuses_symlinked_environment_file(
        self, home: Path, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`environments/dev.json -> ../../leak.json` shipped in a repo must not redirect `save`."""
        write_config(project / ".prime", api_key="local-key")
        env_dir = project / ".prime" / "environments"
        env_dir.mkdir()
        (env_dir / "dev.json").symlink_to(Path("..") / ".." / "leak.json")

        result = runner.invoke(app, ["config", "save", "dev"], env=TEST_ENV)

        assert result.exit_code == 1, result.output
        assert "symlink" in result.output
        assert not (project / "leak.json").exists()

    def test_trust_skips_symlinked_environment_files(self, home: Path, project: Path) -> None:
        write_config(project / ".prime", trusted=False, api_key="local-key")
        real = self.write_env(project, "real", base_url="https://api.dev.example")
        (project / ".prime" / "environments" / "linked.json").symlink_to(real)

        result = runner.invoke(app, ["config", "trust", str(project)], env=TEST_ENV)

        assert result.exit_code == 0, result.output
        trusted = load_trusted_configs()
        assert str(real) in trusted
        assert str(project / ".prime" / "environments" / "linked.json") not in trusted

    def test_logout_keeps_environment_file_trusted(
        self, home: Path, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_config(project / ".prime", api_key="local-key")
        assert runner.invoke(app, ["config", "save", "dev"], env=TEST_ENV).exit_code == 0
        assert runner.invoke(app, ["config", "use", "dev"], env=TEST_ENV).exit_code == 0

        result = runner.invoke(app, ["logout", "--yes"], env=TEST_ENV)
        assert result.exit_code == 0, result.output
        env_file = project / ".prime" / "environments" / "dev.json"
        assert json.loads(env_file.read_text())["api_key"] == ""

        result = runner.invoke(app, ["config", "use", "dev"], env=TEST_ENV)
        assert result.exit_code == 0, result.output

    def test_global_config_environments_need_no_trust(
        self, home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_dir = home / ".prime" / "environments"
        env_dir.mkdir(parents=True)
        (env_dir / "dev.json").write_text(json.dumps({"base_url": "https://api.dev.example"}))
        monkeypatch.setenv("PRIME_CONTEXT", "dev")

        assert Config().base_url == "https://api.dev.example"


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

    def test_view_labels_global_config_and_lists_ignored_files(
        self, home: Path, project: Path
    ) -> None:
        write_config(project / ".prime", trusted=False, api_key="planted-key")

        result = runner.invoke(app, ["config", "view"], env=TEST_ENV)

        assert result.exit_code == 0, result.output
        assert "(global)" in result.output
        assert "Ignored Config" in result.output

    def test_trust_and_untrust_commands(
        self, home: Path, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_config(home / ".prime", api_key="global-key")
        local = write_config(project / ".prime", trusted=False, api_key="local-key")
        assert Config().api_key == "global-key"

        result = runner.invoke(app, ["config", "trust", str(project)], env=TEST_ENV)
        assert result.exit_code == 0, result.output
        assert str(local) in load_trusted_configs()
        assert Config().api_key == "local-key"

        # Accepts the .prime dir or the file itself, and defaults to cwd.
        monkeypatch.chdir(project)
        result = runner.invoke(app, ["config", "untrust"], env=TEST_ENV)
        assert result.exit_code == 0, result.output
        assert Config().api_key == "global-key"
        result = runner.invoke(app, ["config", "trust", str(project / ".prime")], env=TEST_ENV)
        assert result.exit_code == 0, result.output
        assert Config().api_key == "local-key"

    def test_trust_rejects_missing_file(self, home: Path, project: Path) -> None:
        result = runner.invoke(app, ["config", "trust", str(project)], env=TEST_ENV)

        assert result.exit_code == 1
        assert "No config file" in result.output

    def test_set_api_key_local_creates_trusted_project_config(
        self, home: Path, project: Path, no_network: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        global_file = write_config(home / ".prime", api_key="global-key")
        monkeypatch.chdir(project)
        git_init(project)

        result = runner.invoke(
            app, ["config", "set-api-key", "--local", "local-key-1234567"], env=TEST_ENV
        )

        assert result.exit_code == 0, result.output
        local_file = project / ".prime" / "config.json"
        assert json.loads(local_file.read_text())["api_key"] == "local-key-1234567"
        assert json.loads(global_file.read_text())["api_key"] == "global-key"
        assert str(local_file) in load_trusted_configs()
        assert "project-local config" in result.output
        assert ".gitignore" in result.output  # a repo, and .prime/ is not ignored

    def test_set_api_key_local_inherits_urls_from_active_config(
        self, home: Path, project: Path, no_network: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A new local config keeps targeting the deployment already configured."""
        write_config(home / ".prime", api_key="global-key", base_url="https://api.dev.example")
        monkeypatch.chdir(project)

        result = runner.invoke(app, ["config", "set-api-key", "--local", "k"], env=TEST_ENV)

        assert result.exit_code == 0, result.output
        saved = json.loads((project / ".prime" / "config.json").read_text())
        assert saved["base_url"] == "https://api.dev.example"
        assert saved["api_key"] == "k"

    def test_gitignore_warning_from_subdirectory_of_repo(
        self, home: Path, project: Path, no_network: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The repo root may be several levels up from where --local is run."""
        git_init(project)
        workdir = project / "services" / "app"
        workdir.mkdir(parents=True)
        monkeypatch.chdir(workdir)

        result = runner.invoke(app, ["config", "set-api-key", "--local", "k"], env=TEST_ENV)

        assert result.exit_code == 0, result.output
        assert ".gitignore" in result.output
        assert str(project) in result.output

    def test_gitignore_warning_silent_when_root_gitignore_covers_subdirectory(
        self, home: Path, project: Path, no_network: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        git_init(project)
        (project / ".gitignore").write_text("outputs/\n.prime/\n")
        workdir = project / "services" / "app"
        workdir.mkdir(parents=True)
        monkeypatch.chdir(workdir)

        result = runner.invoke(app, ["config", "set-api-key", "--local", "k"], env=TEST_ENV)

        assert result.exit_code == 0, result.output
        assert ".gitignore" not in result.output

    def test_local_refuses_existing_untrusted_file(
        self, home: Path, project: Path, no_network: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--local must not adopt (and then trust) a planted config that is already there."""
        planted = write_config(
            project / ".prime", trusted=False, base_url="https://evil.example", api_key="theirs"
        )
        before = planted.read_text()
        monkeypatch.chdir(project)

        def refuse(*args: object, **kwargs: object) -> None:
            raise AssertionError("login must not contact the planted base_url")

        monkeypatch.setattr("prime_cli.commands.login.httpx.post", refuse)

        for args in (["config", "set-api-key", "--local", "k"], ["login", "--local", "--headless"]):
            result = runner.invoke(app, args, env=TEST_ENV)
            assert result.exit_code == 1, result.output
            assert "not trusted" in result.output

        assert planted.read_text() == before
        assert str(planted) not in load_trusted_configs()
        assert Config().config_source == "global"

    def test_local_refuses_existing_file_owned_by_another_user(
        self, home: Path, project: Path, no_network: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even a previously trusted path, if swapped for someone else's file."""
        planted = write_config(project / ".prime", api_key="theirs")
        before = planted.read_text()
        monkeypatch.chdir(project)
        real_uid = os.getuid()
        monkeypatch.setattr(os, "getuid", lambda: real_uid + 1)

        result = runner.invoke(app, ["config", "set-api-key", "--local", "k"], env=TEST_ENV)

        assert result.exit_code == 1, result.output
        assert "not owned by you" in result.output
        assert planted.read_text() == before

    def test_local_refuses_dangling_symlink_config(
        self, home: Path, project: Path, no_network: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cloned `.prime/config.json -> ../leak.json` must not redirect the credential write."""
        (project / ".prime").mkdir(parents=True)
        (project / ".prime" / "config.json").symlink_to(Path("..") / "leak.json")
        monkeypatch.chdir(project)

        result = runner.invoke(app, ["config", "set-api-key", "--local", "k"], env=TEST_ENV)

        assert result.exit_code == 1, result.output
        assert "symlink" in result.output
        assert not (project / "leak.json").exists()
        assert load_trusted_configs() == {}

    def test_local_updates_existing_trusted_file(
        self, home: Path, project: Path, no_network: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        local = write_config(project / ".prime", api_key="old", base_url="https://api.dev.example")
        monkeypatch.chdir(project)

        result = runner.invoke(app, ["config", "set-api-key", "--local", "new"], env=TEST_ENV)

        assert result.exit_code == 0, result.output
        saved = json.loads(local.read_text())
        assert saved["api_key"] == "new"
        assert saved["base_url"] == "https://api.dev.example"
        assert Config().api_key == "new"

    def test_local_from_home_directory_is_the_global_config(
        self, home: Path, no_network: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """~/.prime is never in the trust registry; --local there must not refuse it."""
        global_file = write_config(home / ".prime", api_key="old")
        monkeypatch.chdir(home)

        result = runner.invoke(app, ["config", "set-api-key", "--local", "new"], env=TEST_ENV)

        assert result.exit_code == 0, result.output
        assert json.loads(global_file.read_text())["api_key"] == "new"
        assert load_trusted_configs() == {}
        assert "project-local" not in result.output

    def test_local_from_symlinked_home_is_still_the_global_config(
        self, tmp_path: Path, home: Path, no_network: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Path.cwd() is the physical path while Path.home() may go through a
        symlink; the global check must not depend on which spelling it sees."""
        link = tmp_path / "home-link"
        link.symlink_to(home, target_is_directory=True)
        monkeypatch.setenv("HOME", str(link))
        monkeypatch.setattr(Path, "home", lambda: link)
        global_file = write_config(link / ".prime", api_key="old")
        monkeypatch.chdir(home)  # physical path

        result = runner.invoke(app, ["config", "set-api-key", "--local", "new"], env=TEST_ENV)

        assert result.exit_code == 0, result.output
        assert json.loads(global_file.read_text())["api_key"] == "new"
        assert load_trusted_configs() == {}
        assert "project-local" not in result.output
        assert Config.local(create=False).is_global

    def test_failed_login_local_leaves_no_file_behind(
        self, home: Path, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An aborted login must not create an empty local config that would
        shadow the working global one."""
        write_config(home / ".prime", api_key="global-key")
        monkeypatch.chdir(project)

        def refuse(*args: object, **kwargs: object) -> None:
            raise httpx.ConnectError("offline")

        monkeypatch.setattr("prime_cli.commands.login.httpx.post", refuse)

        result = runner.invoke(app, ["login", "--local", "--headless"], env=TEST_ENV)

        assert result.exit_code == 1
        assert not (project / ".prime").exists()
        assert Config().api_key == "global-key"
