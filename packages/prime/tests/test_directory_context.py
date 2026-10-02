"""Directory contexts (.prime/context.json) in the CLI and the SDK configs."""

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

import prime_evals.core.config as evals_config
import prime_sandboxes.core.config as sandboxes_config
import prime_traces.core.config as traces_config
import prime_tunnel.core.config as tunnel_config
import pytest
from prime_cli.core import Config
from prime_cli.main import app
from prime_traces.core.client import BaseTracesAPIClient
from typer.testing import CliRunner

runner = CliRunner()
TEST_ENV = {"COLUMNS": "200", "PRIME_DISABLE_VERSION_CHECK": "1"}
EDISON, ACME, GLOBAL = (
    "cedison000000000000000000",
    "cacme00000000000000000000",
    "cglobal000000000000000000",
)
SDK_MODULES = [sandboxes_config, evals_config, tunnel_config, traces_config]
CONFIGS = [Config, *(module.Config for module in SDK_MODULES)]


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    (home / "code" / "edison" / "src").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    for name in ("PRIME_CONTEXT", "PRIME_API_KEY", "PRIME_TEAM_ID", "PRIME_USER_ID"):
        monkeypatch.delenv(name, raising=False)
    for name in ("PRIME_API_BASE_URL", "PRIME_BASE_URL", "PRIME_DISABLE_CONTEXT_NOTICE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    config = Config(use_context=False)
    config.set_api_key("global-key")
    config.set_team(GLOBAL, team_name="Global Team", team_role="admin")
    config.set_user_id("global-user")
    _write(
        home / ".prime" / "environments" / "customer.json",
        {"api_key": "customer-key", "team_id": ACME},
    )
    monkeypatch.chdir(home / "code" / "edison" / "src")
    return home


@pytest.fixture
def repo(home: Path) -> Path:
    return home / "code" / "edison"


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> None:
    teams = [
        {"teamId": EDISON, "name": "Edison", "slug": "edison", "role": "member"},
        {"teamId": ACME, "name": "Acme", "slug": "acme", "role": "admin"},
    ]

    def get(self: Any, endpoint: str, params: Optional[dict] = None, **_: Any) -> dict:
        if endpoint == "/user/whoami":
            return {"data": {"id": "global-user", "name": "Global User", "scope": {}}}
        return {"data": teams if endpoint == "/user/teams" else []}

    monkeypatch.setattr("prime_cli.core.APIClient.get", get)


def _write(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data))
    return path


def _pin(directory: Path, data: Any) -> Path:
    return _write(directory / ".prime" / "context.json", data)


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _invoke(*args: str, **env: str) -> Any:
    return runner.invoke(app, list(args), env={**TEST_ENV, **env})


# The shared resolution spec: every Config below, and other pin readers such as
# prime-agent, should agree on these cases.
CASES = json.loads((Path(__file__).parent / "data" / "directory_context_cases.json").read_text())
assert CASES["global"] == {"api_key": "global-key", "team_id": GLOBAL, "team_name": "Global Team"}


@pytest.mark.parametrize("config_class", CONFIGS)
@pytest.mark.parametrize("case", CASES["cases"], ids=lambda case: case["name"])
def test_resolution(
    home: Path, monkeypatch: pytest.MonkeyPatch, config_class: Any, case: dict
) -> None:
    for name, saved in CASES["saved_contexts"].items():
        _write(home / ".prime" / "environments" / f"{name}.json", saved)
    for relative, data in case["pins"].items():
        directory = home / relative
        if data == "symlink-file":
            (directory / ".prime").mkdir()
            target = _pin(home / "elsewhere", {"team_id": EDISON})
            (directory / ".prime" / "context.json").symlink_to(target)
        elif data == "symlink-dir":
            (directory / ".prime").symlink_to(_pin(home / "elsewhere", {"team_id": EDISON}).parent)
        else:
            _pin(directory, data)
    for name, value in case["env"].items():
        monkeypatch.setenv(name, value)

    if case.get("error"):
        with pytest.raises(ValueError):
            config = config_class()
            (config.team_id, config.api_key)
        return
    config = config_class()

    assert (config.team_id, config.api_key) == (case["team_id"], case["api_key"])


@pytest.mark.parametrize("module", SDK_MODULES)
def test_sdk_empty_team_env_means_personal(
    home: Path, monkeypatch: pytest.MonkeyPatch, module: Any
) -> None:
    monkeypatch.setenv("PRIME_TEAM_ID", "")
    assert module.Config().team_id is None


@pytest.mark.parametrize("module", SDK_MODULES)
@pytest.mark.parametrize("content", ['{"context": "missing"}', "[]"])
def test_sdk_broken_pin_fails_only_values_read_from_it(
    repo: Path, monkeypatch: pytest.MonkeyPatch, module: Any, content: str
) -> None:
    _pin(repo, content)
    config = module.Config()
    with pytest.raises(ValueError, match="context"):
        config.team_id
    monkeypatch.setenv("PRIME_API_KEY", "env-key")
    monkeypatch.setenv("PRIME_TEAM_ID", "env-team")
    assert (config.api_key, config.team_id) == ("env-key", "env-team")


def test_traces_client_with_explicit_values_ignores_a_broken_pin(repo: Path) -> None:
    _pin(repo, {"context": "missing"})
    client = BaseTracesAPIClient(api_key="k", base_url="https://traces.example", team_id="t")
    assert (client.api_key, client.team_id) == ("k", "t")


@pytest.mark.parametrize("module", SDK_MODULES)
def test_sdk_logs_an_applied_pin_once(repo: Path, caplog: Any, module: Any) -> None:
    _pin(repo, {"team_id": EDISON})
    module._LOGGED_PINS.clear()
    with caplog.at_level(logging.INFO, logger=module.__name__):
        module.Config()
        module.Config()
    assert len([r for r in caplog.records if "context.json" in r.getMessage()]) == 1


@pytest.mark.parametrize(
    "content",
    [
        "[]",
        "{not json",
        '{"context": "../x"}',
        '{"team_id": 3}',
        '{"context": "missing"}',
        '{"context": "broken"}',
    ],
)
def test_cli_broken_pin_is_a_readable_error_that_unpin_fixes(
    repo: Path, home: Path, content: str
) -> None:
    _write(home / ".prime" / "environments" / "broken.json", {"frontend_url": None})
    pin = _pin(repo, content)

    with pytest.raises(ValueError, match="context"):
        Config()
    result = _invoke("whoami")
    assert result.exit_code == 1 and "Traceback" not in result.output
    assert "context.json" in result.output.replace("\n", "")

    assert _invoke("config", "unpin").exit_code == 0
    assert not pin.exists()


def test_cli_writes_follow_the_directory_context(repo: Path, home: Path) -> None:
    global_file = home / ".prime" / "config.json"
    customer = home / ".prime" / "environments" / "customer.json"
    dev = _write(home / ".prime" / "environments" / "dev.json", {"team_id": GLOBAL})
    Config(use_context=False).set_current_environment("dev")

    # Team pin: writes stay global and never pick up the pinned team.
    _pin(repo, {"team_id": EDISON})
    config = Config()
    config.set_api_key("rotated")
    config.update_current_environment_file()
    assert (_read(global_file)["api_key"], _read(global_file)["team_id"]) == ("rotated", GLOBAL)
    assert _read(dev)["team_id"] == GLOBAL

    # Context pin: credential writes go to that context's file only.
    _pin(repo, {"context": "customer"})
    before = global_file.read_text()
    config = Config()
    config.set_api_key("customer-2")
    config.set_user_id("customer-user")
    assert _read(customer)["api_key"] == "customer-2" and global_file.read_text() == before
    with pytest.raises(ValueError, match="selects 'customer'"):
        Config().load_environment("dev")
    assert _invoke("config", "delete", "customer").exit_code == 1
    assert _invoke("logout", "--yes").exit_code == 0
    assert _read(customer)["api_key"] == "" and _read(global_file)["api_key"] == "rotated"


@pytest.mark.parametrize(
    ("pin", "args", "cwd", "pin_after", "global_team"),
    [
        (
            None,
            ["switch", "edison", "--local"],
            "src",
            {"team_id": EDISON, "team_name": "Edison"},
            GLOBAL,
        ),
        ({"team_id": EDISON}, ["switch", "personal"], "src", {"team_id": None}, GLOBAL),
        ({"team_id": EDISON}, ["switch", "acme", "--global"], "", {"team_id": EDISON}, ACME),
        (
            {"team_id": EDISON},
            ["config", "set-team-id", ACME],
            "",
            {"team_id": ACME, "team_name": "Acme"},
            GLOBAL,
        ),
        (
            {"team_id": EDISON},
            ["config", "use", "customer", "--local"],
            "",
            {"context": "customer"},
            GLOBAL,
        ),
        (
            {"context": "customer"},
            ["config", "use", "production", "--global"],
            "",
            {"context": "customer"},
            None,  # production resets the global team to personal
        ),
    ],
)
def test_cli_selection_commands(
    repo: Path,
    home: Path,
    api: None,
    monkeypatch: pytest.MonkeyPatch,
    pin: Any,
    args: list,
    cwd: str,
    pin_after: dict,
    global_team: Optional[str],
) -> None:
    (repo / ".git").mkdir()  # --local pins the repository root
    if pin is not None:
        _pin(repo, pin)
    monkeypatch.chdir(repo / cwd)

    result = _invoke(*args)

    assert result.exit_code == 0, result.output
    assert _read(repo / ".prime" / "context.json") == pin_after
    assert _read(home / ".prime" / "config.json")["team_id"] == global_team
    assert not (repo / "src" / ".prime").exists()


@pytest.mark.parametrize(
    ("setup", "args", "message"),
    [
        (None, ["switch", "edison", "--local", "--global"], "either --local or --global"),
        ("home", ["switch", "edison", "--local"], "cannot pin your home directory"),
        ("symlink", ["switch", "edison", "--local"], "symlink"),
        (None, ["config", "use", "nope", "--local"], "Unknown environment: nope"),
    ],
)
def test_cli_selection_errors(
    repo: Path,
    home: Path,
    api: None,
    monkeypatch: pytest.MonkeyPatch,
    setup: Any,
    args: list,
    message: str,
) -> None:
    global_before = (home / ".prime" / "config.json").read_text()
    if setup == "home":
        monkeypatch.chdir(home)
    if setup == "symlink":
        (repo / ".prime").symlink_to(home / ".prime")
        monkeypatch.chdir(repo)

    result = _invoke(*args)

    assert result.exit_code == 1
    assert message in result.output.replace("\n", "")
    assert (home / ".prime" / "config.json").read_text() == global_before
    assert not (home / ".prime" / "context.json").exists()


def test_cli_shows_the_pin(repo: Path, api: None) -> None:
    pin = _pin(repo, {"team_id": EDISON, "team_name": "[green]Personal[/green] verified"})
    (repo / ".prime" / "lab.json").write_text("{}")

    view = _invoke("config", "view").output.replace("\n", "")
    assert "Directory Context" in view and str(pin) in view and "(directory context)" in view
    whoami = _invoke("whoami", PRIME_DISABLE_CONTEXT_NOTICE="1").output
    assert "[green]Personal[/green] verified" in whoami and "Pinned By" in whoami
    assert "still selects this directory" in _invoke("config", "reset", "--yes").output
    assert pin.exists()

    assert _invoke("config", "unpin").exit_code == 0
    assert not pin.exists() and (repo / ".prime" / "lab.json").exists()


@pytest.mark.parametrize(
    ("pin", "env", "shown"),
    [
        (
            {"team_id": EDISON, "team_name": "Edison"},
            {},
            f"Using team 'Edison' ({EDISON}), pinned by",
        ),
        ({"team_id": GLOBAL}, {}, None),
        ({"team_id": None}, {"PRIME_DISABLE_CONTEXT_NOTICE": "1"}, None),
    ],
)
def test_cli_notice_when_a_pin_changes_the_account(
    repo: Path, api: None, pin: dict, env: dict, shown: Any
) -> None:
    _pin(repo, pin)

    output = _invoke("switch", "acme", **env).output.replace("\n", "")

    assert (shown in output) if shown else ("pinned by" not in output)


def test_cli_without_a_pin_does_not_need_a_loadable_config(home: Path) -> None:
    (home / ".prime" / "config.json").write_text("{corrupt")
    assert _invoke("sandbox", "--help").exit_code == 0


def test_cli_switch_global_uses_the_global_account(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pin(repo, {"context": "customer"})
    used: list = []

    def get(self: Any, endpoint: str, params: Optional[dict] = None, **_: Any) -> dict:
        used.append(self.api_key)
        return {"data": [{"teamId": ACME, "name": "Acme", "slug": "acme"}]}

    monkeypatch.setattr("prime_cli.core.APIClient.get", get)

    assert _invoke("switch", "acme", "--global").exit_code == 0
    assert used == ["global-key"]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, check=True, text=True
    ).stdout


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_cli_local_pin_stays_out_of_git(repo: Path, api: None) -> None:
    _git(repo, "init", "-q")

    result = _invoke("switch", "edison", "--local")

    assert result.exit_code == 0, result.output
    assert "/.prime/context.json" in (repo / ".git" / "info" / "exclude").read_text().splitlines()
    assert _git(repo, "status", "--porcelain", "--untracked-files=all") == ""
    assert _invoke("switch", "acme").exit_code == 0  # no duplicate entry on later writes
    exclude = (repo / ".git" / "info" / "exclude").read_text()
    assert exclude.count("/.prime/context.json") == 1


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_cli_committed_pin_is_left_alone(repo: Path, api: None) -> None:
    _git(repo, "init", "-q")
    _pin(repo, {"team_id": EDISON})
    _git(repo, "add", ".prime/context.json")
    exclude = repo / ".git" / "info" / "exclude"
    exclude_before = exclude.read_text() if exclude.exists() else None

    switched = _invoke("switch", "acme")
    used = _invoke("config", "use", "customer", "--local")

    assert switched.exit_code == 0 and used.exit_code == 0, used.output
    assert (exclude.read_text() if exclude.exists() else None) == exclude_before
    assert "is committed" in used.output.replace("\n", "")
