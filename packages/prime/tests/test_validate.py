"""Tests for `prime eval validate` — the wrapper around verifiers' model-free `validate`."""

import pytest
import typer
from prime_cli.main import app
from prime_cli.verifiers_bridge import ResolvedEnvironment, run_validate_passthrough
from prime_cli.verifiers_plugin import PrimeVerifiersPlugin
from typer.testing import CliRunner

runner = CliRunner()


class DummyConfig:
    api_key = "test-api-key"
    inference_url = "https://api.pinference.ai/api/v1"
    team_id = "team-abc"


class DummyPlugin:
    eval_module = "verifiers.v1.cli.eval.main"
    init_module = "verifiers.v1.cli.init"
    validate_module = "verifiers.v1.cli.validate"
    gepa_module = "verifiers.scripts.gepa"

    def build_module_command(self, module: str, args: list[str]) -> list[str]:
        return [module, *args]


def _install_dummies(monkeypatch, captured: dict):
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.load_verifiers_prime_plugin", lambda: DummyPlugin()
    )
    monkeypatch.setattr("prime_cli.verifiers_bridge.Config", lambda: DummyConfig())

    def fake_run(command, env=None):
        captured.update(command=command, env=env)

    monkeypatch.setattr("prime_cli.verifiers_bridge._run_command", fake_run)


def test_validate_help_flags_use_backend_help(monkeypatch):
    def fake_help() -> None:
        print("BACKEND_HELP")

    monkeypatch.setattr("prime_cli.commands.evals.print_validate_help", fake_help)
    for flag in ("-h", "--help"):
        result = runner.invoke(
            app, ["eval", "validate", flag], env={"PRIME_DISABLE_VERSION_CHECK": "1"}
        )
        assert result.exit_code == 0, result.output
        assert "BACKEND_HELP" in result.output


def test_validate_missing_environment_shows_help():
    # no_args_is_help: bare `prime eval validate` prints help and exits 2
    result = runner.invoke(app, ["eval", "validate"], env={"PRIME_DISABLE_VERSION_CHECK": "1"})
    assert result.exit_code == 2, result.output
    assert "Usage: prime eval validate" in result.output


def test_validate_rejects_leading_flag_as_environment():
    # with no declared typer options, the first unknown flag is captured as the
    # positional `environment` and rejected with the "must be the first argument" guard
    result = runner.invoke(
        app,
        ["eval", "validate", "--num-tasks", "5"],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )
    assert result.exit_code == 2, result.output
    assert "must be the first argument" in result.output


def test_validate_requires_api_key(monkeypatch):
    class NoKeyConfig:
        api_key = ""
        inference_url = ""
        team_id = None

    monkeypatch.setattr("prime_cli.verifiers_bridge.Config", lambda: NoKeyConfig())
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.load_verifiers_prime_plugin", lambda: DummyPlugin()
    )

    with pytest.raises(typer.Exit) as exc_info:
        run_validate_passthrough(environment="gsm8k-v1", passthrough_args=[])

    assert exc_info.value.exit_code == 1


def test_validate_environment_id_resolves_and_passes_bare_name(monkeypatch, tmp_path):
    captured: dict = {}
    _install_dummies(monkeypatch, captured)
    prepared = {}

    monkeypatch.setattr(
        "prime_cli.verifiers_bridge._prepare_v1_environment",
        lambda env_name, env_dir: (
            prepared.update(env=env_name, env_dir=env_dir)
            or ResolvedEnvironment(original=env_name, env_name=env_name, install_mode="local")
        ),
    )

    run_validate_passthrough(
        environment="gsm8k-v1",
        passthrough_args=["--num-tasks", "5", "--env-dir-path", str(tmp_path / "envs")],
    )

    command = captured["command"]
    assert command[0] == "verifiers.v1.cli.validate"
    assert command[1] == "gsm8k-v1"  # bare env name, not the path
    assert "--num-tasks" in command and "5" in command
    assert "--env-dir-path" not in command  # stripped (prime-only)
    assert prepared == {"env": "gsm8k-v1", "env_dir": str(tmp_path / "envs")}
    assert captured["env"]["PRIME_API_KEY"] == "test-api-key"
    assert captured["env"]["PRIME_INFERENCE_URL"] == DummyConfig.inference_url
    assert captured["env"]["PRIME_TEAM_ID"] == "team-abc"


def test_validate_strips_at_prefix_for_config_target(monkeypatch, tmp_path):
    captured: dict = {}
    _install_dummies(monkeypatch, captured)
    config_path = tmp_path / "validate.toml"
    config_path.write_text('[taskset]\nid = "gsm8k-v1"\n', encoding="utf-8")

    run_validate_passthrough(
        environment=f"@{config_path}",
        passthrough_args=["--num-tasks", "5"],
    )

    command = captured["command"]
    assert command[0] == "verifiers.v1.cli.validate"
    # pydantic-config reads a root config file as two tokens: `@ path`
    assert command[1:3] == ["@", str(config_path)]
    assert "--num-tasks" in command


def test_validate_at_without_path_errors(monkeypatch):
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.load_verifiers_prime_plugin", lambda: DummyPlugin()
    )
    monkeypatch.setattr("prime_cli.verifiers_bridge.Config", lambda: DummyConfig())

    result = runner.invoke(app, ["eval", "validate", "@"], env={"PRIME_DISABLE_VERSION_CHECK": "1"})
    assert result.exit_code == 2, result.output
    assert "must be followed by a config path" in result.output


def test_validate_module_uses_console_entrypoint_shim(monkeypatch):
    monkeypatch.setattr("prime_cli.verifiers_plugin.resolve_workspace_python", lambda: "python")
    plugin = PrimeVerifiersPlugin()

    command = plugin.build_module_command(plugin.validate_module, ["--help"])

    assert command[:2] == ["python", "-c"]
    assert "sys.argv[0] = 'validate'" in command[2]
    assert "from verifiers.v1.cli.validate import main" in command[2]
    assert command[-1] == "--help"


def test_sanitize_help_rewrites_validate_console_script():
    from prime_cli.verifiers_bridge import _sanitize_help_text

    # validate's prog is `validate` (not `main.py` like eval); both usage lines carry it.
    raw = (
        "usage: validate [<taskset-id>] [--runtime.type subprocess] [options] [@ file.toml]\n"
        "       runs each task's `validate` hook\n"
        "usage: validate [-h] --taskset.id ID [OPTIONS]\n"
    )
    help_text = _sanitize_help_text(raw, "verifiers.v1.cli.validate", "prime eval validate")

    assert help_text.count("Usage: prime eval validate") == 2
    assert "verifiers.v1.cli.validate" not in help_text
    # the bare `usage: validate` lines are rewritten, not left behind
    assert help_text.lower().count("usage: validate ") == 0
