import json
import subprocess

import click
import pytest
from prime_cli.main import app
from prime_cli.verifiers_bridge import (
    _append_eval_options,
    _sanitize_help_text,
    exec_eval_process,
)
from prime_cli.verifiers_plugin import PrimeVerifiersPlugin
from typer.testing import CliRunner

runner = CliRunner()


class ExecCalled(Exception):
    pass


def test_eval_run_help_flags_use_backend_help(monkeypatch):
    def fake_help() -> None:
        print("BACKEND_HELP")

    monkeypatch.setattr("prime_cli.commands.evals.print_eval_run_help", fake_help)
    for flag in ("-h", "--help"):
        result = runner.invoke(app, ["eval", "run", flag], env={"PRIME_DISABLE_VERSION_CHECK": "1"})
        assert result.exit_code == 0, result.output
        assert "BACKEND_HELP" in result.output


def test_lab_setup_help_flags_use_prime_owned_help():
    for flag in ("-h", "--help"):
        result = runner.invoke(
            app,
            ["lab", "setup", flag],
            env={"PRIME_DISABLE_VERSION_CHECK": "1"},
        )
        assert result.exit_code == 0, result.output
        assert "Set up a Lab workspace." in result.output
        assert "--skip-install" in result.output
        assert "--prime-rl" not in result.output


def test_sanitize_help_removes_vf_eval_aliases():
    raw = (
        "usage: python -m verifiers.cli.commands.eval [-h] env_id_or_config\n"
        "Run vf-eval with verifiers.cli.commands.eval\n"
    )
    help_text = _sanitize_help_text(raw, "verifiers.cli.commands.eval", "prime eval run")

    assert "Usage: prime eval run [-h] environment" in help_text
    assert "verifiers.cli.commands.eval" not in help_text
    assert "vf-eval" not in help_text
    assert "env_id_or_config" not in help_text


def test_sanitize_help_rewrites_v1_console_script():
    raw = "usage: uv run eval [<taskset-id>]\nusage: main.py [-h] [@ FILE] [OPTIONS]\n"

    help_text = _sanitize_help_text(raw, "verifiers.v1.cli.eval.main", "prime eval run")

    assert help_text.count("Usage: prime eval run") == 2
    assert "uv run eval" not in help_text


def test_v1_module_command_calls_console_entrypoint(monkeypatch):
    monkeypatch.setattr("prime_cli.verifiers_plugin.resolve_workspace_python", lambda: "python")
    plugin = PrimeVerifiersPlugin()

    command = plugin.build_module_command(plugin.eval_module, ["--help"])

    assert command[:2] == ["python", "-c"]
    assert "sys.argv[0] = 'eval'" in command[2]
    assert command[-1] == "--help"


def test_append_eval_options_mentions_tunnel_access():
    help_text = _append_eval_options("Usage: prime eval run [-h] environment\n")

    assert "--allow-tunnel-access" in help_text


def test_exec_eval_process_resolves_and_reuses_run_id(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[-1] == "--protocol-version":
            payload = {
                "protocol_version": 1,
                "trace_schema_version": 1,
                "operations": ["run", "resolve"],
            }
        else:
            payload = {
                "operation": "resolve",
                "protocol_version": 1,
                "trace_schema_version": 1,
                "run_id": "resolved-run-id",
                "output_dir": str(tmp_path / "outputs" / "resolved-run-id"),
                "resume": False,
                "config": {"model": "test-model"},
            }
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    def fake_exec(executable, command, env):
        calls.append(("exec", executable, command, env))
        raise ExecCalled

    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.resolve_workspace_python", lambda _cwd: "python"
    )
    monkeypatch.setattr("prime_cli.verifiers_bridge.subprocess.run", fake_run)
    monkeypatch.setattr("prime_cli.verifiers_bridge.os.execvpe", fake_exec)

    with pytest.raises(ExecCalled):
        exec_eval_process(["gsm8k-v1", "--dry-run"], plain=True)

    assert calls[0][0] == ["python", "-m", "verifiers.v1.cli.eval.main", "--protocol-version"]
    assert calls[1][0] == [
        "python",
        "-m",
        "verifiers.v1.cli.eval.main",
        "resolve",
        "--format",
        "json",
        "gsm8k-v1",
        "--dry-run",
    ]
    assert calls[2][2] == [
        "python",
        "-m",
        "verifiers.v1.cli.eval.main",
        "run",
        "gsm8k-v1",
        "--dry-run",
        "--uuid",
        "resolved-run-id",
    ]
    assert calls[2][3]["NO_COLOR"] == "1"
    assert calls[2][3]["PYDANTIC_CONFIG_PLAIN"] == "1"


def test_exec_eval_process_does_not_resolve_help(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    calls = []
    protocol = {
        "protocol_version": 1,
        "trace_schema_version": 1,
        "operations": ["run", "resolve"],
    }

    def fake_run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps(protocol), "")

    def fake_exec(_executable, command, _env):
        calls.append(command)
        raise ExecCalled

    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.resolve_workspace_python", lambda _cwd: "python"
    )
    monkeypatch.setattr("prime_cli.verifiers_bridge.subprocess.run", fake_run)
    monkeypatch.setattr("prime_cli.verifiers_bridge.os.execvpe", fake_exec)

    with pytest.raises(ExecCalled):
        exec_eval_process(["--help"])

    assert len(calls) == 2
    assert calls[-1][-2:] == ["run", "--help"]


def test_exec_eval_process_rejects_invalid_resolve_response(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    responses = iter(
        [
            {
                "protocol_version": 1,
                "trace_schema_version": 1,
                "operations": ["run", "resolve"],
            },
            {
                "operation": "resolve",
                "protocol_version": 1,
                "trace_schema_version": 2,
                "run_id": "run-id",
                "output_dir": "outputs/run-id",
                "resume": False,
                "config": {},
            },
        ]
    )

    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.resolve_workspace_python", lambda _cwd: "python"
    )
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.subprocess.run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, json.dumps(next(responses)), ""
        ),
    )

    with pytest.raises(click.ClickException, match="Unsupported Verifiers resolve response"):
        exec_eval_process(["gsm8k-v1"])


def test_eval_view_uses_prime_viewer(monkeypatch):
    calls = {}

    def fake_run_eval_view(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr("prime_lab_app.run_eval_view", fake_run_eval_view)

    result = runner.invoke(
        app,
        ["eval", "view", "--limit", "25", "--env-dir", "envs", "--outputs-dir", "outs"],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )

    assert result.exit_code == 0, result.output
    assert calls["limit"] == 25
    assert calls["env_dir"] == "envs"
    assert calls["outputs_dir"] == "outs"


def test_eval_view_rejects_non_positive_limit():
    result = runner.invoke(
        app,
        ["eval", "view", "--limit", "0"],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )

    assert result.exit_code == 1
    assert "--limit must be at least 1" in result.output


def test_eval_tui_points_to_eval_view():
    result = runner.invoke(
        app,
        ["eval", "tui", "--limit", "0", "--env-dir", "envs", "--outputs-dir", "outs"],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )

    assert result.exit_code == 1
    assert "Deprecated" in result.output
    assert "prime eval view" in result.output
