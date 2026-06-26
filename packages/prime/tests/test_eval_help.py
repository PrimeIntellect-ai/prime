import pytest
from prime_cli.main import app
from prime_cli.verifiers_bridge import _sanitize_help_text, exec_eval_process
from typer.testing import CliRunner

runner = CliRunner()


class ExecCalled(Exception):
    pass


def test_eval_run_help_flags_are_forwarded(monkeypatch):
    captured = []
    monkeypatch.setattr(
        "prime_cli.commands.evals.exec_eval_process",
        lambda args, plain: captured.append((list(args), plain)),
    )

    for flag in ("-h", "--help"):
        result = runner.invoke(
            app,
            ["eval", "run", flag],
            env={"PRIME_DISABLE_VERSION_CHECK": "1"},
        )
        assert result.exit_code == 0, result.output

    assert captured == [(["-h"], False), (["--help"], False)]


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


def test_exec_eval_process_forwards_once(monkeypatch):
    captured = {}

    def fake_exec(executable, command, env):
        captured.update(executable=executable, command=command, env=env)
        raise ExecCalled

    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.resolve_workspace_python",
        lambda: "python",
    )
    monkeypatch.setattr("prime_cli.verifiers_bridge.os.execvpe", fake_exec)

    with pytest.raises(ExecCalled):
        exec_eval_process(["gsm8k-v1", "--dry-run"], plain=True)

    assert captured["executable"] == "python"
    assert captured["command"] == [
        "python",
        "-c",
        "from verifiers.v1.cli.eval.main import main; main()",
        "gsm8k-v1",
        "--dry-run",
    ]
    assert captured["env"]["NO_COLOR"] == "1"
    assert captured["env"]["PYDANTIC_CONFIG_PLAIN"] == "1"


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
