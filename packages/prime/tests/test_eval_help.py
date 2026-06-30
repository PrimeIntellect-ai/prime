from prime_cli.main import app
from prime_cli.verifiers_bridge import _append_eval_options, _sanitize_help_text
from typer.testing import CliRunner

runner = CliRunner()


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


def test_append_eval_options_mentions_tunnel_access():
    help_text = _append_eval_options("Usage: prime eval run [-h] environment\n")

    assert "--allow-tunnel-access" in help_text
    assert "--project TEXT" in help_text
    assert "--no-project" in help_text


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
