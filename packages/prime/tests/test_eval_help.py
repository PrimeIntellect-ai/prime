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


def test_lab_setup_help_flag_uses_backend_help(monkeypatch):
    def fake_help() -> None:
        print("BACKEND_HELP")

    monkeypatch.setattr("prime_cli.commands.lab.print_lab_setup_help", fake_help)
    result = runner.invoke(app, ["lab", "setup", "-h"], env={"PRIME_DISABLE_VERSION_CHECK": "1"})
    assert result.exit_code == 0, result.output
    assert "BACKEND_HELP" in result.output


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
