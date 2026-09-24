from prime_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


def test_eval_run_help_flags_print_hosted_usage():
    for args in (["eval", "run", "-h"], ["eval", "run", "gsm8k", "--help"]):
        result = runner.invoke(app, args, env={"PRIME_DISABLE_VERSION_CHECK": "1"})
        assert result.exit_code == 0, result.output
        assert "Usage: prime eval run ENVIRONMENT --hosted" in result.output
        assert "--allow-tunnel-access" in result.output
        assert "--rollouts-per-example" in result.output


def test_eval_run_without_hosted_points_to_prime_rl():
    result = runner.invoke(
        app,
        ["eval", "run", "gsm8k", "-n", "2"],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )

    assert result.exit_code == 1
    assert "no longer runs evaluations locally" in result.output
    assert "--hosted" in result.output
    assert "uv run eval" in result.output


def test_eval_view_and_tui_are_removed():
    for command in ("view", "tui"):
        result = runner.invoke(
            app,
            ["eval", command],
            env={"PRIME_DISABLE_VERSION_CHECK": "1"},
        )
        assert result.exit_code != 0
