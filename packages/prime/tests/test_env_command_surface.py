from prime_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


def test_env_eval_command_is_removed() -> None:
    """`prime env eval` should not be registered anymore."""
    result = runner.invoke(app, ["env", "eval"])

    assert result.exit_code != 0
    assert "No such command 'eval'" in result.output
