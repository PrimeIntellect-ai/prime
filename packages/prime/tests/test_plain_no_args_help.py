import pytest
from prime_cli.main import app
from prime_cli.utils.plain import HELP_NOTE
from typer.testing import CliRunner

runner = CliRunner()
ENV = {"PRIME_DISABLE_VERSION_CHECK": "1"}


def _invoke(args):
    return runner.invoke(app, args, env=ENV)


def test_plain_alone_shows_help_like_bare_invocation():
    bare = _invoke([])
    plain = _invoke(["--plain"])

    assert plain.exit_code == bare.exit_code
    assert "Usage: prime [OPTIONS] COMMAND [ARGS]..." in plain.output
    assert plain.output.startswith(f"Note: {HELP_NOTE[:40]}")


def test_plain_alone_matches_help_plain_text():
    plain = _invoke(["--plain"])
    help_plain = _invoke(["--plain", "--help"])

    assert plain.output.strip() == help_plain.output.strip()


@pytest.mark.parametrize("args", [["--plain"], ["--plain", "--help"]])
def test_plain_help_has_no_rich_decorations(args):
    result = _invoke(args)

    assert "─" not in result.output
    assert "╭" not in result.output
    assert "[" not in result.output.split("Usage:")[0]
