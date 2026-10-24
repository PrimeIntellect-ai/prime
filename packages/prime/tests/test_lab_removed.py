"""`prime lab` was removed.

The CLI keeps the old entry points so scripts do not break with a usage error,
but they explain the removal instead of writing a workspace.
"""

from pathlib import Path

import pytest
from prime_cli.main import app
from prime_cli.utils import strip_ansi
from typer.testing import CliRunner

runner = CliRunner()


@pytest.mark.parametrize(
    "args",
    [
        ["setup"],
        ["setup", "--agent", "codex", "--no-interactive"],
        ["sync", "--agent", "claude"],
        ["doctor", "--fix"],
        ["register-github"],
    ],
)
def test_lab_commands_explain_the_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["lab", *args])

    assert result.exit_code == 1
    assert "`prime lab` was removed" in strip_ansi(result.output)
    assert list(tmp_path.iterdir()) == []


def test_lab_hygiene_passes_so_existing_workflows_keep_working(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["lab", "hygiene"])

    assert result.exit_code == 0
    output = strip_ansi(result.output)
    assert "`prime lab` was removed" in output
    assert "prime-lab-hygiene.yml" in output
    assert list(tmp_path.iterdir()) == []


def test_lab_group_is_hidden_from_help() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert " lab " not in strip_ansi(result.output)
