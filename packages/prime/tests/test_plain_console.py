import sys

import pytest
from prime_cli.utils.plain import PrimeConsole
from rich.table import Table


@pytest.fixture(autouse=True)
def _plain_mode_on_dumb_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["prime", "--plain"])
    # Rich treats any non-empty FORCE_COLOR as a terminal and sizes a TERM=dumb
    # terminal at a fixed 80 columns.
    monkeypatch.setenv("TERM", "dumb")
    monkeypatch.setenv("FORCE_COLOR", "0")


def test_plain_table_strips_markup_and_keeps_each_row_on_one_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    url = "https://example.com/" + "a" * 120
    table = Table()
    table.add_column("CLUSTER", style="cyan")
    table.add_column("STATUS")
    table.add_column("URL")
    table.add_row("[cyan]\\[1][/cyan] cluster-a", "[green]online[/green]", url)

    PrimeConsole().print(table)

    assert [line.split() for line in capsys.readouterr().out.splitlines()] == [
        ["CLUSTER", "STATUS", "URL"],
        ["[1]", "cluster-a", "online", url],
    ]


def test_plain_table_rows_have_no_trailing_spaces(capsys: pytest.CaptureFixture[str]) -> None:
    table = Table()
    table.add_column("NAME")
    table.add_column("STATE")
    table.add_row("[1] cluster-a", "")
    table.add_row("  bash", "running")

    PrimeConsole().print(table)

    assert capsys.readouterr().out.splitlines() == [
        "NAME           STATE",
        "[1] cluster-a",
        "  bash         running",
    ]


def test_plain_long_line_is_not_rewrapped(capsys: pytest.CaptureFixture[str]) -> None:
    line = "Logs: https://example.com/" + "a" * 200

    PrimeConsole().print(line)

    assert capsys.readouterr().out == line + "\n"
