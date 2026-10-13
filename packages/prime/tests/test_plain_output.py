import io

from prime_cli.utils.plain import PrimeConsole
from rich.console import Console
from rich.table import Table
from rich.text import Text

ADVERSARIAL_VALUES = (
    "[bold]x[/bold]",
    "[link=https://example.test]x[/link]",
    "[red",
    "🚀",
    "x" * 300,
)


def test_plain_table_renders_adversarial_cells_literally(monkeypatch) -> None:
    stream = io.StringIO()
    console = PrimeConsole(file=stream, width=48, record=True)
    monkeypatch.setattr("prime_cli.utils.plain.is_plain_mode", lambda: True)

    table = Table(title="Values [dim]from the API[/dim]")
    table.add_column("Value")
    for value in ADVERSARIAL_VALUES:
        # Dynamic values must arrive as Text, not as presentation markup.
        table.add_row(Text(value))

    console.print(table)
    output = stream.getvalue()

    assert "MarkupError" not in output
    assert "[bold]x[/bold]" in output
    assert "[link=https://example.test]x[/link]" in output
    assert "[red" in output
    assert "🚀" in output
    assert "x" * 40 in output
    assert "[dim]from the API[/dim]" not in output
    assert "\\[bold]" not in output
    assert max(map(len, output.splitlines())) <= 48


def test_style_snapshot_keeps_intentional_table_colors() -> None:
    stream = io.StringIO()
    console = Console(
        file=stream,
        record=True,
        force_terminal=True,
        color_system="standard",
        width=60,
    )
    table = Table()
    table.add_column("Status")
    table.add_row(Text("Available", style="green"))
    table.add_row(Text("At Capacity", style="red"))

    console.print(table)

    assert console.export_text(styles=True) == (
        "┏━━━━━━━━━━━━━┓\n"
        "┃\x1b[1m \x1b[0m\x1b[1mStatus     \x1b[0m\x1b[1m \x1b[0m┃\n"
        "┡━━━━━━━━━━━━━┩\n"
        "│ \x1b[32mAvailable  \x1b[0m │\n"
        "│ \x1b[31mAt Capacity\x1b[0m │\n"
        "└─────────────┘\n"
    )
