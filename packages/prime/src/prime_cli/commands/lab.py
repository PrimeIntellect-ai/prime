"""Prime Lab workspace commands."""

from pathlib import Path

import typer

from ..lab_setup import run_lab_doctor, run_lab_setup, run_lab_sync
from ..utils import PlainTyper, get_console

app = PlainTyper(help="Lab platform commands", no_args_is_help=True)
console = get_console()


@app.command(
    add_help_option=False,
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
)
def setup(ctx: typer.Context) -> None:
    """Set up a Lab workspace."""

    code = run_lab_setup(list(ctx.args), console=console)
    if code != 0:
        raise typer.Exit(code)


@app.command(
    add_help_option=False,
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
)
def sync(ctx: typer.Context) -> None:
    """Refresh Lab skills and local agent guidance."""

    code = run_lab_sync(list(ctx.args), console=console)
    if code != 0:
        raise typer.Exit(code)


@app.command(
    add_help_option=False,
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
)
def doctor(ctx: typer.Context) -> None:
    """Check a Lab workspace."""

    code = run_lab_doctor(list(ctx.args), console=console)
    if code != 0:
        raise typer.Exit(code)


@app.command(
    add_help_option=True,
)
def mcp(workspace: Path = typer.Option(Path.cwd(), "--workspace")) -> None:
    """Run the Lab MCP server over stdio."""

    _ = workspace
    console.print("Prime Lab MCP support is not enabled yet.")
    raise typer.Exit(1)
