from importlib.metadata import version

import typer

from .commands.availability import app as availability_app
from .commands.config import app as config_app
from .commands.pods import app as pods_app

app = typer.Typer(name="prime", help="Prime Intellect CLI")

app.add_typer(availability_app, name="availability")
app.add_typer(config_app, name="config")
app.add_typer(pods_app, name="pods")


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"Prime CLI Version: {version('prime-cli')}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def callback(
    ctx: typer.Context,
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """Prime Intellect CLI"""
    if ctx.invoked_subcommand is None:
        ctx.get_help()


def run() -> None:
    """Entry point for the CLI"""
    app()
