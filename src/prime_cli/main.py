from importlib.metadata import version

import typer

from .commands.availability import app as availability_app
from .commands.config import app as config_app
from .commands.pods import app as pods_app

__version__ = version("prime-cli")

app = typer.Typer(name="prime", help=f"Prime Intellect CLI (v{__version__})")

app.add_typer(availability_app, name="availability")
app.add_typer(config_app, name="config")
app.add_typer(pods_app, name="pods")


@app.callback(invoke_without_command=True)
def callback(
    ctx: typer.Context,
    version_flag: bool = typer.Option(
        False, "--version", "-v", help="Show version and exit"
    ),
) -> None:
    """Prime Intellect CLI"""
    if version_flag:
        typer.echo(f"Prime CLI version: {version('prime-cli')}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        ctx.get_help()


def run() -> None:
    """Entry point for the CLI"""
    app()
