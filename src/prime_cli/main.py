from importlib.metadata import version

import typer

from .commands.availability import app as availability_app
from .commands.config import app as config_app
from .commands.env import app as env_app
from .commands.login import app as login_app
from .commands.pods import app as pods_app
from .commands.sandbox import app as sandbox_app

__version__ = version("prime")

app = typer.Typer(name="prime", help=f"Prime Intellect CLI (v{__version__})")

app.add_typer(availability_app, name="availability")
app.add_typer(config_app, name="config")
app.add_typer(pods_app, name="pods")
app.add_typer(sandbox_app, name="sandbox")
app.add_typer(login_app, name="login")
app.add_typer(env_app, name="env")


@app.callback(invoke_without_command=True)
def callback(
    ctx: typer.Context,
    version_flag: bool = typer.Option(False, "--version", "-v", help="Show version and exit"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    """Prime Intellect CLI"""
    if version_flag:
        typer.echo(f"Prime CLI version: {__version__}")
        raise typer.Exit()

    # Store debug flag in context for subcommands to access
    ctx.obj = {"debug": debug}

    if ctx.invoked_subcommand is None:
        ctx.get_help()


def run() -> None:
    """Entry point for the CLI"""
    app()
