from typing import Optional

import typer
from rich.console import Console

from . import __version__
from .commands.availability import app as availability_app
from .commands.config import app as config_app
from .commands.disks import app as disks_app
from .commands.env import app as env_app
from .commands.evals import app as evals_app
from .commands.inference import app as inference_app
from .commands.login import app as login_app
from .commands.pods import app as pods_app
from .commands.sandbox import app as sandbox_app
from .commands.teams import app as teams_app
from .commands.whoami import app as whoami_app
from .core import Config
from .utils.version_check import check_for_update

app = typer.Typer(
    name="prime",
    help=f"Prime Intellect CLI (v{__version__})",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)

app.add_typer(availability_app, name="availability")
app.add_typer(config_app, name="config")
app.add_typer(disks_app, name="disks")
app.add_typer(pods_app, name="pods")
app.add_typer(sandbox_app, name="sandbox")
app.add_typer(login_app, name="login")
app.add_typer(env_app, name="env")
app.add_typer(inference_app, name="inference")
app.add_typer(whoami_app, name="whoami")
app.add_typer(teams_app, name="teams")
app.add_typer(evals_app, name="eval")


@app.callback(invoke_without_command=True)
def callback(
    ctx: typer.Context,
    version_flag: bool = typer.Option(False, "--version", "-v", help="Show version and exit"),
    context: Optional[str] = typer.Option(
        None,
        "--context",
        "-c",
        help="Use a specific config context/environment for this command",
    ),
) -> None:
    """Prime Intellect CLI"""
    if version_flag:
        typer.echo(f"Prime CLI version: {__version__}")
        raise typer.Exit()

    if context:
        import os

        config = Config()
        # Check if the context exists
        if context.lower() != "production" and context not in config.list_environments():
            typer.echo(f"Error: Unknown context '{context}'", err=True)
            typer.echo("Available contexts:", err=True)
            for env_name in config.list_environments():
                typer.echo(f"  - {env_name}", err=True)
            raise typer.Exit(1)

        # Set environment variable so Config instances in subcommands pick it up
        os.environ["PRIME_CONTEXT"] = context

    # Check for updates (only when a subcommand is being executed)
    if ctx.invoked_subcommand is not None:
        update_available, latest = check_for_update()
        if update_available and latest:
            console = Console(stderr=True)
            console.print(
                f"[yellow]A new version of prime is available: {latest} "
                f"(installed: {__version__})[/yellow]"
            )
            console.print(
                "[dim]Run: uv pip install --upgrade prime  or  uv tool upgrade prime[/dim]\n"
            )


def run() -> None:
    """Entry point for the CLI"""
    try:
        app()
    except typer.Abort:
        typer.echo("\nOperation cancelled")
        raise typer.Exit(0)
