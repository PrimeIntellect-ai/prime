import sys
from typing import Optional

import typer
from rich.console import Console

from . import __version__
from .commands.availability import app as availability_app
from .commands.config import app as config_app
from .commands.disks import app as disks_app
from .commands.env import app as env_app
from .commands.evals import app as evals_app
from .commands.feedback import app as feedback_app
from .commands.images import app as images_app
from .commands.inference import app as inference_app
from .commands.lab import app as lab_app
from .commands.login import app as login_app
from .commands.pods import app as pods_app
from .commands.registry import app as registry_app
from .commands.rl import app as rl_app
from .commands.sandbox import app as sandbox_app
from .commands.teams import app as teams_app
from .commands.tunnel import app as tunnel_app
from .commands.upgrade import app as upgrade_app
from .commands.whoami import app as whoami_app
from .core import Config
from .utils.version_check import check_for_update

app = typer.Typer(
    name="prime",
    help=f"Prime Intellect CLI (v{__version__})",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)

# Lab commands
app.add_typer(env_app, name="env", rich_help_panel="Lab")
app.add_typer(evals_app, name="eval", rich_help_panel="Lab")
app.add_typer(rl_app, name="rl", rich_help_panel="Lab")
app.add_typer(lab_app, name="lab", rich_help_panel="Lab")

# Compute commands
app.add_typer(availability_app, name="availability", rich_help_panel="Compute")
app.add_typer(disks_app, name="disks", rich_help_panel="Compute")
app.add_typer(pods_app, name="pods", rich_help_panel="Compute")
app.add_typer(sandbox_app, name="sandbox", rich_help_panel="Compute")
app.add_typer(images_app, name="images", rich_help_panel="Compute")
app.add_typer(registry_app, name="registry", rich_help_panel="Compute")
app.add_typer(tunnel_app, name="tunnel", rich_help_panel="Compute")
app.add_typer(inference_app, name="inference", rich_help_panel="Compute")

# Account commands
app.add_typer(login_app, name="login", rich_help_panel="Account")
app.add_typer(whoami_app, name="whoami", rich_help_panel="Account")
app.add_typer(config_app, name="config", rich_help_panel="Account")
app.add_typer(teams_app, name="teams", rich_help_panel="Account")
app.add_typer(feedback_app, name="feedback", rich_help_panel="Account")
app.add_typer(upgrade_app, name="upgrade", rich_help_panel="Account")


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
            console = Console(stderr=True, force_terminal=sys.stderr.isatty())
            console.print(
                f"[yellow]A new version of prime is available: {latest} "
                f"(installed: {__version__})[/yellow]"
            )
            console.print("[dim]Run: prime upgrade[/dim]")
            console.print("[dim]Set PRIME_DISABLE_VERSION_CHECK=1 to disable this check[/dim]\n")


def run() -> None:
    """Entry point for the CLI"""
    try:
        app()
    except typer.Abort:
        typer.echo("\nOperation cancelled")
        raise typer.Exit(0)
