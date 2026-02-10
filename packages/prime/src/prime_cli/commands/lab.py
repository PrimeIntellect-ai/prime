"""Lab commands for verifiers development."""

import subprocess

import typer
from rich.console import Console

from ..verifiers_plugin import load_verifiers_prime_plugin

app = typer.Typer(help="Lab commands for verifiers development", no_args_is_help=True)
console = Console()


@app.command(
    add_help_option=False,
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
)
def setup(ctx: typer.Context) -> None:
    """Set up a verifiers training workspace."""
    plugin = load_verifiers_prime_plugin(console=console)
    command = plugin.build_module_command(plugin.setup_module, list(ctx.args))
    result = subprocess.run(command)
    if result.returncode != 0:
        raise typer.Exit(result.returncode)
