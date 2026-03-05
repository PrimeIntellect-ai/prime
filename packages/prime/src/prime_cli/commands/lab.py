"""Lab commands for verifiers development."""

import subprocess

import typer
from rich.console import Console

from ..verifiers_bridge import is_help_request, print_lab_setup_help
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
    passthrough_args = list(ctx.args)
    if is_help_request("", passthrough_args):
        print_lab_setup_help()
        raise typer.Exit(0)

    plugin = load_verifiers_prime_plugin(console=console)
    command = plugin.build_module_command(plugin.setup_module, passthrough_args)
    result = subprocess.run(command)
    if result.returncode != 0:
        raise typer.Exit(result.returncode)
