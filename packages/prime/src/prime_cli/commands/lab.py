"""Lab commands for verifiers development."""

import subprocess
from pathlib import Path

import typer

from ..utils import PlainTyper, get_console
from ..verifiers_bridge import is_help_request, print_lab_setup_help
from ..verifiers_plugin import load_verifiers_prime_plugin

app = PlainTyper(help="Lab commands for verifiers development", no_args_is_help=True)
console = get_console()


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


@app.command("view")
def view(
    limit: int = typer.Option(100, "--limit", "-n", help="Max rows to load per section"),
    env_dir: str = typer.Option(
        "./environments",
        "--env-dir",
        help="Local environments directory for discovering eval outputs",
    ),
    outputs_dir: str = typer.Option(
        "./outputs",
        "--outputs-dir",
        help="Local outputs directory for discovering eval outputs",
    ),
) -> None:
    """Launch the interactive Lab viewer."""
    if limit < 1:
        console.print("[red]Error:[/red] --limit must be at least 1")
        raise typer.Exit(1)

    from prime_lab_view import run_lab_view

    run_lab_view(
        limit=limit,
        env_dir=env_dir,
        outputs_dir=outputs_dir,
        workspace=Path.cwd(),
    )
