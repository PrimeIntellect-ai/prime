"""GEPA commands."""

import typer
from rich.console import Console
from typer.core import TyperGroup

from ..verifiers_bridge import print_gepa_run_help, run_gepa_passthrough

console = Console()


class DefaultGroup(TyperGroup):
    def __init__(self, *args, default_cmd_name: str = "run", **kwargs):
        super().__init__(*args, **kwargs)
        self.default_cmd_name = default_cmd_name

    def parse_args(self, ctx, args):
        if not args:
            return super().parse_args(ctx, args)
        if args[0] in ("--help", "-h"):
            return super().parse_args(ctx, args)
        if args[0] in self.commands:
            return super().parse_args(ctx, args)
        args = [self.default_cmd_name] + list(args)
        return super().parse_args(ctx, args)

    def format_usage(self, ctx, formatter):
        formatter.write_usage(
            ctx.command_path,
            "run ENV_OR_CONFIG [ARGS]... | COMMAND [ARGS]...",
        )


app = typer.Typer(
    cls=DefaultGroup,
    help="Run GEPA prompt optimization.",
    no_args_is_help=True,
)


@app.command(
    "run",
    no_args_is_help=True,
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
)
def run_gepa_cmd(
    ctx: typer.Context,
    environment_or_config: str = typer.Argument(
        ...,
        help="Environment name/slug or TOML config path",
    ),
    backend_help: bool = typer.Option(
        False,
        "--backend-help",
        help="Show backend vf-gepa help (all passthrough flags/options)",
    ),
) -> None:
    """Run optimization with local-first environment resolution."""
    passthrough_args = list(ctx.args)
    if backend_help:
        print_gepa_run_help()
        raise typer.Exit(0)

    if environment_or_config.startswith("-"):
        console.print("[red]Error:[/red] Environment/config must be the first argument.")
        console.print("[dim]Example: prime gepa run wordle --max-calls 100[/dim]")
        raise typer.Exit(2)

    run_gepa_passthrough(environment_or_config, passthrough_args)
