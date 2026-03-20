"""GEPA commands."""

import typer

from ..utils import PlainAwareTyperGroup, PlainTyper, get_console
from ..verifiers_bridge import is_help_request, print_gepa_run_help, run_gepa_passthrough

console = get_console()


class DefaultGroup(PlainAwareTyperGroup):
    def __init__(self, *args, default_cmd_name: str = "run", **kwargs):
        super().__init__(*args, **kwargs)
        self.default_cmd_name = default_cmd_name

    def parse_args(self, ctx, args):
        if not args:
            return super().parse_args(ctx, args)

        decision_args = [arg for arg in args if arg != "--plain"]
        if not decision_args:
            return super().parse_args(ctx, args)

        if decision_args[0] in ("--help", "-h"):
            return super().parse_args(ctx, args)
        if decision_args[0] in self.commands:
            return super().parse_args(ctx, args)
        args = [self.default_cmd_name] + list(args)
        return super().parse_args(ctx, args)

    def format_usage(self, ctx, formatter):
        formatter.write_usage(
            ctx.command_path,
            "run ENV_OR_CONFIG [ARGS]... | COMMAND [ARGS]...",
        )


app = PlainTyper(
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
        "help_option_names": [],
    },
)
def run_gepa_cmd(
    ctx: typer.Context,
    environment_or_config: str | None = typer.Argument(
        None,
        help="Environment name/slug or TOML config path",
    ),
) -> None:
    """Run optimization with local-first environment resolution."""
    passthrough_args = list(ctx.args)

    if is_help_request(environment_or_config or "", passthrough_args):
        print_gepa_run_help()
        raise typer.Exit(0)

    if environment_or_config is None:
        console.print("[red]Error:[/red] Missing argument 'ENV_OR_CONFIG'.")
        console.print("[dim]Example: prime gepa run wordle --max-calls 100[/dim]")
        raise typer.Exit(2)

    if environment_or_config.startswith("-"):
        console.print("[red]Error:[/red] Environment/config must be the first argument.")
        console.print("[dim]Example: prime gepa run wordle --max-calls 100[/dim]")
        raise typer.Exit(2)

    run_gepa_passthrough(environment_or_config, passthrough_args)
