"""GEPA commands."""

import typer

from ..utils import DefaultCommandGroup, PlainTyper, is_plain_mode
from ..verifiers_bridge import exec_verifiers_process


class DefaultGroup(DefaultCommandGroup):
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
    """Run Verifiers' native GEPA command."""
    args = ([environment_or_config] if environment_or_config else []) + list(ctx.args)
    exec_verifiers_process("gepa", args, plain=is_plain_mode())
