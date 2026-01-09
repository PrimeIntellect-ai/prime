"""Feedback submission command."""

from typing import Optional

import typer
from rich.console import Console

from ..utils.feedback import run_feedback_command

console = Console()

app = typer.Typer(
    help="Submit feedback about Prime.",
    no_args_is_help=False,
    invoke_without_command=True,
)


@app.callback(invoke_without_command=True)
def feedback(
    ctx: typer.Context,
    message: Optional[str] = typer.Argument(None, help="Your feedback"),
    bug: bool = typer.Option(False, "--bug", "-b", help="Report a bug"),
    feature: bool = typer.Option(False, "--feature", "-f", help="Request a feature"),
    run_id: Optional[str] = typer.Option(None, "--run", "-r", help="Related run ID"),
) -> None:
    """Submit feedback about Prime.

    Examples:
        prime feedback "Great CLI!"
        prime feedback --bug "Something broke"
        prime feedback -f "Add dark mode"
        prime feedback --bug --run abc123 "OOM error"
        prime feedback
    """
    if ctx.invoked_subcommand is None:
        run_feedback_command(message, bug=bug, feature=feature, run_id=run_id)
