from typing import Literal

import questionary
import typer
from click.exceptions import Abort

from prime_cli import __version__

from ..client import APIClient, APIError
from ..utils import PlainTyper, get_console

app = PlainTyper(
    help="Submit feedback about Prime Intellect.",
    no_args_is_help=False,
    invoke_without_command=True,
)
console = get_console()

FeedbackCategory = Literal["bug", "feature", "general"]


_CATEGORY_CHOICES: tuple[tuple[FeedbackCategory, str], ...] = (
    ("bug", "Bug report"),
    ("feature", "Feature request"),
    ("general", "General feedback"),
)


def _prompt_category() -> FeedbackCategory:
    selected = questionary.select(
        "What are you sharing?",
        choices=[questionary.Choice(label, value=cat) for cat, label in _CATEGORY_CHOICES],
        default="general",
    ).ask()
    if selected is None:
        raise Abort()
    return selected


def _prompt_message() -> str:
    while True:
        answer = questionary.text("Your feedback (required)").ask()
        if answer is None:
            raise Abort()
        message = answer.strip()
        if message:
            return message
        console.print("[red]Feedback cannot be empty.[/red]")


def _prompt_run_id() -> str | None:
    answer = questionary.text("Related run ID (optional)").ask()
    if answer is None:
        raise Abort()
    return answer.strip() or None


def submit_feedback(
    *,
    message: str,
    category: FeedbackCategory,
    run_id: str | None = None,
) -> None:
    payload = {
        "message": message,
        "category": category,
        "cli_version": __version__,
        "run_id": run_id,
    }

    with console.status("Submitting...", spinner="dots"):
        APIClient().post("/feedback", json=payload)


@app.callback(invoke_without_command=True)
def feedback(ctx: typer.Context) -> None:
    """Submit feedback (bug, feature request, or general) to the Prime team."""
    if ctx.invoked_subcommand is not None:
        return

    console.print("[bold]Prime Feedback[/bold]")
    console.print("[dim]Share bugs, feature ideas, or general thoughts.[/dim]")

    try:
        category = _prompt_category()
        run_id = _prompt_run_id()
        message = _prompt_message()
    except Abort:
        console.print("\n[yellow]Cancelled[/yellow]")
        raise typer.Exit(0)

    try:
        submit_feedback(message=message, category=category, run_id=run_id)
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    console.print("[green]Feedback submitted. Thanks![/green]")
