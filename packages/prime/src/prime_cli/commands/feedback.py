from typing import Literal

from prime_cli import __version__

from ..client import APIClient, APIError
from ..utils import get_console
from ..utils.prompt import prompt
from .feedback_configs import FeedbackConfig

console = get_console()

FeedbackCategory = Literal["bug", "feature", "general"]


_CATEGORY_CHOICES: tuple[tuple[FeedbackCategory, str], ...] = (
    ("bug", "Bug report"),
    ("feature", "Feature request"),
    ("general", "General feedback"),
)


def _prompt_category() -> FeedbackCategory:
    console.print("\n[bold]What are you sharing?[/bold]")
    for idx, (_, label) in enumerate(_CATEGORY_CHOICES, start=1):
        console.print(f"  [cyan]({idx})[/cyan] {label}")

    while True:
        selection = prompt("Select", type=int, default=3)
        if 1 <= selection <= len(_CATEGORY_CHOICES):
            return _CATEGORY_CHOICES[selection - 1][0]
        console.print(f"[red]Invalid selection. Enter 1-{len(_CATEGORY_CHOICES)}.[/red]")


def _prompt_message() -> str:
    console.print("\n[bold]Your feedback[/bold] [dim](required)[/dim]")
    while True:
        message = prompt("", prompt_suffix="> ").strip()
        if message:
            return message
        console.print("[red]Feedback cannot be empty.[/red]")


def _prompt_run_id() -> str | None:
    run_id = prompt(
        "Related run ID (optional)",
        default="",
        show_default=False,
    ).strip()
    return run_id or None


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


def feedback(config: FeedbackConfig) -> None:
    """Submit feedback (bug, feature request, or general) to the Prime team."""
    console.print("[bold]Prime Feedback[/bold]")
    console.print("[dim]Share bugs, feature ideas, or general thoughts.[/dim]")

    try:
        category = _prompt_category()
        run_id = _prompt_run_id()
        message = _prompt_message()
    except (EOFError, KeyboardInterrupt):
        console.print("\n[yellow]Cancelled[/yellow]")
        raise SystemExit(0)

    try:
        submit_feedback(message=message, category=category, run_id=run_id)
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)

    console.print("[green]Feedback submitted. Thanks![/green]")
