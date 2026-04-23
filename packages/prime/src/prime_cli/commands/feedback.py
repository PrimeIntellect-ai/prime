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


_CATEGORY_CHOICES: tuple[tuple[str, str], ...] = (
    ("bug", "Bug report"),
    ("feature", "Feature request"),
    ("general", "General feedback"),
)


def _prompt_category() -> str:
    console.print("\n[bold]What are you sharing?[/bold]")
    for idx, (_, label) in enumerate(_CATEGORY_CHOICES, start=1):
        console.print(f"  [cyan]({idx})[/cyan] {label}")

    while True:
        selection = typer.prompt("Select", type=int, default=3)
        if 1 <= selection <= len(_CATEGORY_CHOICES):
            return _CATEGORY_CHOICES[selection - 1][0]
        console.print(f"[red]Invalid selection. Enter 1-{len(_CATEGORY_CHOICES)}.[/red]")


def _prompt_message() -> str:
    console.print("\n[bold]Your feedback[/bold] [dim](required)[/dim]")
    while True:
        message = typer.prompt("", prompt_suffix="> ").strip()
        if message:
            return message
        console.print("[red]Feedback cannot be empty.[/red]")


def _prompt_run_id() -> str | None:
    run_id = typer.prompt(
        "Related run ID (optional)",
        default="",
        show_default=False,
    ).strip()
    return run_id or None


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

    payload = {
        "message": message,
        "category": category,
        "cli_version": __version__,
        "run_id": run_id,
    }

    try:
        with console.status("Submitting...", spinner="dots"):
            APIClient().post("/feedback", json=payload)
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    console.print("[green]Feedback submitted. Thanks![/green]")
