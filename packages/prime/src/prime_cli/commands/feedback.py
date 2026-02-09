from typing import Optional

import typer
from rich.console import Console

from prime_cli import __version__
from prime_cli.core import APIClient, APIError, Config

app = typer.Typer(
    help="Submit feedback about Prime.",
    no_args_is_help=False,
    invoke_without_command=True,
)
console = Console()


def send_feedback(
    message: str,
    product: str,
    category: str = "general",
    run_id: Optional[str] = None,
) -> None:
    cfg = Config()

    payload = {
        "message": message,
        "product": product,
        "category": category,
        "run_id": run_id,
        "cli_version": __version__,
        "user_id": cfg.user_id,
        "team_id": cfg.team_id,
        "team_name": cfg.team_name,
    }

    client = APIClient()
    client.post("/feedback", json=payload)


def prompt_for_feedback() -> tuple[Optional[str], str, str, Optional[str]]:
    console.print("\n[bold]Prime Feedback[/bold]")
    console.print("[dim]Share bugs, feature ideas, or general thoughts.[/dim]\n")

    # Product selection
    console.print("[bold]Product:[/bold]")
    console.print("  1. Hosted RL")
    console.print("  2. Other")

    try:
        product_choice = typer.prompt("\nSelect", type=int, default=1)
    except (KeyboardInterrupt, typer.Abort):
        return None, "other", "general", None

    product = {1: "hosted rl"}.get(product_choice, "other")

    # Category selection
    console.print("\n[bold]Feedback type:[/bold]")
    console.print("  1. Bug report")
    console.print("  2. Feature request")
    console.print("  3. General feedback")

    try:
        category_choice = typer.prompt("\nSelect", type=int, default=3)
    except (KeyboardInterrupt, typer.Abort):
        return None, product, "general", None

    category = {1: "bug", 2: "feature"}.get(category_choice, "general")

    # Run ID prompt (only for hosted rl)
    run_id = None
    if product == "hosted rl":
        console.print(
            "\n[dim]If related to a specific run, enter Run ID (or Enter to skip)[/dim]"
        )
        try:
            run_id = typer.prompt("Run ID", default="", show_default=False).strip() or None
        except (KeyboardInterrupt, typer.Abort):
            return None, product, category, None

    # Feedback message
    console.print("\n[bold]Enter your feedback:[/bold]")

    try:
        message = typer.prompt("").strip()
    except (KeyboardInterrupt, typer.Abort):
        return None, product, category, None

    return message if message else None, product, category, run_id


@app.callback(invoke_without_command=True)
def feedback(ctx: typer.Context) -> None:
    """Submit feedback about Prime.

    Example:
        prime feedback
    """
    if ctx.invoked_subcommand is not None:
        return

    message, product, category, run_id = prompt_for_feedback()
    if not message:
        console.print("\n[yellow]Cancelled[/yellow]")
        raise typer.Exit(0)

    try:
        with console.status("Submitting...", spinner="dots"):
            send_feedback(message, product, category, run_id)
        console.print("[green]Feedback submitted. Thanks![/green]")
    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
