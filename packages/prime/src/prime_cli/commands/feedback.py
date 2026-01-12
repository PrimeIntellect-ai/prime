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
    if ctx.invoked_subcommand is not None:
        return

    # Determine category from flags
    if bug:
        category: Optional[str] = "bug"
    elif feature:
        category = "feature"
    else:
        category = None

    # If message provided via CLI, still need to prompt for product
    if message:
        console.print("\n[bold]Prime Feedback[/bold]\n")

        # Product selection
        console.print("[bold]Product:[/bold]")
        console.print("  1. Hosted RL")
        console.print("  2. Other")

        try:
            product_choice = typer.prompt("\nSelect", type=int, default=1)
        except (KeyboardInterrupt, typer.Abort):
            console.print("\n[yellow]Cancelled[/yellow]")
            raise typer.Exit(0)

        product = {1: "hosted rl"}.get(product_choice, "other")

        # If no category from flags, prompt for it
        if category is None:
            console.print("\n[bold]Feedback type:[/bold]")
            console.print("  1. Bug report")
            console.print("  2. Feature request")
            console.print("  3. General feedback")

            try:
                category_choice = typer.prompt("\nSelect", type=int, default=3)
            except (KeyboardInterrupt, typer.Abort):
                console.print("\n[yellow]Cancelled[/yellow]")
                raise typer.Exit(0)

            category = {1: "bug", 2: "feature"}.get(category_choice, "general")

        # Prompt for run ID if hosted rl and not provided
        if product == "hosted rl" and run_id is None:
            console.print(
                "\n[dim]If related to a specific run, enter Run ID (or Enter to skip)[/dim]"
            )
            try:
                run_id = typer.prompt("Run ID", default="", show_default=False).strip() or None
            except (KeyboardInterrupt, typer.Abort):
                console.print("\n[yellow]Cancelled[/yellow]")
                raise typer.Exit(0)
    else:
        # Fully interactive mode
        prompted_message, product, prompted_category, prompted_run_id = prompt_for_feedback()
        if not prompted_message:
            console.print("\n[yellow]Cancelled[/yellow]")
            raise typer.Exit(0)
        message = prompted_message
        if category is None:
            category = prompted_category
        if run_id is None:
            run_id = prompted_run_id

    try:
        with console.status("Submitting...", spinner="dots"):
            send_feedback(message, product, category or "general", run_id)
        console.print("[green]Feedback submitted. Thanks![/green]")
    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
