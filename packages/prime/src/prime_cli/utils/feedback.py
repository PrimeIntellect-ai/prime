"""Feedback submission utility."""

import os
from datetime import datetime, timezone
from typing import Optional

import httpx
import typer
from rich.console import Console

from prime_cli import __version__
from prime_cli.core import Config

console = Console()

SLACK_WEBHOOK_URL = os.getenv(
    "PRIME_FEEDBACK_WEBHOOK",
    "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
)


def _get_user_info() -> dict:
    """Fetch user info from API."""
    from prime_cli.core import APIClient, APIError

    cfg = Config()
    info = {
        "user_id": cfg.user_id,
        "username": None,
        "email": None,
        "team_id": cfg.team_id,
        "team_name": cfg.team_name,
    }

    if cfg.api_key:
        try:
            client = APIClient()
            response = client.get("/user/whoami")
            data = response.get("data", {})
            if isinstance(data, dict):
                info["username"] = data.get("slug") or data.get("name")
                info["email"] = data.get("email")
        except (APIError, Exception):
            pass

    return info


def send_feedback(
    message: str,
    product: str,
    category: str = "general",
    run_id: Optional[str] = None,
) -> bool:
    """Send feedback to Slack.

    Args:
        message: The feedback message
        product: One of 'hosted rl' or 'other'
        category: One of 'bug', 'feature', or 'general'
        run_id: Optional run ID related to the feedback

    Returns:
        True if sent successfully, False otherwise
    """
    user_info = _get_user_info()
    submitted_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    if user_info["username"] and user_info["user_id"]:
        user_display = f"{user_info['username']} ({user_info['user_id']})"
    elif user_info["user_id"]:
        user_display = user_info["user_id"]
    else:
        user_display = "anonymous"

    if user_info["team_name"] and user_info["team_id"]:
        team_display = f"{user_info['team_name']} ({user_info['team_id']})"
    elif user_info["team_id"]:
        team_display = user_info["team_id"]
    else:
        team_display = "personal"

    fields = [
        f"*Datetime submitted:* {submitted_at}",
        f"*CLI version:* {__version__}",
        f"*User:* {user_display}",
        f"*Email:* {user_info['email'] or 'N/A'}",
        f"*Team:* {team_display}",
        f"*Product:* {product}",
        f"*Feedback type:* {category}",
    ]

    if run_id:
        fields.append(f"*Run ID:* {run_id}")

    fields.append(f"*Feedback:* {message}")

    payload = {
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "Feedback Submission", "emoji": False},
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "\n".join(fields)},
            },
        ],
        "text": f"Feedback ({product} / {category}): {message[:80]}",
    }

    try:
        resp = httpx.post(SLACK_WEBHOOK_URL, json=payload, timeout=10.0)
        return resp.status_code == 200
    except httpx.RequestError:
        return False


def prompt_for_feedback() -> tuple[Optional[str], str, str, Optional[str]]:
    """Interactive prompt for feedback.

    Returns:
        Tuple of (message, product, category, run_id) or (None, ...) if cancelled
    """
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
        console.print("\n[dim]If this is related to a specific run, enter the Run ID (or press Enter to skip)[/dim]")
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


def run_feedback_command(
    message: Optional[str] = None,
    bug: bool = False,
    feature: bool = False,
    run_id: Optional[str] = None,
) -> None:
    """Main entry point for the feedback command.

    Args:
        message: Optional feedback message (if None, prompts interactively)
        bug: If True, categorize as bug report
        feature: If True, categorize as feature request
        run_id: Optional run ID related to the feedback
    """
    # Determine category from flags
    if bug:
        category = "bug"
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
                "\n[dim]If this is related to a specific run, enter the Run ID (or press Enter to skip)[/dim]"
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

    with console.status("Submitting...", spinner="dots"):
        success = send_feedback(message, product, category, run_id)

    if success:
        console.print("[green]Feedback submitted. Thanks![/green]")
    else:
        console.print("[red]Failed to send. Please try again or email support@primeintellect.ai[/red]")
        raise typer.Exit(1)
