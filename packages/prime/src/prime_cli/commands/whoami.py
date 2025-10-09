from typing import Any, Dict

import typer
from rich.console import Console
from rich.table import Table

from ..api.client import APIClient, APIError
from ..config import Config

app = typer.Typer(help="Show current authenticated user and update config", no_args_is_help=False)
console = Console()


@app.callback(invoke_without_command=True)
def whoami() -> None:
    """Fetch identity from the API and set user_id in config."""
    try:
        client = APIClient()
        response: Dict[str, Any] = client.get("/user/whoami")
        data = response.get("data") if isinstance(response, dict) else None
        if not isinstance(data, dict):
            console.print("[red]Unexpected response from whoami endpoint[/red]")
            raise typer.Exit(1)

        user_id = data.get("id")
        email = data.get("email")
        name = data.get("name")

        # Update config
        config = Config()
        if user_id:
            config.set_user_id(user_id)
            config.update_current_environment_file()

        # Display table
        table = Table(title="Current User")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("User ID", user_id or "Unknown")
        table.add_row("Name", name or "Unknown")
        table.add_row("Email", email or "Unknown")
        console.print(table)

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)
