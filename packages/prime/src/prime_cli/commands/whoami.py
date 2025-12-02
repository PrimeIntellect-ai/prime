from typing import Any, Dict

import typer
from prime_intellect_core import Config
from rich.console import Console
from rich.table import Table

from ..client import APIClient, APIError

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
        scope = data.get("scope", {})

        # Update config
        config = Config()
        if user_id:
            config.set_user_id(user_id)
            config.update_current_environment_file()

        # Display user info table
        table = Table(title="Current User")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("User ID", user_id or "Unknown")
        table.add_row("Name", name or "Unknown")
        table.add_row("Email", email or "Unknown")
        console.print(table)

        # Display permissions table
        if scope:
            console.print()
            perms_table = Table(title="Token Permissions")
            perms_table.add_column("Scope", style="cyan")
            perms_table.add_column("Read", style="magenta", justify="center")
            perms_table.add_column("Write", style="magenta", justify="center")

            for scope_name, permissions in scope.items():
                if permissions is None:
                    perms_table.add_row(scope_name, "-", "-")
                else:
                    read_val = "✓" if permissions.get("read", False) else "✗"
                    write_val = "✓" if permissions.get("write", False) else "✗"
                    perms_table.add_row(scope_name, read_val, write_val)

            console.print(perms_table)

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)
