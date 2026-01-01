from typing import Any, Dict

import typer
from rich.console import Console
from rich.table import Table

from prime_cli.core import Config

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
        slug = data.get("slug")
        scope = data.get("scope", {})

        # Update config
        config = Config()
        if user_id:
            config.set_user_id(user_id)
            config.update_current_environment_file()

        # Display account info table
        table = Table(title="Account")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")

        # Account type (Team or Personal) - shown first
        if config.team_id:
            table.add_row("Type", "Team")
            table.add_section()
            table.add_row("Team ID", config.team_id)
            table.add_row("Team Name", config.team_name or "[dim]Unknown[/dim]")
            if config.team_role:
                table.add_row("Role", config.team_role)
        else:
            table.add_row("Type", "Personal")

        # Add section divider between account and user details
        table.add_section()

        # User details
        table.add_row("User ID", user_id or "Unknown")
        table.add_row("Username", slug or "[dim]Not set[/dim]")
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
