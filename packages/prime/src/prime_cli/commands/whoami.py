from typing import Any, Dict, Optional

import typer
from rich.console import Console
from rich.table import Table

from prime_cli.core import Config

from ..client import APIClient, APIError
from .teams import fetch_teams

app = typer.Typer(help="Show current authenticated user and update config", no_args_is_help=False)
console = Console()


def _resolve_team_context(
    client: APIClient, config: Config
) -> tuple[str, Optional[str], Optional[str]]:
    """
    Resolve the current account context based on configured team_id.

    Returns:
        (account_label, team_label, role_label)
    """
    team_id = config.team_id
    if not team_id:
        return "Personal", None, None

    team_name: Optional[str] = None if config.team_id_from_env else config.team_name
    role: Optional[str] = None

    try:
        teams = fetch_teams(client)
        for t in teams:
            if t.get("teamId") == team_id:
                team_name = t.get("name") or team_name
                role = t.get("role") or role
                break
    except Exception:
        # Best-effort only; whoami should still work without teams resolution.
        pass

    if team_name and (not config.team_id_from_env) and (config.team_name != team_name):
        config.set_team(team_id, team_name=team_name)
        config.update_current_environment_file()

    team_label = f"{team_name} ({team_id})" if team_name else team_id
    role_label = role.lower() if isinstance(role, str) else None
    return "Team", team_label, role_label


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

        # Display user info table
        table = Table(title="Current User")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")
        account_label, team_label, role_label = _resolve_team_context(client, config)
        table.add_row("Account", account_label)
        if team_label:
            table.add_row("Team", team_label)
        if role_label:
            table.add_row("Team Role", role_label)
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
