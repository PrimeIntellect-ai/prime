from typing import Any, Dict, Optional

import typer
from rich.table import Table
from rich.text import Text

from prime_cli.core import Config

from ..client import APIClient, APIError
from ..utils import PlainTyper, get_console

app = PlainTyper(help="Show current authenticated user and update config", no_args_is_help=False)
console = get_console()

# (label, account limit field, API key limit field). A None field means that
# side has no such limit: the account caps VM sandboxes but not sandboxes
# overall, and only API keys carry a concurrent or hourly sandbox cap.
LIMIT_ROWS = [
    ("Concurrent sandboxes", None, "max_concurrent_sandboxes"),
    ("Sandbox creations / hour", None, "max_sandbox_creations_per_hour"),
    ("Sandbox CPU cores", "sandbox_total_cpu_limit", "max_sandbox_cpu_cores"),
    ("Concurrent VM sandboxes", "vm_sandbox_limit", None),
    ("Sandbox GPUs", "vm_sandbox_gpu_limit", "max_sandbox_gpu_count"),
    ("Concurrent tunnels", "tunnel_limit", "max_concurrent_tunnels"),
    ("Tunnel creations / hour", "tunnel_creations_per_hour_limit", "max_tunnel_creations_per_hour"),
    ("Tunnel TTL (hours)", "tunnel_ttl_hours", "max_tunnel_ttl_hours"),
]


def _limit_cell(value: Optional[int]) -> Any:
    return str(value) if value is not None else Text("-", style="dim")


def build_limits_table(
    account_limits: Optional[Dict[str, Any]], key_limits: Optional[Dict[str, Any]]
) -> Table:
    """Account limits next to the API key's, with the lower one as effective."""
    table = Table(title="Limits")
    table.add_column("Limit", style="cyan")
    table.add_column("Account", justify="right")
    table.add_column("API Key", justify="right")
    table.add_column("Effective", style="green", justify="right")

    for label, account_field, key_field in LIMIT_ROWS:
        account = (account_limits or {}).get(account_field) if account_field else None
        key = (key_limits or {}).get(key_field) if key_field else None
        set_values = [value for value in (account, key) if value is not None]
        effective = min(set_values) if set_values else None
        table.add_row(label, _limit_cell(account), _limit_cell(key), _limit_cell(effective))

    return table


@app.callback(invoke_without_command=True)
def whoami() -> None:
    """Fetch identity from the API and set user_id in config."""
    try:
        client = APIClient()
        # Account limits are per wallet, so ask for the active team's.
        team_id = Config().team_id
        response: Dict[str, Any] = client.get(
            "/user/whoami", params={"teamId": team_id} if team_id else None
        )
        data = response.get("data") if isinstance(response, dict) else None
        if not isinstance(data, dict):
            console.print("[red]Unexpected response from whoami endpoint[/red]")
            raise typer.Exit(1)

        user_id = data.get("id")
        email = data.get("email")
        name = data.get("name")
        slug = data.get("slug")
        scope = data.get("scope", {})
        account_limits = data.get("account_limits")
        key_limits = data.get("key_limits")

        # Update config
        config = Config()
        if user_id and config.context_override is None:
            config.set_user_id(user_id, user_name=name)
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
            table.add_row("Team Name", config.team_name or Text("Unknown", style="dim"))
            if config.team_role:
                table.add_row("Role", config.team_role)
        else:
            table.add_row("Type", "Personal")

        # Add section divider between account and user details
        table.add_section()

        # User details
        table.add_row("User ID", user_id or "Unknown")
        table.add_row("Username", slug or Text("Not set", style="dim"))
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

        console.print()
        console.print(build_limits_table(account_limits, key_limits))
        if not account_limits:
            console.print("[dim]Account limits are unavailable for this account.[/dim]")

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)
