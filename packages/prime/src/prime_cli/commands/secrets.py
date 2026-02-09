from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.table import Table

from prime_cli.core import Config

from ..client import APIClient, APIError
from ..utils import output_data_as_json, validate_output_format
from ..utils.prompt import (
    any_provided,
    prompt_for_value,
    select_item_interactive,
    validate_env_var_name,
)
from ..utils.time_utils import format_time_ago

app = typer.Typer(help="Manage global secrets", no_args_is_help=True)
console = Console()


def _fetch_secrets(client: APIClient, config: Config) -> List[Dict[str, Any]]:
    """Fetch secrets for the current user/team context."""
    params: Dict[str, Any] = {}
    if config.team_id:
        params["teamId"] = config.team_id
    response = client.get("/secrets/", params=params if params else None)
    return response.get("data", [])


@app.command("list")
def secret_list(
    output: str = typer.Option(
        "table",
        "--output",
        "-o",
        help="Output format: table or json",
    ),
) -> None:
    """List your global secrets."""
    validate_output_format(output, console)

    try:
        client = APIClient()
        config = Config()
        secrets = _fetch_secrets(client, config)

        if output == "json":
            output_data_as_json({"secrets": secrets}, console)
            return

        if not secrets:
            scope = "team" if config.team_id else "personal"
            console.print(f"[yellow]No {scope} secrets found.[/yellow]")
            return

        title = "Team Secrets" if config.team_id else "Personal Secrets"
        table = Table(title=title)
        table.add_column("ID", style="dim", no_wrap=True)
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="dim")
        table.add_column("Created", style="dim")

        for secret in secrets:
            secret_id = secret.get("id", "")
            name = secret.get("name", "")
            description = secret.get("description") or ""
            created = secret.get("createdAt", "")
            if created:
                created = format_time_ago(created)
            table.add_row(secret_id, name, description, created)

        console.print(table)

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("create")
def secret_create(
    name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Secret name (used as environment variable name)",
    ),
    value: Optional[str] = typer.Option(
        None,
        "--value",
        "-v",
        help="Secret value",
    ),
    description: Optional[str] = typer.Option(
        None,
        "--description",
        "-d",
        help="Secret description",
    ),
    is_file: bool = typer.Option(
        False,
        "--file",
        "-f",
        help="Treat value as file content (base64 encoded)",
    ),
    output: str = typer.Option(
        "table",
        "--output",
        "-o",
        help="Output format: table or json",
    ),
) -> None:
    """Create a new global secret."""
    validate_output_format(output, console)

    try:
        if not name:
            name = prompt_for_value("Secret name")
            if not name:
                console.print("\n[dim]Cancelled.[/dim]")
                raise typer.Exit()

        if not validate_env_var_name(name, "secret"):
            raise typer.Exit(1)

        if not value:
            value = prompt_for_value("Secret value", hide_input=True)
            if not value:
                console.print("\n[dim]Cancelled.[/dim]")
                raise typer.Exit()

        client = APIClient()
        config = Config()

        payload: Dict[str, Any] = {"name": name, "value": value}
        if description:
            payload["description"] = description
        if is_file:
            payload["isFile"] = True
        if config.team_id:
            payload["teamId"] = config.team_id

        response = client.post("/secrets/", json=payload)
        secret = response.get("data", {})

        if output == "json":
            output_data_as_json(secret, console)
            return

        scope = "team" if config.team_id else "personal"
        console.print(f"[green]✓ Created {scope} secret '{name}'[/green]")
        console.print(f"[dim]ID: {secret.get('id')}[/dim]")

    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")
        raise typer.Exit()
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("update")
def secret_update(
    secret_id: Optional[str] = typer.Argument(
        None,
        help="Secret ID to update (interactive selection if not provided)",
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="New secret name",
    ),
    value: Optional[str] = typer.Option(
        None,
        "--value",
        "-v",
        help="New secret value",
    ),
    description: Optional[str] = typer.Option(
        None,
        "--description",
        "-d",
        help="New secret description",
    ),
    output: str = typer.Option(
        "table",
        "--output",
        "-o",
        help="Output format: table or json",
    ),
) -> None:
    """Update an existing global secret."""
    validate_output_format(output, console)

    try:
        client = APIClient()
        config = Config()

        if not secret_id:
            secrets = _fetch_secrets(client, config)
            if not secrets:
                scope = "team" if config.team_id else "personal"
                console.print(f"[yellow]No {scope} secrets to update.[/yellow]")
                raise typer.Exit()

            selected = select_item_interactive(secrets, "update", item_type="secret")
            if not selected:
                console.print("\n[dim]Cancelled.[/dim]")
                raise typer.Exit()

            secret_id = selected.get("id")

        if not any_provided(name, value, description):
            console.print("\n[bold]What would you like to update?[/bold]")
            new_value = prompt_for_value("New value", required=False, hide_input=True)
            if new_value:
                value = new_value

            if not value:
                console.print("\n[dim]No changes made.[/dim]")
                raise typer.Exit()

        if name is not None and not validate_env_var_name(name, "secret"):
            raise typer.Exit(1)

        payload: Dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if value is not None:
            payload["value"] = value
        if description is not None:
            payload["description"] = description

        response = client.patch(
            f"/secrets/{secret_id}",
            json=payload,
        )
        secret = response.get("data", {})

        if output == "json":
            output_data_as_json(secret, console)
            return

        console.print(f"[green]✓ Updated secret '{secret.get('name')}'[/green]")

    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")
        raise typer.Exit()
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("delete")
def secret_delete(
    secret_id: Optional[str] = typer.Argument(
        None,
        help="Secret ID to delete (interactive selection if not provided)",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt",
    ),
) -> None:
    """Delete a global secret."""
    try:
        client = APIClient()
        config = Config()

        if not secret_id:
            secrets = _fetch_secrets(client, config)
            if not secrets:
                scope = "team" if config.team_id else "personal"
                console.print(f"[yellow]No {scope} secrets to delete.[/yellow]")
                raise typer.Exit()

            selected = select_item_interactive(secrets, "delete", item_type="secret")
            if not selected:
                console.print("\n[dim]Cancelled.[/dim]")
                raise typer.Exit()

            secret_id = selected.get("id")
            secret_name = selected.get("name")
        else:
            response = client.get(f"/secrets/{secret_id}")
            secret_data = response.get("data", {})
            secret_name = secret_data.get("name", secret_id)

        if not yes:
            confirm = typer.confirm(f"Delete secret '{secret_name}'?")
            if not confirm:
                console.print("\n[dim]Cancelled.[/dim]")
                raise typer.Exit()

        client.delete(f"/secrets/{secret_id}")
        console.print(f"[green]✓ Deleted secret '{secret_name}'[/green]")

    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")
        raise typer.Exit()
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("get")
def secret_get(
    secret_id: str = typer.Argument(
        ...,
        help="Secret ID to get",
    ),
    output: str = typer.Option(
        "table",
        "--output",
        "-o",
        help="Output format: table or json",
    ),
) -> None:
    """Get details of a specific secret."""
    validate_output_format(output, console)

    try:
        client = APIClient()

        response = client.get(f"/secrets/{secret_id}")
        secret = response.get("data", {})

        if output == "json":
            output_data_as_json(secret, console)
            return

        console.print("\n[bold]Secret Details[/bold]")
        console.print(f"  ID:          {secret.get('id')}")
        console.print(f"  Name:        {secret.get('name')}")
        console.print(f"  Description: {secret.get('description') or '-'}")
        console.print(f"  Created:     {format_time_ago(secret.get('createdAt', ''))}")
        console.print(f"  Updated:     {format_time_ago(secret.get('updatedAt', ''))}")
        console.print()

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
