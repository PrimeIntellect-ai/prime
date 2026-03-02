from typing import Any, Dict, List, Optional

import typer
from prime_sandboxes import (
    APIClient,
    APIError,
    Config,
    DockerImageCheckResponse,
    RegistryCredentialSummary,
    TemplateClient,
    UnauthorizedError,
)
from rich.console import Console
from rich.markup import escape

from ..utils import (
    build_table,
    human_age,
    iso_timestamp,
    output_data_as_json,
    validate_output_format,
)
from ..utils.prompt import any_provided, prompt_for_value, require_selection

app = typer.Typer(help="Manage registry credentials and private images", no_args_is_help=True)
console = Console()
config = Config()


def _format_registry_row(credential: RegistryCredentialSummary) -> dict:
    server = credential.server or "registry-1.docker.io"
    scope = credential.team_id or (
        "user:" + credential.user_id if credential.user_id else "personal"
    )
    return {
        "id": credential.id,
        "name": credential.name,
        "server": server,
        "scope": scope,
        "team_id": credential.team_id,
        "user_id": credential.user_id,
        "created_at": iso_timestamp(credential.created_at),
        "updated_at": iso_timestamp(credential.updated_at),
        "age": human_age(credential.created_at),
    }


@app.command("list")
def list_registry_credentials(
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List registry credentials available to the current user."""
    validate_output_format(output, console)

    try:
        client = TemplateClient(APIClient())
        credentials = client.list_registry_credentials()
        formatted = [_format_registry_row(cred) for cred in credentials]

        if output == "json":
            output_data_as_json({"credentials": formatted}, console)
            return

        table = build_table(
            "Registry Credentials",
            [
                ("ID", "cyan"),
                ("Name", "green"),
                ("Server", "blue"),
                ("Scope", "magenta"),
                ("Created", "white"),
            ],
        )

        if not formatted:
            console.print("No registry credentials found.")
            return

        for item in formatted:
            table.add_row(
                item["id"],
                item["name"],
                item["server"],
                item["scope"],
                f"{item['created_at']} ({item['age']})",
            )

        console.print(table)

    except UnauthorizedError as e:
        console.print(f"[red]Unauthorized:[/red] {str(e)}")
        raise typer.Exit(1)
    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        console.print_exception(show_locals=True)
        raise typer.Exit(1)


@app.command("check-image")
def check_docker_image(
    image: str = typer.Argument(..., help="Image reference, e.g. ghcr.io/org/repo:tag"),
    registry_credentials_id: Optional[str] = typer.Option(
        None,
        "--registry-credentials-id",
        help="Registry credentials ID for private images",
    ),
) -> None:
    """Verify that an image is accessible (optionally using registry credentials)."""
    try:
        client = TemplateClient(APIClient())
        result: DockerImageCheckResponse = client.check_docker_image(
            image=image, registry_credentials_id=registry_credentials_id
        )

        if result.accessible:
            console.print(f"[green]Image accessible:[/green] {image}")
            if result.details:
                console.print(result.details)
        else:
            console.print(f"[red]Image not accessible:[/red] {result.details or 'Unknown error'}")
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except UnauthorizedError as e:
        console.print(f"[red]Unauthorized:[/red] {str(e)}")
        raise typer.Exit(1)
    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        console.print_exception(show_locals=True)
        raise typer.Exit(1)


def _fetch_credentials(client: TemplateClient) -> List[Dict[str, Any]]:
    """Fetch registry credentials as dicts for interactive selection."""
    credentials = client.list_registry_credentials()
    return [{"id": c.id, "name": c.name, "server": c.server} for c in credentials]


def _credential_display(item: Dict[str, Any]) -> str:
    return f"{item['name']} ({item['server']})"


@app.command("create")
def create_registry_credential(
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Credential name"),
    username: Optional[str] = typer.Option(None, "--username", "-u", help="Registry username"),
    password: Optional[str] = typer.Option(None, "--password", "-p", help="Registry password"),
    server: Optional[str] = typer.Option(
        None, "--server", "-s", help="Registry server (e.g. ghcr.io)"
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Create a new registry credential."""
    validate_output_format(output, console)

    try:
        if not name:
            name = prompt_for_value("Credential name")
            if not name:
                console.print("\n[dim]Cancelled.[/dim]")
                raise typer.Exit()

        if not username:
            username = prompt_for_value("Registry username")
            if not username:
                console.print("\n[dim]Cancelled.[/dim]")
                raise typer.Exit()

        if not password:
            password = prompt_for_value("Registry password", hide_input=True)
            if not password:
                console.print("\n[dim]Cancelled.[/dim]")
                raise typer.Exit()

        if not server:
            server = prompt_for_value("Registry server (e.g. ghcr.io, docker.io)")
            if not server:
                console.print("\n[dim]Cancelled.[/dim]")
                raise typer.Exit()

        client = TemplateClient(APIClient())
        credential = client.create_registry_credential(
            name=name,
            username=username,
            password=password,
            server=server,
            team_id=config.team_id,
        )

        if output == "json":
            output_data_as_json(_format_registry_row(credential), console)
            return

        scope = "team" if config.team_id else "personal"
        console.print(f"[green]Created {scope} registry credential '{name}'[/green]")
        console.print(f"[dim]ID: {credential.id}[/dim]")

    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")
        raise typer.Exit()
    except typer.Exit:
        raise
    except UnauthorizedError as e:
        console.print(f"[red]Unauthorized:[/red] {str(e)}")
        raise typer.Exit(1)
    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        raise typer.Exit(1)


@app.command("update")
def update_registry_credential(
    credential_id: Optional[str] = typer.Argument(
        None, help="Credential ID to update (interactive selection if not provided)"
    ),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="New credential name"),
    username: Optional[str] = typer.Option(None, "--username", "-u", help="New registry username"),
    password: Optional[str] = typer.Option(None, "--password", "-p", help="New registry password"),
    server: Optional[str] = typer.Option(None, "--server", "-s", help="New registry server"),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Update an existing registry credential."""
    validate_output_format(output, console)

    try:
        client = TemplateClient(APIClient())

        if not credential_id:
            credentials = _fetch_credentials(client)
            selected = require_selection(
                credentials,
                "update",
                "No registry credentials to update.",
                item_type="credential",
                display_fn=_credential_display,
            )
            credential_id = selected["id"]

        if not any_provided(name, username, password, server):
            console.print("\n[bold]What would you like to update?[/bold]")
            new_password = prompt_for_value("New password", required=False, hide_input=True)
            if new_password:
                password = new_password

            if not password:
                console.print("\n[dim]No changes made.[/dim]")
                raise typer.Exit()

        credential = client.update_registry_credential(
            credential_id=credential_id,
            name=name,
            username=username,
            password=password,
            server=server,
            team_id=config.team_id,
        )

        if output == "json":
            output_data_as_json(_format_registry_row(credential), console)
            return

        console.print(f"[green]Updated registry credential '{credential.name}'[/green]")

    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")
        raise typer.Exit()
    except typer.Exit:
        raise
    except UnauthorizedError as e:
        console.print(f"[red]Unauthorized:[/red] {str(e)}")
        raise typer.Exit(1)
    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        raise typer.Exit(1)


@app.command("delete")
def delete_registry_credential(
    credential_id: Optional[str] = typer.Argument(
        None, help="Credential ID to delete (interactive selection if not provided)"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Delete a registry credential."""
    try:
        client = TemplateClient(APIClient())

        if not credential_id:
            credentials = _fetch_credentials(client)
            selected = require_selection(
                credentials,
                "delete",
                "No registry credentials to delete.",
                item_type="credential",
                display_fn=_credential_display,
            )
            credential_id = selected["id"]
            credential_name = selected["name"]
        else:
            # Fetch credential name for confirmation
            credentials = _fetch_credentials(client)
            match = next((c for c in credentials if c["id"] == credential_id), None)
            credential_name = match["name"] if match else credential_id

        if not yes:
            confirm = typer.confirm(f"Delete registry credential '{credential_name}'?")
            if not confirm:
                console.print("\n[dim]Cancelled.[/dim]")
                raise typer.Exit()

        client.delete_registry_credential(
            credential_id=credential_id,
            team_id=config.team_id,
        )
        console.print(f"[green]Deleted registry credential '{credential_name}'[/green]")

    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")
        raise typer.Exit()
    except typer.Exit:
        raise
    except UnauthorizedError as e:
        console.print(f"[red]Unauthorized:[/red] {str(e)}")
        raise typer.Exit(1)
    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        raise typer.Exit(1)
