from typing import Optional

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

app = typer.Typer(help="Manage registry credentials and private images", no_args_is_help=True)
console = Console()
config = Config()


def _format_registry_row(credential: RegistryCredentialSummary) -> dict:
    server = credential.server or "registry-1.docker.io"
    return {
        "id": credential.id,
        "name": credential.name,
        "server": server,
        "scope": credential.team_id or credential.user_id or "personal",
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
            scope_display = item["team_id"] or (
                "user:" + item["user_id"] if item["user_id"] else "personal"
            )
            table.add_row(
                item["id"],
                item["name"],
                item["server"],
                scope_display,
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
