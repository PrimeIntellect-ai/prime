from prime_sandboxes import (
    APIClient,
    APIError,
    DockerImageCheckResponse,
    RegistryCredentialSummary,
    TemplateClient,
    UnauthorizedError,
)
from rich.markup import escape

from ..utils import (
    build_table,
    get_console,
    human_age,
    iso_timestamp,
    json_output_help,
    output_data_as_json,
    validate_output_format,
)
from .registry_configs import RegistryCheckImageConfig, RegistryListConfig

console = get_console()

LIST_REGISTRY_JSON_HELP = json_output_help(
    ".credentials[] = {id, name, server, scope, team_id, user_id, created_at, updated_at, age}",
)


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


def list_registry_credentials(config: RegistryListConfig) -> None:
    """List registry credentials available to the current user."""
    output = config.output

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
        raise SystemExit(1)
    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        console.print_exception(show_locals=True)
        raise SystemExit(1)


def check_docker_image(config: RegistryCheckImageConfig) -> None:
    """Verify that an image is accessible (optionally using registry credentials)."""
    image = config.image
    registry_credentials_id = config.registry_credentials_id

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
            raise SystemExit(1)

    except SystemExit:
        raise
    except UnauthorizedError as e:
        console.print(f"[red]Unauthorized:[/red] {str(e)}")
        raise SystemExit(1)
    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        console.print_exception(show_locals=True)
        raise SystemExit(1)
