from __future__ import annotations

from typing import Any, Dict, List

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig
from rich.table import Table

from prime_cli.core import Config as PrimeConfig

from ..client import APIClient, APIError
from ..utils import (
    get_console,
    json_output_help,
    optional_team_params,
    output_data_as_json,
    validate_output_format,
)
from ..utils.prompt import (
    any_provided,
    confirm,
    prompt_for_value,
    require_selection,
    validate_env_var_name,
)
from ..utils.time_utils import format_time_ago

console = get_console()

SECRET_LIST_JSON_HELP = json_output_help(
    ".secrets[] = {id, name, description?, createdAt, updatedAt?}",
)

SECRET_DETAIL_JSON_HELP = json_output_help(
    ". = {id, name, description?, value?, createdAt, updatedAt?}",
)


def _fetch_secrets(client: APIClient, config: PrimeConfig) -> List[Dict[str, Any]]:
    """Fetch secrets for the current user/team context."""
    response = client.get("/secrets/", params=optional_team_params(config))
    return response.get("data", [])


def secret_list(config: SecretListConfig) -> None:
    """List your global secrets."""
    output = config.output

    validate_output_format(output, console)

    try:
        client = APIClient()
        prime_config = PrimeConfig()
        secrets = _fetch_secrets(client, prime_config)

        if output == "json":
            output_data_as_json({"secrets": secrets}, console)
            return

        if not secrets:
            scope = "team" if prime_config.team_id else "personal"
            console.print(f"[yellow]No {scope} secrets found.[/yellow]")
            return

        title = "Team Secrets" if prime_config.team_id else "Personal Secrets"
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
        raise SystemExit(1)


def secret_create(config: SecretCreateConfig) -> None:
    """Create a new global secret."""
    name = config.name
    value = config.value
    description = config.description
    is_file = config.file
    output = config.output

    validate_output_format(output, console)

    try:
        if not name:
            name = prompt_for_value("Secret name")
            if not name:
                console.print("\n[dim]Cancelled.[/dim]")
                raise SystemExit(0)

        if not validate_env_var_name(name, "secret"):
            raise SystemExit(1)

        if not value:
            value = prompt_for_value("Secret value", hide_input=True)
            if not value:
                console.print("\n[dim]Cancelled.[/dim]")
                raise SystemExit(0)

        client = APIClient()
        prime_config = PrimeConfig()

        payload: Dict[str, Any] = {"name": name, "value": value}
        if description:
            payload["description"] = description
        if is_file:
            payload["isFile"] = True
        if prime_config.team_id:
            payload["teamId"] = prime_config.team_id

        response = client.post("/secrets/", json=payload)
        secret = response.get("data", {})

        if output == "json":
            output_data_as_json(secret, console)
            return

        scope = "team" if prime_config.team_id else "personal"
        console.print(f"[green]✓ Created {scope} secret '{name}'[/green]")
        console.print(f"[dim]ID: {secret.get('id')}[/dim]")

    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")
        raise SystemExit(0)
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)


def secret_update(config: SecretUpdateConfig) -> None:
    """Update an existing global secret."""
    secret_id = config.secret_id
    name = config.name
    value = config.value
    description = config.description
    output = config.output

    validate_output_format(output, console)

    try:
        client = APIClient()
        prime_config = PrimeConfig()

        if not secret_id:
            secrets = _fetch_secrets(client, prime_config)
            scope = "team" if prime_config.team_id else "personal"
            selected = require_selection(secrets, "update", f"No {scope} secrets to update.")
            secret_id = selected.get("id")

        if not any_provided(name, value, description):
            console.print("\n[bold]What would you like to update?[/bold]")
            new_value = prompt_for_value("New value", required=False, hide_input=True)
            if new_value:
                value = new_value

            if not value:
                console.print("\n[dim]No changes made.[/dim]")
                raise SystemExit(0)

        if name is not None and not validate_env_var_name(name, "secret"):
            raise SystemExit(1)

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
            params=optional_team_params(prime_config),
        )
        secret = response.get("data", {})

        if output == "json":
            output_data_as_json(secret, console)
            return

        console.print(f"[green]✓ Updated secret '{secret.get('name')}'[/green]")

    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")
        raise SystemExit(0)
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)


def secret_delete(config: SecretDeleteConfig) -> None:
    """Delete a global secret."""
    secret_id = config.secret_id
    yes = config.yes

    try:
        client = APIClient()
        prime_config = PrimeConfig()

        if not secret_id:
            secrets = _fetch_secrets(client, prime_config)
            scope = "team" if prime_config.team_id else "personal"
            selected = require_selection(secrets, "delete", f"No {scope} secrets to delete.")
            secret_id = selected.get("id")
            secret_name = selected.get("name")
        else:
            response = client.get(
                f"/secrets/{secret_id}", params=optional_team_params(prime_config)
            )
            secret_data = response.get("data", {})
            secret_name = secret_data.get("name", secret_id)

        if not yes:
            confirmed = confirm(f"Delete secret '{secret_name}'?")
            if not confirmed:
                console.print("\n[dim]Cancelled.[/dim]")
                raise SystemExit(0)

        client.delete(f"/secrets/{secret_id}", params=optional_team_params(prime_config))
        console.print(f"[green]✓ Deleted secret '{secret_name}'[/green]")

    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")
        raise SystemExit(0)
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)


def secret_get(config: SecretGetConfig) -> None:
    """Get details of a specific secret."""
    secret_id = config.secret_id
    output = config.output

    validate_output_format(output, console)

    try:
        client = APIClient()
        prime_config = PrimeConfig()

        response = client.get(f"/secrets/{secret_id}", params=optional_team_params(prime_config))
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
        raise SystemExit(1)


# --- inlined config schemas (previously in secrets_configs) ---
class SecretCreateConfig(BaseConfig):
    """Create a new global secret."""

    name: str | None = Field(
        None,
        validation_alias=AliasChoices("name", "n"),
        description="Secret name (used as environment variable name)",
    )
    value: str | None = Field(
        None, validation_alias=AliasChoices("value", "v"), description="Secret value"
    )
    description: str | None = Field(
        None, validation_alias=AliasChoices("description", "d"), description="Secret description"
    )
    file: bool = Field(
        False,
        validation_alias=AliasChoices("file", "f"),
        description="Treat value as file content (base64 encoded)",
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class SecretDeleteConfig(BaseConfig):
    """Delete a global secret."""

    secret_id: str | None = Field(
        None, description="Secret ID to delete (interactive selection if not provided)"
    )
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


class SecretGetConfig(BaseConfig):
    """Get details of a specific secret."""

    secret_id: str = Field(..., description="Secret ID to get")
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class SecretListConfig(BaseConfig):
    """List your global secrets."""

    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class SecretUpdateConfig(BaseConfig):
    """Update an existing global secret."""

    secret_id: str | None = Field(
        None, description="Secret ID to update (interactive selection if not provided)"
    )
    name: str | None = Field(
        None, validation_alias=AliasChoices("name", "n"), description="New secret name"
    )
    value: str | None = Field(
        None, validation_alias=AliasChoices("value", "v"), description="New secret value"
    )
    description: str | None = Field(
        None,
        validation_alias=AliasChoices("description", "d"),
        description="New secret description",
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )
