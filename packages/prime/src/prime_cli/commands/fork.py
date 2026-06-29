from __future__ import annotations

from typing import Dict, Optional, Tuple

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig

from ..client import APIClient, APIError
from ..utils import get_console, json_output_help, output_data_as_json, validate_output_format

console = get_console()

FORK_JSON_HELP = json_output_help(
    ". = {success, message, environment_id, version_id, name, owner, slug}",
)


def _parse_fork_source(environment: str) -> Tuple[str, str]:
    """Parse a fork source slug in owner/name format."""
    if "@" in environment:
        raise ValueError("Forking a specific version is not supported; omit the @version suffix")

    parts = environment.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("Invalid environment format. Expected: owner/name")

    return parts[0], parts[1]


def _build_fork_payload(client: APIClient, team: Optional[str]) -> Dict[str, str]:
    """Build the fork request payload from CLI flags and configured context."""
    if team:
        return {"team_slug": team}
    if client.config.team_id:
        return {"team_id": client.config.team_id}
    return {}


def fork(config: ForkConfig) -> None:
    """Fork a public environment into your Prime Intellect namespace."""
    environment = config.environment
    team = config.team
    output = config.output

    validate_output_format(output, console)

    try:
        owner, name = _parse_fork_source(environment)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise SystemExit(1)

    try:
        client = APIClient()
        payload = _build_fork_payload(client, team)
        response = client.post(f"/environmentshub/{owner}/{name}/fork", json=payload)
        data = response.get("data", response)

        if output == "json":
            output_data_as_json(data, console)
            return

        slug = data.get("slug")
        environment_id = data.get("environment_id")
        version_id = data.get("version_id")

        console.print(f"[green]✓ Forked {environment} to {slug}[/green]")
        console.print(f"Environment ID: [dim]{environment_id}[/dim]")
        console.print(f"Version ID: [dim]{version_id}[/dim]")
        console.print("\nNext steps:")
        console.print(f"  prime env pull {slug}")
        console.print(
            "  # edit the pulled environment, then run `prime env push` from that directory"
        )
    except APIError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise SystemExit(1)


# --- inlined config schemas (previously in fork_configs) ---
class ForkConfig(BaseConfig):
    """Fork a public environment into your Prime Intellect namespace."""

    environment: str = Field(..., description="Public environment to fork, in owner/name format")
    team: str | None = Field(
        None,
        validation_alias=AliasChoices("team", "t"),
        description="Team slug to fork into (uses configured team ID if omitted)",
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )
