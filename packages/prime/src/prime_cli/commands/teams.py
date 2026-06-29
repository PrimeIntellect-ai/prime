from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig
from rich.table import Table

from prime_cli.core import Config

from ..client import APIClient, APIError
from ..utils import (
    get_console,
    json_output_help,
    output_data_as_json,
    validate_output_format,
)

console = get_console()

LIST_TEAMS_JSON_HELP = json_output_help(
    ".teams[] = {teamId, name, slug, role, createdAt}",
    ".total_count = number",
)


def fetch_teams(client: APIClient) -> list[dict]:
    """Fetch all teams for the current user across paginated responses."""
    all_teams: list[dict] = []
    offset = 0
    limit = 100

    while True:
        response = client.get("/user/teams", params={"offset": offset, "limit": limit})
        batch = response.get("data", []) if isinstance(response, dict) else []
        all_teams.extend(batch)
        total = response.get("total_count", len(all_teams)) if isinstance(response, dict) else 0

        if not batch or len(all_teams) >= total:
            break

        offset += limit

    return all_teams


def fetch_team_members(client: APIClient, team_id: str) -> list[dict]:
    """Fetch members of a team."""
    response = client.get(f"/teams/{team_id}/members")
    return response.get("data", []) if isinstance(response, dict) else []


def list_teams(config: TeamsListConfig) -> None:
    """List teams for the current user."""
    limit = config.limit
    offset = config.offset
    output = config.output

    validate_output_format(output, console)

    try:
        client = APIClient()
        response = client.get("/user/teams", params={"offset": offset, "limit": limit})

        teams = response.get("data", []) if isinstance(response, dict) else []
        total_count = (
            response.get("total_count", len(teams)) if isinstance(response, dict) else len(teams)
        )

        if output == "json":
            output_data_as_json(
                {
                    "teams": teams,
                    "total_count": total_count,
                    "offset": offset,
                    "limit": limit,
                },
                console,
            )
            return

        table = Table(title=f"Teams (Total: {total_count})", show_lines=True)
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Name", style="blue")
        table.add_column("Slug", style="green")
        table.add_column("Role", style="yellow")
        table.add_column("Created", style="magenta")

        for t in teams:
            table.add_row(
                str(t["teamId"]),
                t["name"],
                t["slug"],
                t["role"],
                str(t["createdAt"]),
            )

        console.print(table)

        if total_count > offset + limit:
            remaining = total_count - (offset + limit)
            console.print(
                f"\n[yellow]Showing {limit} of {total_count} teams. "
                f"Use --offset {offset + limit} to see the next "
                f"{min(limit, remaining)} teams.[/yellow]"
            )

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise SystemExit(1)


def list_members(config: TeamsMembersConfig) -> None:
    """List members of a team."""
    team_id = config.team_id
    output = config.output

    validate_output_format(output, console)

    prime_config = Config()
    resolved_team_id = team_id or prime_config.team_id

    if not resolved_team_id:
        console.print(
            "[red]Error: No team selected. "
            "Use --team-id or set a team with 'prime config set-team-id'[/red]"
        )
        raise SystemExit(1)

    try:
        client = APIClient()
        members = fetch_team_members(client, resolved_team_id)

        if output == "json":
            output_data_as_json({"members": members, "total_count": len(members)}, console)
            return

        table = Table(title=f"Team Members (Total: {len(members)})", show_lines=True)
        table.add_column("User ID", style="cyan", no_wrap=True)
        table.add_column("Name", style="blue")
        table.add_column("Email", style="green")
        table.add_column("Role", style="yellow")
        table.add_column("Joined", style="magenta")

        for m in members:
            table.add_row(
                str(m.get("userId", "")),
                m.get("userName") or "N/A",
                m.get("userEmail") or "N/A",
                m.get("role", ""),
                str(m.get("joinedAt", "")),
            )

        console.print(table)

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise SystemExit(1)


# --- inlined config schemas (previously in teams_configs) ---
class TeamsListConfig(BaseConfig):
    """List teams for the current user."""

    limit: int = Field(100, description="Maximum number of teams to list")
    offset: int = Field(0, description="Number of teams to skip")
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class TeamsMembersConfig(BaseConfig):
    """List members of a team."""

    team_id: str | None = Field(None, description="Team ID (uses config team_id if omitted)")
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )
