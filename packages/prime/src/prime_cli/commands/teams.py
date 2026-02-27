import typer
from rich.console import Console
from rich.table import Table

from prime_cli.core import Config

from ..client import APIClient, APIError
from ..utils import output_data_as_json, validate_output_format

app = typer.Typer(help="List your teams", no_args_is_help=True)
console = Console()


def fetch_teams(client: APIClient) -> list[dict]:
    """Fetch teams for the current user (returns data list only)."""
    response = client.get("/user/teams")
    return response.get("data", []) if isinstance(response, dict) else []


def fetch_team_members(client: APIClient, team_id: str) -> list[dict]:
    """Fetch members of a team."""
    response = client.get(f"/teams/{team_id}/members")
    return response.get("data", []) if isinstance(response, dict) else []


@app.command(name="list")
def list_teams(
    limit: int = typer.Option(100, help="Maximum number of teams to list"),
    offset: int = typer.Option(0, help="Number of teams to skip"),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List teams for the current user."""
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
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command(name="members")
def list_members(
    team_id: str = typer.Option(
        None, "--team-id", help="Team ID (uses config team_id if not specified)"
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List members of a team."""
    validate_output_format(output, console)

    config = Config()
    resolved_team_id = team_id or config.team_id

    if not resolved_team_id:
        console.print(
            "[red]Error: No team selected. "
            "Use --team-id or set a team with 'prime config set-team-id'[/red]"
        )
        raise typer.Exit(1)

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
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)
