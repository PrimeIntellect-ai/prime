import typer
from rich.console import Console
from rich.table import Table

from ..api.client import APIClient, APIError
from ..utils import output_data_as_json, validate_output_format

app = typer.Typer(help="List your teams", no_args_is_help=True)
console = Console()


@app.command(name="list")
def list_teams(
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List teams for the current user."""
    validate_output_format(output, console)

    try:
        client = APIClient()
        response = client.get("/user/teams")
        data = response.get("data") if isinstance(response, dict) else []

        if output == "json":
            teams = data if isinstance(data, list) else []
            output_data_as_json({"teams": teams, "total_count": len(teams)}, console)
            return

        teams = data if isinstance(data, list) else []
        table = Table(title=f"Teams (Total: {len(teams)})", show_lines=True)
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

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)
