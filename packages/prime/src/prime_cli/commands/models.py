"""Models command for listing trained models from RL runs."""

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ..api.deployments import DeploymentsClient
from ..client import APIClient, APIError
from ..core import Config
from ..utils import output_data_as_json, validate_output_format
from ..utils.display import ADAPTER_STATUS_COLORS, DEPLOYMENT_STATUS_COLORS, status_color

console = Console()

app = typer.Typer(
    help="List trained models from RL runs.",
    no_args_is_help=True,
)


@app.command(name="list")
def list_models(
    team: Optional[str] = typer.Option(None, "--team", "-t", help="Filter by team ID"),
    output: str = typer.Option("table", "-o", "--output", help="Output format: table or json"),
) -> None:
    """List trained models from RL runs.

    Shows all trained models created from completed RL training runs.

    Example:

        prime models list

        prime models list -o json

        prime models list --team <team_id>
    """
    validate_output_format(output, console)

    try:
        api_client = APIClient()
        deployments_client = DeploymentsClient(api_client)
        config = Config()

        team_id = team or config.team_id

        models = deployments_client.list_adapters(team_id=team_id)

        if output == "json":
            output_data_as_json({"models": [m.model_dump() for m in models]}, console)
            return

        if not models:
            console.print("[yellow]No models found.[/yellow]")
            console.print("[dim]Models are created from completed RL training runs.[/dim]")
            return

        table = Table(title="Trained Models")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Name", style="white")
        table.add_column("Base Model", style="magenta")
        table.add_column("Status", style="bold")
        table.add_column("Deployment", style="bold")
        table.add_column("Created", style="dim")

        for model in models:
            created_at = model.created_at.strftime("%Y-%m-%d %H:%M") if model.created_at else ""
            status_clr = status_color(model.status, ADAPTER_STATUS_COLORS)
            deployment_clr = status_color(model.deployment_status, DEPLOYMENT_STATUS_COLORS)
            base_model = model.base_model[:30] if len(model.base_model) > 30 else model.base_model

            table.add_row(
                model.id,
                model.display_name or "-",
                base_model,
                f"[{status_clr}]{model.status}[/{status_clr}]",
                f"[{deployment_clr}]{model.deployment_status}[/{deployment_clr}]",
                created_at,
            )

        console.print(table)
        console.print(f"\n[dim]Total: {len(models)} model(s)[/dim]")
        console.print("[dim]Deploy with: prime deployments create <model_id>[/dim]")

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
