"""Deployments command for managing model deployments for inference."""

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ..api.deployments import DeploymentsClient
from ..client import APIClient, APIError
from ..core import Config
from ..utils import output_data_as_json, validate_output_format

console = Console()

app = typer.Typer(
    help="Manage adapter deployments.",
    no_args_is_help=True,
)


@app.command(name="list")
def list_deployments(
    team: Optional[str] = typer.Option(None, "--team", "-t", help="Filter by team ID"),
    output: str = typer.Option("table", "-o", "--output", help="Output format: table or json"),
) -> None:
    """List adapters and their deployment status.

    Example:

        prime deployments list

        prime deployments list -o json

        prime deployments list --team <team_id>
    """
    validate_output_format(output, console)

    try:
        api_client = APIClient()
        deployments_client = DeploymentsClient(api_client)
        config = Config()

        team_id = team or config.team_id

        models = deployments_client.list_adapters(team_id=team_id)

        if output == "json":
            output_data_as_json(
                {"models": [m.model_dump() for m in models]},
                console,
            )
            return

        if not models:
            console.print("[yellow]No models found.[/yellow]")
            console.print("[dim]Train a model first, then deploy it here.[/dim]")
            return

        table = Table(title="Adapter Deployments")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Name", style="white")
        table.add_column("Base Model", style="magenta")
        table.add_column("Status", style="white")
        table.add_column("Deployed At", style="dim")

        from ..utils.display import DEPLOYMENT_STATUS_COLORS

        for model in models:
            deployed_at = model.deployed_at.strftime("%Y-%m-%d %H:%M") if model.deployed_at else "-"
            base_model = model.base_model[:30] if len(model.base_model) > 30 else model.base_model
            status_color = DEPLOYMENT_STATUS_COLORS.get(model.deployment_status, "white")
            status = f"[{status_color}]{model.deployment_status}[/{status_color}]"

            table.add_row(
                model.id,
                model.display_name or "-",
                base_model,
                status,
                deployed_at,
            )

        console.print(table)
        console.print(f"\n[dim]Total: {len(models)} model(s)[/dim]")

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command(name="create")
def create_deployment(
    ctx: typer.Context,
    model_id: Optional[str] = typer.Argument(None, help="Model ID to deploy"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Deploy a model for inference.

    Makes the trained model available for inference requests.
    Model must be in READY status.

    Example:

        prime deployments create <model_id>

        prime deployments create <model_id> --yes
    """
    if model_id is None:
        console.print(ctx.get_help())
        raise typer.Exit(0)

    try:
        api_client = APIClient()
        deployments_client = DeploymentsClient(api_client)

        # Get model to validate status
        model = deployments_client.get_adapter(model_id)

        if model.status != "READY":
            console.print("[red]Error:[/red] Model is not ready for deployment.")
            console.print(f"Current status: [yellow]{model.status}[/yellow]")
            console.print("[dim]Only models with READY status can be deployed.[/dim]")
            raise typer.Exit(1)

        if model.deployment_status == "DEPLOYED":
            console.print("[yellow]Model is already deployed.[/yellow]")
            raise typer.Exit(0)

        if model.deployment_status in ("DEPLOYING", "UNLOADING"):
            console.print("[yellow]Model deployment is in progress.[/yellow]")
            console.print(f"Current status: {model.deployment_status}")
            raise typer.Exit(1)

        # Show model details and confirm
        console.print("[bold]Deploying model:[/bold]")
        console.print(f"  ID: {model.id}")
        if model.display_name:
            console.print(f"  Name: {model.display_name}")
        console.print(f"  Base Model: {model.base_model}")
        console.print()

        if not yes:
            confirm = typer.confirm("Are you sure you want to deploy this model?")
            if not confirm:
                console.print("Cancelled.")
                raise typer.Exit(0)

        # Deploy the model
        updated_model = deployments_client.deploy_adapter(model_id)

        console.print("[green]Deployment initiated successfully![/green]")
        console.print(f"Status: [yellow]{updated_model.deployment_status}[/yellow]")
        console.print("\n[dim]The model is being deployed. This may take a few minutes.[/dim]")
        console.print("[dim]Use 'prime deployments list' to check deployment status.[/dim]")

        console.print("\n[bold]Once deployed, you can run inference with:[/bold]")
        console.print(
            f"""
[dim]curl -X POST https://api.pinference.ai/api/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $PRIME_API_KEY" \\
  -d '{{
    "model": "{model.base_model}:{model.id}",
    "messages": [{{"role": "user", "content": "Hello"}}],
    "max_tokens": 100
  }}'[/dim]"""
        )

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command(name="delete")
def delete_deployment(
    ctx: typer.Context,
    model_id: Optional[str] = typer.Argument(None, help="Model ID to unload"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Unload a model from inference.

    Removes the model from serving. Model files remain stored
    and can be deployed again.

    Example:

        prime deployments delete <model_id>

        prime deployments delete <model_id> --yes
    """
    if model_id is None:
        console.print(ctx.get_help())
        raise typer.Exit(0)

    try:
        api_client = APIClient()
        deployments_client = DeploymentsClient(api_client)

        # Get model to validate status
        model = deployments_client.get_adapter(model_id)

        if model.deployment_status == "NOT_DEPLOYED":
            console.print("[yellow]Model is not deployed.[/yellow]")
            raise typer.Exit(0)

        if model.deployment_status in ("DEPLOYING", "UNLOADING"):
            console.print("[yellow]Model deployment is in progress.[/yellow]")
            console.print(f"Current status: {model.deployment_status}")
            raise typer.Exit(1)

        if model.deployment_status not in ("DEPLOYED", "DEPLOY_FAILED", "UNLOAD_FAILED"):
            console.print("[red]Error:[/red] Cannot unload model in current state.")
            console.print(f"Current status: {model.deployment_status}")
            raise typer.Exit(1)

        # Show model details and confirm
        console.print("[bold]Unloading model:[/bold]")
        console.print(f"  ID: {model.id}")
        if model.display_name:
            console.print(f"  Name: {model.display_name}")
        console.print(f"  Base Model: {model.base_model}")
        console.print()

        if not yes:
            confirm = typer.confirm("Are you sure you want to unload this model?")
            if not confirm:
                console.print("Cancelled.")
                raise typer.Exit(0)

        # Unload the model
        updated_model = deployments_client.unload_adapter(model_id)

        console.print("[green]Unload initiated successfully![/green]")
        console.print(f"Status: [yellow]{updated_model.deployment_status}[/yellow]")
        console.print("\n[dim]The model is being unloaded.[/dim]")
        console.print("[dim]Use 'prime deployments list' to check status.[/dim]")

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
