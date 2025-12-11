"""RL (Reinforcement Learning) training commands."""

from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.table import Table

from prime_cli.core import Config

from ..api.rft import RFTClient, RFTRun
from ..client import APIClient, APIError
from ..utils import output_data_as_json, validate_output_format

app = typer.Typer(help="Manage RL training runs", no_args_is_help=True)
console = Console()

# Status color mapping
RUN_STATUS_COLORS = {
    "PENDING": "yellow",
    "RUNNING": "green",
    "COMPLETED": "cyan",
    "FAILED": "red",
    "STOPPED": "magenta",
}


def _get_status_color(status: str) -> str:
    """Get color for run status."""
    return RUN_STATUS_COLORS.get(status.upper(), "white")


def _format_run_for_display(run: RFTRun) -> Dict[str, Any]:
    """Format run data for display (both table and JSON)."""
    created_at = run.created_at.strftime("%Y-%m-%d %H:%M") if run.created_at else ""
    env_names = [env.get("name", env.get("id", "?")) for env in run.environments]
    envs_display = ", ".join(env_names[:3])
    if len(env_names) > 3:
        envs_display += f" (+{len(env_names) - 3})"

    return {
        "id": run.id,
        "status": run.status,
        "model": run.model_name,
        "environments": envs_display,
        "steps": f"{run.max_steps}",
        "rollouts": str(run.rollouts_per_example),
        "created_at": created_at,
        "team_id": run.team_id,
    }


def _resolve_environment(client: APIClient, env_slug: str) -> Dict[str, Any]:
    """Resolve an environment slug (owner/name) to its ID and metadata."""
    if "/" not in env_slug:
        raise ValueError(
            f"Invalid environment format: '{env_slug}'. Expected 'owner/name' format."
        )

    owner, name = env_slug.split("/", 1)

    try:
        response = client.get(f"/environmentshub/{owner}/{name}/@latest")
        data = response.get("data", response)
        return {
            "id": data.get("id"),
            "name": name,
            "args": {},
        }
    except APIError as e:
        raise APIError(f"Failed to resolve environment '{env_slug}': {e}")


@app.command("models")
def list_models(
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List available models for RL training."""
    validate_output_format(output, console)

    try:
        api_client = APIClient()
        rft_client = RFTClient(api_client)

        models = rft_client.list_models()

        if output == "json":
            output_data_as_json({"models": [m.model_dump() for m in models]}, console)
            return

        if not models:
            console.print("[yellow]No models available for RL training.[/yellow]")
            console.print(
                "[dim]This could mean no healthy RFT clusters are running.[/dim]"
            )
            return

        table = Table(title="Prime RL — Models")
        table.add_column("id", style="cyan")

        for model in models:
            table.add_row(model.name)

        console.print(table)

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("runs")
def list_runs(
    team: Optional[str] = typer.Option(None, "--team", "-t", help="Filter by team ID"),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List your RL training runs."""
    validate_output_format(output, console)

    try:
        api_client = APIClient()
        rft_client = RFTClient(api_client)
        config = Config()

        # Use provided team or default from config
        team_id = team or config.team_id

        runs = rft_client.list_runs(team_id=team_id)

        if output == "json":
            output_data_as_json({"runs": [r.model_dump() for r in runs]}, console)
            return

        if not runs:
            console.print("[yellow]No RL training runs found.[/yellow]")
            return

        table = Table(title="RL Training Runs")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Status", style="bold")
        table.add_column("Model", style="magenta")
        table.add_column("Environments", style="green")
        table.add_column("Steps", justify="right")
        table.add_column("Created", style="dim")

        for run in runs:
            formatted = _format_run_for_display(run)
            status_color = _get_status_color(run.status)
            table.add_row(
                formatted["id"][:12] + "...",
                f"[{status_color}]{formatted['status']}[/{status_color}]",
                formatted["model"][:30],
                formatted["environments"],
                formatted["steps"],
                formatted["created_at"],
            )

        console.print(table)
        console.print(f"\n[dim]Total: {len(runs)} run(s)[/dim]")

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("stop")
def stop_run(
    run_id: str = typer.Argument(..., help="Run ID to stop"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Stop an RL training run."""
    try:
        if not force:
            confirm = typer.confirm(f"Are you sure you want to stop run {run_id}?")
            if not confirm:
                console.print("Cancelled.")
                raise typer.Exit(0)

        api_client = APIClient()
        rft_client = RFTClient(api_client)

        run = rft_client.stop_run(run_id)

        console.print(f"[green]✓ Run {run_id} stopped successfully[/green]")
        console.print(f"Status: {run.status}")

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("delete")
def delete_run(
    run_id: str = typer.Argument(..., help="Run ID to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Delete an RL training run."""
    try:
        if not force:
            confirm = typer.confirm(
                f"Are you sure you want to permanently delete run {run_id}?"
            )
            if not confirm:
                console.print("Cancelled.")
                raise typer.Exit(0)

        api_client = APIClient()
        rft_client = RFTClient(api_client)

        success = rft_client.delete_run(run_id)

        if success:
            console.print(f"[green]✓ Run {run_id} deleted successfully[/green]")
        else:
            console.print(f"[red]Failed to delete run {run_id}[/red]")
            raise typer.Exit(1)

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("run", no_args_is_help=True)
def create_run(

    environments: List[str] = typer.Argument(
        ...,
        help="Environment slugs to train on (e.g., 'owner/env-name')",
    ),
    model: str = typer.Option(
        ..., "-m", "--model", help="Model to fine-tune"
    ),
    rollouts: int = typer.Option(
        8, "-r", "--rollouts", help="Number of rollouts per example"
    ),
    seq_len: int = typer.Option(4096, "-s", "--seq-len", help="Sequence length"),
    max_steps: int = typer.Option(100, "--max-steps", help="Maximum training steps"),
    wandb_project: Optional[str] = typer.Option(
        None, "--wandb-project", help="Weights & Biases project name"
    ),
    wandb_name: Optional[str] = typer.Option(
        None, "--wandb-name", help="Weights & Biases run name"
    ),
    wandb_api_key: Optional[str] = typer.Option(
        None,
        "--wandb-api-key",
        help="Weights & Biases API key (or set WANDB_API_KEY env var)",
        envvar="WANDB_API_KEY",
    ),
    team: Optional[str] = typer.Option(
        None, "-t", "--team", help="Team ID for team-owned run"
    ),
    output: str = typer.Option(
        "table", "--output", "-o", help="Output format: table or json"
    ),
) -> None:
    """Create an RL training run with specified environments and model.

    Example usage:

        prime rl run owner/env1 owner/env2 -m deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

        prime rl run primeintellect/gpqa -m model-name --max-steps 200 --rollouts 16
    """


    validate_output_format(output, console)

    try:
        api_client = APIClient()
        rft_client = RFTClient(api_client)
        config = Config()

        # Use provided team or default from config
        team_id = team or config.team_id

        console.print("[bold]Creating RL training run...[/bold]\n")

        # Resolve environments
        console.print("[dim]Resolving environments...[/dim]")
        resolved_envs = []
        for env_slug in environments:
            try:
                env_data = _resolve_environment(api_client, env_slug)
                resolved_envs.append(env_data)
                console.print(f"  [green]✓[/green] {env_slug}")
            except (APIError, ValueError) as e:
                console.print(f"  [red]✗[/red] {env_slug}: {e}")
                raise typer.Exit(1)

        console.print()

        # Show configuration
        console.print("[bold]Configuration:[/bold]")
        console.print(f"  Model: {model}")
        console.print(f"  Environments: {', '.join(environments)}")
        console.print(f"  Max Steps: {max_steps}")
        console.print(f"  Rollouts per Example: {rollouts}")
        console.print(f"  Sequence Length: {seq_len}")
        if wandb_project:
            console.print(f"  W&B Project: {wandb_project}")
        if team_id:
            console.print(f"  Team: {team_id}")
        console.print()

        # Create the run
        run = rft_client.create_run(
            model_name=model,
            environments=resolved_envs,
            rollouts_per_example=rollouts,
            seq_len=seq_len,
            max_steps=max_steps,
            wandb_project=wandb_project,
            wandb_run_name=wandb_name,
            wandb_api_key=wandb_api_key,
            team_id=team_id,
        )

        if output == "json":
            output_data_as_json({"run": run.model_dump()}, console)
            return

        console.print("[green]✓ Run created successfully![/green]")
        console.print(f"\n[bold]Run ID:[/bold] {run.id}")
        console.print(f"[bold]Status:[/bold] {run.status}")

        console.print("\n[dim]View your runs with:[/dim]")
        console.print("  prime rl runs")

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
