"""RL (Reinforcement Learning) training commands."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table
from typer.core import TyperGroup

from prime_cli.core import Config

from ..api.rl import RLClient, RLRun
from ..client import APIClient, APIError
from ..utils import BaseConfig, output_data_as_json, validate_output_format
from ..utils.env_metadata import find_environment_metadata

console = Console()

# Default model for RL training
DEFAULT_RL_MODEL = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"


def generate_rl_config_template(environment: str | None = None) -> str:
    """Generate a TOML config template for RL training."""
    env_value = environment or "your-username/your-environment"

    return f'''\
model = "{DEFAULT_RL_MODEL}"
environments = ["{env_value}"]

rollouts = 8      # number of attempts per prompt/example
max_steps = 100   # total training iterations
seq_len = 4096    # max tokens per response

# name = "my-experiment"

# [wandb]
# entity = "my-team"
# project = "my-project"
# name = "experiment-1"
'''


class WandbConfig(BaseModel):
    """Weights & Biases configuration."""

    entity: str | None = None
    project: str | None = None
    name: str | None = None
    api_key: str | None = None


class RLRunConfig(BaseConfig):
    """Configuration for an RL training run."""

    model: str | None = None
    environments: list[str] = Field(default_factory=list)
    name: str | None = None
    rollouts: int = 8
    seq_len: int = 4096
    max_steps: int = 100
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    run_config: Optional[Dict[str, Any]] = Field(default=None)


class DefaultGroup(TyperGroup):
    def __init__(self, *args, default_cmd_name: str = "run", **kwargs):
        super().__init__(*args, **kwargs)
        self.default_cmd_name = default_cmd_name

    def parse_args(self, ctx, args):
        if not args:
            return super().parse_args(ctx, args)

        if args[0] in ("--help", "-h"):
            return super().parse_args(ctx, args)

        if args[0] in self.commands:
            return super().parse_args(ctx, args)

        args = [self.default_cmd_name] + list(args)
        return super().parse_args(ctx, args)

    def format_usage(self, ctx, formatter):
        formatter.write_usage(
            ctx.command_path,
            "[OPTIONS] ENVIRONMENTS... | COMMAND [ARGS]...",
        )


subcommands_app = typer.Typer()

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


def _format_run_for_display(run: RLRun) -> Dict[str, Any]:
    """Format run data for display (both table and JSON)."""
    created_at = run.created_at.strftime("%Y-%m-%d %H:%M") if run.created_at else ""
    env_names = [
        env.get("slug") or env.get("name") or env.get("id") or "?" for env in run.environments
    ]
    envs_display = ", ".join(env_names[:3])
    if len(env_names) > 3:
        envs_display += f" (+{len(env_names) - 3})"

    return {
        "id": run.id,
        "status": run.status,
        "model": run.base_model,
        "environments": envs_display,
        "steps": f"{run.max_steps}",
        "rollouts": str(run.rollouts_per_example),
        "created_at": created_at,
        "team_id": run.team_id,
    }


@subcommands_app.command("models")
def list_models(
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List available models for RL training."""
    validate_output_format(output, console)

    try:
        api_client = APIClient()
        rl_client = RLClient(api_client)

        models = rl_client.list_models()

        if output == "json":
            output_data_as_json({"models": [m.model_dump() for m in models]}, console)
            return

        if not models:
            console.print("[yellow]No models available for RL training.[/yellow]")
            console.print("[dim]This could mean no healthy RL clusters are running.[/dim]")
            return

        table = Table(title="Prime RL — Models")
        table.add_column("id", style="cyan")

        for model in models:
            table.add_row(model.name)

        console.print(table)

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@subcommands_app.command("list")
def list_runs(
    team: Optional[str] = typer.Option(None, "--team", "-t", help="Filter by team ID"),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List your RL training runs."""
    validate_output_format(output, console)

    try:
        api_client = APIClient()
        rl_client = RLClient(api_client)
        config = Config()

        # Use provided team or default from config
        team_id = team or config.team_id

        runs = rl_client.list_runs(team_id=team_id)

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
                formatted["id"],
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


@subcommands_app.command("stop")
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
        rl_client = RLClient(api_client)

        run = rl_client.stop_run(run_id)

        console.print(f"[green]✓ Run {run_id} stopped successfully[/green]")
        console.print(f"Status: {run.status}")

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@subcommands_app.command("delete")
def delete_run(
    run_id: str = typer.Argument(..., help="Run ID to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Delete an RL training run."""
    try:
        if not force:
            confirm = typer.confirm(f"Are you sure you want to permanently delete run {run_id}?")
            if not confirm:
                console.print("Cancelled.")
                raise typer.Exit(0)

        api_client = APIClient()
        rl_client = RLClient(api_client)

        rl_client.delete_run(run_id)
        console.print(f"[green]✓ Run {run_id} deleted successfully[/green]")

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@subcommands_app.command("logs")
def get_logs(
    run_id: str = typer.Argument(..., help="Run ID to get logs for"),
    tail: int = typer.Option(1000, "--tail", "-n", help="Number of lines to show"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
) -> None:
    """Get logs for an RL training run."""
    try:
        api_client = APIClient()
        rl_client = RLClient(api_client)

        if follow:
            import time

            seen_lines = 0
            while True:
                logs = rl_client.get_logs(run_id, tail_lines=tail)
                lines = logs.splitlines()
                if len(lines) > seen_lines:
                    for line in lines[seen_lines:]:
                        console.print(line)
                    seen_lines = len(lines)
                time.sleep(2)
        else:
            logs = rl_client.get_logs(run_id, tail_lines=tail)
            if logs:
                console.print(logs)
            else:
                console.print("[yellow]No logs available yet.[/yellow]")

    except KeyboardInterrupt:
        pass
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@subcommands_app.command("init")
def init_config(
    output: str = typer.Argument(
        "configs/rl.toml",
        help="Output path for the config file",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing file"),
) -> None:
    """Generate a template TOML config file for RL training.

    Auto-detects the environment if run inside an environment directory
    (looks for .prime/.env-metadata.json).

    Example:

        prime rl init                     # Creates configs/rl.toml

        prime rl init my-experiment.toml  # Custom path

        prime rl init -f                  # Overwrite existing
    """
    output_path = Path(output)

    # Check if file exists
    if output_path.exists() and not force:
        console.print(f"[red]Error:[/red] {output} already exists. Use --force to overwrite.")
        raise typer.Exit(1)

    # Try to auto-detect environment from .env-metadata.json
    environment: str | None = None
    metadata = find_environment_metadata()
    if metadata:
        owner = metadata.get("owner")
        name = metadata.get("name")
        if owner and name:
            environment = f"{owner}/{name}"
            console.print(f"[dim]Detected environment: {environment}[/dim]")

    # Create parent directories if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write template
    template = generate_rl_config_template(environment)
    output_path.write_text(template)

    console.print(f"[green]✓[/green] Created {output}")
    console.print(f"\n[dim]Run with:[/dim] prime rl -c {output}")


app = typer.Typer(
    cls=DefaultGroup,
    help=(
        "Manage hosted RL training runs.\n\n"
        "By default, 'prime rl <environments>' runs 'prime rl run <environments>'."
    ),
    no_args_is_help=True,
)

app.add_typer(subcommands_app, name="")


@app.command("run", help="Create and start an RL training run [default]")
def create_run(
    ctx: typer.Context,
    environments: Optional[List[str]] = typer.Argument(
        None,
        help="Environment slugs to train on (e.g., 'owner/env-name')",
    ),
    model: Optional[str] = typer.Option(None, "-m", "--model", help="Model to fine-tune"),
    name: Optional[str] = typer.Option(
        None, "-n", "--name", help="Run name (auto-generated if not provided)"
    ),
    rollouts: Optional[int] = typer.Option(
        None, "-r", "--rollouts", help="Number of rollouts per example [default: 8]"
    ),
    seq_len: Optional[int] = typer.Option(
        None, "-s", "--seq-len", help="Sequence length [default: 4096]"
    ),
    max_steps: Optional[int] = typer.Option(
        None, "--max-steps", help="Maximum training steps [default: 100]"
    ),
    wandb_entity: Optional[str] = typer.Option(
        None, "--wandb-entity", help="Weights & Biases entity (username or team name)"
    ),
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
    config_file: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to TOML config file (CLI options override config file values)",
    ),
    run_config: Optional[str] = typer.Option(
        None,
        "--run-config",
        hidden=True,
        help='Additional run configuration as JSON (admin only), e.g. \'{"key": "value"}\'',
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Configuration can be provided via CLI options, a TOML config file, or both.
    CLI options take precedence over config file values.

    Example TOML config (rl-config.toml):

        model = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"
        environments = ["primeintellect/gpqa"]
        rollouts = 16
        max_steps = 200

        [wandb]
        project = "my-project"

    Example usage:

        prime rl run owner/env1 owner/env2 -m model-name

        prime rl --config rl-config.toml

        prime rl --config rl-config.toml --max-steps 500
    """
    # Show help if no meaningful input provided
    if not environments and not config_file and not model:
        console.print(ctx.get_help())
        raise typer.Exit(0)

    validate_output_format(output, console)

    parsed_run_config: Optional[Dict[str, Any]] = None
    if run_config:
        try:
            parsed_run_config = json.loads(run_config)
        except json.JSONDecodeError as e:
            console.print(
                f"[red]Error:[/red] Invalid JSON in --run-config: {e}\n"
                '  Expected format: --run-config \'{"key": "value"}\''
            )
            raise typer.Exit(1)

    # Load and merge config: CLI > TOML > defaults
    if config_file:
        console.print(f"[dim]Loading config from {config_file}[/dim]\n")

    cfg = RLRunConfig.from_sources(
        toml_path=config_file,
        console=console,
        # Pass CLI args (None values are ignored)
        model=model,
        environments=environments or None,  # Convert empty list to None
        name=name,
        rollouts=rollouts,
        seq_len=seq_len,
        max_steps=max_steps,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
        wandb_api_key=wandb_api_key,
        run_config=parsed_run_config,
    )

    # Validate required fields
    if not cfg.environments:
        console.print(
            "[red]Error:[/red] No environments specified. Provide via CLI or config file."
        )
        raise typer.Exit(1)

    if not cfg.model:
        console.print("[red]Error:[/red] No model specified. Use --model or set 'model' in config.")
        raise typer.Exit(1)

    # Warn if wandb is configured but no API key is provided
    if (cfg.wandb.entity or cfg.wandb.project) and not cfg.wandb.api_key:
        console.print(
            "[yellow]Warning:[/yellow] W&B config detected but no API key provided.\n"
            "  Set via: --wandb-api-key or WANDB_API_KEY env var\n"
        )

    try:
        api_client = APIClient()
        rl_client = RLClient(api_client)
        app_config = Config()

        console.print("[bold]Creating RL training run...[/bold]\n")

        # Validate environment slug format
        for env_slug in cfg.environments:
            if "/" not in env_slug:
                console.print(
                    f"[red]Error:[/red] Invalid environment format: '{env_slug}'. "
                    "Expected 'owner/name' format."
                )
                raise typer.Exit(1)

        # Show configuration
        console.print("[bold]Configuration:[/bold]")
        if cfg.name:
            console.print(f"  Name: {cfg.name}")
        console.print(f"  Model: {cfg.model}")
        console.print(f"  Environments: {', '.join(cfg.environments)}")
        console.print(f"  Max Steps: {cfg.max_steps}")
        console.print(f"  Rollouts per Example: {cfg.rollouts}")
        console.print(f"  Sequence Length: {cfg.seq_len}")
        if cfg.wandb.project:
            console.print(f"  W&B Project: {cfg.wandb.project}")
        if app_config.team_id:
            console.print(f"  Team: {app_config.team_id}")
        console.print()

        # Create the run
        run = rl_client.create_run(
            model_name=cfg.model,
            environments=[{"id": slug} for slug in cfg.environments],
            rollouts_per_example=cfg.rollouts,
            seq_len=cfg.seq_len,
            max_steps=cfg.max_steps,
            name=cfg.name,
            wandb_entity=cfg.wandb.entity,
            wandb_project=cfg.wandb.project,
            wandb_run_name=cfg.wandb.name,
            wandb_api_key=cfg.wandb.api_key,
            team_id=app_config.team_id,
            run_config=cfg.run_config,
        )

        if output == "json":
            output_data_as_json({"run": run.model_dump()}, console)
            return

        console.print("[green]✓ Run created successfully![/green]")

        # Show dashboard link
        dashboard_url = f"{app_config.frontend_url}/dashboard/training/{run.id}"
        console.print("\n[cyan]Monitor run at:[/cyan]")
        console.print(f"  [link={dashboard_url}]{dashboard_url}[/link]")

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
