"""RL (Reinforcement Learning) training commands."""

import json
import re
import time
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

# ANSI escape code pattern
ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

# Progress bar pattern (tqdm-style progress bars)
PROGRESS_BAR = re.compile(r".*\|[█▏▎▍▌▋▊▉ ]{10,}\|.*")


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return ANSI_ESCAPE.sub("", text)


def filter_progress_bars(text: str) -> str:
    """Filter out progress bar updates, keeping only 100% completion lines.

    Progress bars from tqdm often appear as multiple updates on the same line
    (due to carriage return handling). This extracts just the final 100% part.
    """
    lines = text.splitlines()
    filtered = []
    for line in lines:
        # Check if line contains progress bars
        if PROGRESS_BAR.search(line) or re.search(r"\d+%\|", line):
            # If it has 100%, extract just that part
            if "100%" in line:
                # Find the last 100% progress bar and extract it
                # Pattern: text before + "100%|...bars...|" + stats after
                match = re.search(r"([^|]*100%\|[█▏▎▍▌▋▊▉ ]+\|[^\n]*?)(?=\d+%\||$)", line)
                if match:
                    filtered.append(match.group(1).strip())
                else:
                    # Fallback: just include the line
                    filtered.append(line)
            # Skip lines with only non-100% progress
            continue
        # Keep non-progress-bar lines, but skip empty lines
        if line.strip():
            filtered.append(line)
    return "\n".join(filtered)


def clean_logs(text: str) -> str:
    """Clean logs by stripping ANSI codes and filtering progress bars."""
    return filter_progress_bars(strip_ansi(text))


def generate_rl_config_template(environment: str | None = None) -> str:
    """Generate a TOML config template for RL training."""
    env_value = environment or "your-username/your-environment"

    return f'''\
model = "{DEFAULT_RL_MODEL}"
environments = ["{env_value}"]

rollouts = 8      # number of attempts per prompt/example
max_steps = 100   # total training iterations
seq_len = 4096    # max tokens per response
save_steps = 50   # checkpoint save interval

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


class EvalConfig(BaseModel):
    """Evaluation configuration."""

    environments: list[str] = Field(default_factory=list)
    interval: int | None = None
    num_examples: int | None = None
    rollouts_per_example: int | None = None
    base_model: bool | None = None  # whether to evaluate the base model before training


class RLRunConfig(BaseConfig):
    """Configuration for an RL training run."""

    model: str | None = None
    environments: list[str] = Field(default_factory=list)
    name: str | None = None
    rollouts: int = 8
    seq_len: int = 4096
    max_steps: int = 100
    save_steps: int | None = None
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    run_config: Optional[Dict[str, Any]] = Field(default=None)
    eval: EvalConfig = Field(default_factory=EvalConfig)


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
            console.print(f"[dim]Watching logs for run {run_id}... (Ctrl+C to stop)[/dim]\n")
            last_logs = ""
            consecutive_errors = 0

            while True:
                try:
                    logs = clean_logs(rl_client.get_logs(run_id, tail_lines=tail))
                    consecutive_errors = 0

                    if logs != last_logs:
                        old_lines = last_logs.splitlines() if last_logs else []
                        new_lines = logs.splitlines()

                        if not last_logs:
                            # First fetch, print everything
                            for line in new_lines:
                                console.print(line)
                        else:
                            # Find overlap between end of old_lines and start of new_lines
                            # This handles both growth and rotation cases
                            overlap = 0
                            max_overlap = min(len(old_lines), len(new_lines))
                            for i in range(1, max_overlap + 1):
                                if old_lines[-i:] == new_lines[:i]:
                                    overlap = i
                            # Print lines after the overlap
                            for line in new_lines[overlap:]:
                                console.print(line)

                        last_logs = logs
                except APIError as e:
                    consecutive_errors += 1
                    if "429" in str(e):
                        if consecutive_errors >= 3:
                            console.print("[yellow]Rate limited. Waiting 30s...[/yellow]")
                            time.sleep(30)
                        else:
                            time.sleep(10)
                        continue
                    raise

                time.sleep(5)  # Poll every 5 seconds to avoid rate limits
        else:
            logs = clean_logs(rl_client.get_logs(run_id, tail_lines=tail))
            if logs:
                console.print(logs)
            else:
                console.print("[yellow]No logs available yet.[/yellow]")

    except KeyboardInterrupt:
        console.print("\n[dim]Stopped watching logs.[/dim]")
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
    save_steps: Optional[int] = typer.Option(
        None, "--save-steps", help="Checkpoint save interval [default: 50]"
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
    eval_envs: Optional[List[str]] = typer.Option(
        None,
        "--eval-envs",
        help="Environments to evaluate on (e.g., 'owner/env-name')",
    ),
    eval_interval: Optional[int] = typer.Option(
        None,
        "--eval-interval",
        help="Evaluate every N training steps [default: 100]",
    ),
    eval_num_examples: Optional[int] = typer.Option(
        None,
        "--eval-num-examples",
        help="Number of examples per eval environment (-1 for all) [default: -1]",
    ),
    eval_rollouts: Optional[int] = typer.Option(
        None,
        "--eval-rollouts",
        help="Rollouts per example for evaluation [default: 1]",
    ),
    eval_base_model: Optional[bool] = typer.Option(
        None,
        "--eval-base-model/--no-eval-base-model",
        help="Evaluate base model before training [default: True]",
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
        save_steps=save_steps,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
        wandb_api_key=wandb_api_key,
        run_config=parsed_run_config,
        # Eval options (underscore prefix maps to nested eval.* fields)
        eval_environments=eval_envs or None,
        eval_interval=eval_interval,
        eval_num_examples=eval_num_examples,
        eval_rollouts_per_example=eval_rollouts,
        eval_base_model=eval_base_model,
    )

    # Build eval config for API from merged cfg.eval
    parsed_eval_config: Optional[Dict[str, Any]] = None
    has_eval_options = any(
        x is not None
        for x in [
            cfg.eval.interval,
            cfg.eval.num_examples,
            cfg.eval.rollouts_per_example,
            cfg.eval.base_model,
        ]
    )
    if has_eval_options and not cfg.eval.environments:
        console.print(
            "[yellow]Warning:[/yellow] Eval options require eval environments to take effect.\n"
            "  Use --eval-envs or set [eval] environments in config file."
        )
    if cfg.eval.environments:
        parsed_eval_config = {
            "environments": [{"id": env} for env in cfg.eval.environments],
        }
        if cfg.eval.interval is not None:
            parsed_eval_config["interval"] = cfg.eval.interval
        if cfg.eval.num_examples is not None:
            parsed_eval_config["num_examples"] = cfg.eval.num_examples
        if cfg.eval.rollouts_per_example is not None:
            parsed_eval_config["rollouts_per_example"] = cfg.eval.rollouts_per_example
        if cfg.eval.base_model is not None:
            parsed_eval_config["eval_base_model"] = cfg.eval.base_model

    # Validate required fields
    if not cfg.environments:
        console.print(
            "[red]Error:[/red] No environments specified. Provide via CLI or config file."
        )
        raise typer.Exit(1)

    # Validate environment slug format
    for env_slug in cfg.environments:
        if "/" not in env_slug:
            console.print(
                f"[red]Error:[/red] Invalid environment format: '{env_slug}'. "
                "Expected 'owner/name' format."
            )
            raise typer.Exit(1)

    # Validate eval environment slug format
    for env_slug in cfg.eval.environments:
        if "/" not in env_slug:
            console.print(
                f"[red]Error:[/red] Invalid eval environment format: '{env_slug}'. "
                "Expected 'owner/name' format."
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

        # Show configuration
        console.print("[bold]Configuration:[/bold]")
        if cfg.name:
            console.print(f"  Name: {cfg.name}")
        console.print(f"  Model: {cfg.model}")
        console.print(f"  Environments: {', '.join(cfg.environments)}")
        console.print(f"  Max Steps: {cfg.max_steps}")
        if cfg.save_steps:
            console.print(f"  Save Steps: {cfg.save_steps}")
        console.print(f"  Rollouts per Example: {cfg.rollouts}")
        console.print(f"  Sequence Length: {cfg.seq_len}")
        if cfg.wandb.project:
            console.print(f"  W&B Project: {cfg.wandb.project}")
        if app_config.team_id:
            console.print(f"  Team: {app_config.team_id}")
        if parsed_eval_config:
            eval_env_ids = [e["id"] for e in parsed_eval_config.get("environments", [])]
            console.print(f"  Eval Environments: {', '.join(eval_env_ids)}")
            if "interval" in parsed_eval_config:
                console.print(f"  Eval Interval: {parsed_eval_config['interval']}")
        console.print()

        # Create the run
        run = rl_client.create_run(
            model_name=cfg.model,
            environments=[{"id": slug} for slug in cfg.environments],
            rollouts_per_example=cfg.rollouts,
            seq_len=cfg.seq_len,
            max_steps=cfg.max_steps,
            save_steps=cfg.save_steps,
            name=cfg.name,
            wandb_entity=cfg.wandb.entity,
            wandb_project=cfg.wandb.project,
            wandb_run_name=cfg.wandb.name,
            wandb_api_key=cfg.wandb.api_key,
            team_id=app_config.team_id,
            run_config=cfg.run_config,
            eval_config=parsed_eval_config,
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
