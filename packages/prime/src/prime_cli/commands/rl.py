"""RL (Reinforcement Learning) training commands."""

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import toml
import typer
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from prime_cli.core import Config

from ..api.rl import RLClient, RLRun
from ..client import APIClient, APIError
from ..utils import output_data_as_json, validate_output_format
from ..utils.env_metadata import find_environment_metadata

console = Console()

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
        if PROGRESS_BAR.search(line) or re.search(r"\d+%\|", line):
            if "100%" in line:
                match = re.search(r"([^|]*100%\|[█▏▎▍▌▋▊▉ ]+\|[^\n]*?)(?=\d+%\||$)", line)
                if match:
                    filtered.append(match.group(1).strip())
                else:
                    filtered.append(line)
            continue
        if line.strip():
            filtered.append(line)
    return "\n".join(filtered)


def clean_logs(text: str) -> str:
    """Clean logs by stripping ANSI codes and filtering progress bars."""
    return filter_progress_bars(strip_ansi(text))


def generate_rl_config_template(environment: str | None = None) -> str:
    """Generate a TOML config template for RL training."""
    env_value = environment or "primeintellect/your-environment"

    return f'''\
model = "meta-llama/Llama-3.1-8B-Instruct"
max_steps = 100

# Training
batch_size = 128
rollouts_per_example = 8
# trajectory_strategy = "interleaved"  # or "branching"

[sampling]
max_tokens = 2048

[[env]]
id = "{env_value}"

# [[env]]
# id = "primeintellect/another-env"
# args = {{ split = "train", max_examples = 1000 }}

# Optional: W&B logging
# [wandb]
# project = "my-project"
# entity = "my-team"

# Optional: online evaluation
# [eval]
# interval = 100
#
# [[eval.env]]
# id = "primeintellect/eval-env"
# num_examples = 30
# rollouts_per_example = 4
'''


class EnvConfig(BaseModel):
    id: str
    name: str | None = None
    args: Dict[str, Any] = Field(default_factory=dict)


class EvalEnvConfig(BaseModel):
    id: str
    num_examples: int | None = None
    rollouts_per_example: int | None = None


class SamplingConfig(BaseModel):
    max_tokens: int | None = None


class EvalConfig(BaseModel):
    interval: int | None = None
    env: List[EvalEnvConfig] = Field(default_factory=list)


class WandbConfig(BaseModel):
    entity: str | None = None
    project: str | None = None
    name: str | None = None


class RLConfig(BaseModel):
    model: str | None = None
    max_steps: int = 100
    batch_size: int = 128
    rollouts_per_example: int = 8
    trajectory_strategy: str | None = None
    env: List[EnvConfig] = Field(default_factory=list)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)


def load_config(path: str) -> RLConfig:
    """Load config from TOML file."""
    p = Path(path)
    if not p.exists():
        console.print(f"[red]Error:[/red] Config file not found: {path}")
        raise typer.Exit(1)
    try:
        data = toml.load(p)
        return RLConfig.model_validate(data)
    except toml.TomlDecodeError as e:
        console.print(f"[red]Error:[/red] Invalid TOML in {path}: {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] Invalid config: {e}")
        raise typer.Exit(1)


# Status color mapping
RUN_STATUS_COLORS = {
    "PENDING": "yellow",
    "RUNNING": "green",
    "COMPLETED": "cyan",
    "FAILED": "red",
    "STOPPED": "magenta",
}


def _get_status_color(status: str) -> str:
    return RUN_STATUS_COLORS.get(status.upper(), "white")


def _format_run_for_display(run: RLRun) -> Dict[str, Any]:
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


app = typer.Typer(
    help="Manage hosted RL training runs.",
    no_args_is_help=True,
)


@app.command("run")
def create_run(
    config_path: str = typer.Argument(
        ...,
        help="Path to TOML config file (e.g., @ rl.toml)",
    ),
    wandb_api_key: Optional[str] = typer.Option(
        None,
        "--wandb-api-key",
        help="Weights & Biases API key (or set WANDB_API_KEY env var)",
        envvar="WANDB_API_KEY",
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Start an RL training run from a config file.

    Example:

        prime rl run @ rl.toml
    """
    validate_output_format(output, console)

    # Handle @ prefix
    path = config_path[1:].strip() if config_path.startswith("@") else config_path

    console.print(f"[dim]Loading config from {path}[/dim]\n")
    cfg = load_config(path)

    # Validate required fields
    if not cfg.env:
        console.print("[red]Error:[/red] No environments specified. Add [[env]] sections.")
        raise typer.Exit(1)

    for env in cfg.env:
        if "/" not in env.id:
            console.print(
                f"[red]Error:[/red] Invalid environment format: '{env.id}'. "
                "Expected 'owner/name' format."
            )
            raise typer.Exit(1)

    # Validate eval environment IDs
    for env in cfg.eval.env:
        if "/" not in env.id:
            console.print(
                f"[red]Error:[/red] Invalid eval environment format: '{env.id}'. "
                "Expected 'owner/name' format."
            )
            raise typer.Exit(1)

    if not cfg.model:
        console.print("[red]Error:[/red] No model specified.")
        raise typer.Exit(1)

    # Warn if wandb is configured but no API key
    if (cfg.wandb.entity or cfg.wandb.project) and not wandb_api_key:
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
        console.print(f"  Model: {cfg.model}")
        console.print(f"  Environments: {', '.join(e.id for e in cfg.env)}")
        console.print(f"  Max Steps: {cfg.max_steps}")
        console.print(f"  Batch Size: {cfg.batch_size}")
        console.print(f"  Rollouts per Example: {cfg.rollouts_per_example}")
        if cfg.sampling.max_tokens:
            console.print(f"  Max Tokens: {cfg.sampling.max_tokens}")
        if cfg.wandb.project:
            console.print(f"  W&B Project: {cfg.wandb.project}")
        if cfg.eval.env:
            console.print(f"  Eval Environments: {', '.join(e.id for e in cfg.eval.env)}")
        if app_config.team_id:
            console.print(f"  Team: {app_config.team_id}")
        console.print()

        # Build eval config if provided
        eval_config = None
        if cfg.eval.env:
            eval_environments = []
            for e in cfg.eval.env:
                env_cfg: Dict[str, Any] = {"id": e.id}
                if e.num_examples is not None:
                    env_cfg["num_examples"] = e.num_examples
                if e.rollouts_per_example is not None:
                    env_cfg["rollouts_per_example"] = e.rollouts_per_example
                eval_environments.append(env_cfg)
            eval_config = {"environments": eval_environments}
            if cfg.eval.interval is not None:
                eval_config["interval"] = cfg.eval.interval

        # Create the run
        run = rl_client.create_run(
            model_name=cfg.model,
            environments=[{"id": e.id, "name": e.name, "args": e.args} for e in cfg.env],
            rollouts_per_example=cfg.rollouts_per_example,
            max_steps=cfg.max_steps,
            max_tokens=cfg.sampling.max_tokens,
            batch_size=cfg.batch_size,
            trajectory_strategy=cfg.trajectory_strategy,
            wandb_entity=cfg.wandb.entity,
            wandb_project=cfg.wandb.project,
            wandb_run_name=cfg.wandb.name,
            wandb_api_key=wandb_api_key,
            team_id=app_config.team_id,
            eval_config=eval_config,
        )

        if output == "json":
            output_data_as_json({"run": run.model_dump()}, console)
            return

        console.print("[green]✓ Run created successfully![/green]")

        dashboard_url = f"{app_config.frontend_url}/dashboard/training/{run.id}"
        console.print("\n[cyan]Monitor run at:[/cyan]")
        console.print(f"  [link={dashboard_url}]{dashboard_url}[/link]")

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("models")
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


@app.command("list")
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
        rl_client = RLClient(api_client)

        run = rl_client.stop_run(run_id)

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


@app.command("logs")
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
                            for line in new_lines:
                                console.print(line)
                        else:
                            overlap = 0
                            max_overlap = min(len(old_lines), len(new_lines))
                            for i in range(1, max_overlap + 1):
                                if old_lines[-i:] == new_lines[:i]:
                                    overlap = i
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

                time.sleep(5)
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


@app.command("init")
def init_config(
    output_path: str = typer.Argument(
        "rl.toml",
        help="Output path for the config file",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing file"),
) -> None:
    """Generate a template config file for RL training.

    Example:

        prime rl init              # Creates rl.toml

        prime rl init my-config.toml
    """
    path = Path(output_path)

    if path.exists() and not force:
        console.print(f"[red]Error:[/red] {output_path} already exists. Use --force to overwrite.")
        raise typer.Exit(1)

    # Auto-detect environment
    environment: str | None = None
    metadata = find_environment_metadata()
    if metadata:
        owner = metadata.get("owner")
        name = metadata.get("name")
        if owner and name:
            environment = f"{owner}/{name}"
            console.print(f"[dim]Detected environment: {environment}[/dim]")

    path.parent.mkdir(parents=True, exist_ok=True)

    template = generate_rl_config_template(environment)
    path.write_text(template)

    console.print(f"[green]✓[/green] Created {output_path}")
    console.print(f"\n[dim]Run with:[/dim] prime rl run @ {output_path}")
