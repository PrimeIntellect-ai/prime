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
from typer.core import TyperGroup

from prime_cli.core import Config

from ..api.rl import RLClient, RLRun
from ..client import APIClient, APIError
from ..utils import output_data_as_json, validate_output_format
from ..utils.env_metadata import find_environment_metadata
from ..utils.env_vars import EnvParseError, collect_env_vars

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
    env_value = environment or "primeintellect/reverse-text"

    return f'''\
model = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"
max_steps = 100

# env_file = ["secrets.env"] # optional file(s) for keys/secrets

# Training
batch_size = 128
rollouts_per_example = 8
# trajectory_strategy = "interleaved"  # or "branching"
# learning_rate = 1e-6
# lora_alpha = 16
# oversampling_factor = 1.0
# max_async_level = 4

[sampling]
max_tokens = 2048
# temperature = 0.7

[[env]]
id = "{env_value}"

# [[env]] # add multiple [[env]] sections for multi-env training
# id = "primeintellect/another-env"
# args = {{ split = "train", max_examples = 1000 }}

# Optional: W&B logging
# [wandb]
# project = "my-project"
# entity = "my-team"
# name = "my-run-name"

# Optional: online evaluation
# [eval]
# interval = 100
# # optional: default for all environments
# num_examples = -1
# rollouts_per_example = 1
# eval_base_model = true
#
# [[eval.env]]
# id = "primeintellect/eval-env"
# args = {{ split = "test" }}
# # environment-specific overrides
# num_examples = 30
# rollouts_per_example = 4

# Optional: validation during training
# [val]
# num_examples = 64
# rollouts_per_example = 1
# interval = 5

# Optional: buffer configuration for difficulty filtering
# [buffer]
# easy_threshold = 0.8
# hard_threshold = 0.2
# easy_fraction = 0.0
# hard_fraction = 0.0
# online_difficulty_filtering = false
# env_ratios = [0.5, 0.5]
# skip_verification = false
# seed = 42
'''


class EnvConfig(BaseModel):
    id: str
    name: str | None = None
    args: Dict[str, Any] = Field(default_factory=dict)


class EvalEnvConfig(BaseModel):
    id: str
    name: str | None = None
    args: Dict[str, Any] = Field(default_factory=dict)
    num_examples: int | None = None
    rollouts_per_example: int | None = None


class SamplingConfig(BaseModel):
    max_tokens: int | None = None
    temperature: float | None = None


class EvalConfig(BaseModel):
    interval: int | None = None
    num_examples: int | None = None
    rollouts_per_example: int | None = None
    eval_base_model: bool | None = None
    env: List[EvalEnvConfig] = Field(default_factory=list)


class ValConfig(BaseModel):
    num_examples: int | None = None
    rollouts_per_example: int | None = None
    interval: int | None = None


class BufferConfig(BaseModel):
    easy_threshold: float | None = None
    hard_threshold: float | None = None
    easy_fraction: float | None = None
    hard_fraction: float | None = None
    online_difficulty_filtering: bool | None = None
    env_ratios: List[float] | None = None
    skip_verification: bool | None = None
    seed: int | None = None


class WandbConfig(BaseModel):
    entity: str | None = None
    project: str | None = None
    name: str | None = None


class RLConfig(BaseModel):
    name: str | None = None
    model: str | None = None
    max_steps: int = 100
    batch_size: int = 128
    rollouts_per_example: int = 8
    trajectory_strategy: str | None = None
    learning_rate: float | None = None
    lora_alpha: int | None = None
    oversampling_factor: float | None = None
    max_async_level: int | None = None
    env: List[EnvConfig] = Field(default_factory=list)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    val: ValConfig = Field(default_factory=ValConfig)
    buffer: BufferConfig = Field(default_factory=BufferConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    env_file: List[str] = Field(default_factory=list)


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


class DefaultGroup(TyperGroup):
    """Makes 'run' the default command when a config file is passed."""

    def parse_args(self, ctx, args):
        if not args:
            return super().parse_args(ctx, args)
        if args[0] in ("--help", "-h"):
            return super().parse_args(ctx, args)
        if args[0] in self.commands:
            return super().parse_args(ctx, args)
        args = ["run"] + list(args)
        return super().parse_args(ctx, args)

    def format_usage(self, ctx, formatter):
        formatter.write_usage(
            ctx.command_path,
            "[OPTIONS] CONFIG_PATH [ARGS]... | COMMAND [ARGS]...",
        )


app = typer.Typer(
    cls=DefaultGroup,
    help="Manage hosted RL training runs.",
    no_args_is_help=True,
)


@app.command("run")
def create_run(
    config_path: str = typer.Argument(
        ...,
        help="Path to TOML config file (e.g., rl.toml)",
    ),
    env: Optional[List[str]] = typer.Option(
        None,
        "-e",
        "--env-var",
        help=(
            "Environment variable/secret to pass to the training container. "
            "Accepts: KEY=VALUE (direct value), KEY (reads from $KEY), "
            "or path/to/file.env (loads env file)."
        ),
    ),
    env_file: Optional[List[str]] = typer.Option(
        None,
        "--env-file",
        help="Path to .env file containing secrets.",
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Start an RL training run from a config file.

    Example:

        prime rl run rl.toml
    """
    validate_output_format(output, console)

    console.print(f"[dim]Loading config from {config_path}[/dim]\n")
    cfg = load_config(config_path)

    # Validate required fields
    if not cfg.env:
        console.print("[red]Error:[/red] No environments specified. Add [[env]] sections.")
        raise typer.Exit(1)

    for train_env in cfg.env:
        if "/" not in train_env.id:
            console.print(
                f"[red]Error:[/red] Invalid environment format: '{train_env.id}'. "
                "Expected 'owner/name' format."
            )
            raise typer.Exit(1)

    # Validate eval environment IDs
    for eval_env in cfg.eval.env:
        if "/" not in eval_env.id:
            console.print(
                f"[red]Error:[/red] Invalid eval environment format: '{eval_env.id}'. "
                "Expected 'owner/name' format."
            )
            raise typer.Exit(1)

    if not cfg.model:
        console.print("[red]Error:[/red] No model specified.")
        raise typer.Exit(1)

    # Collect secrets from all sources
    def warn(msg: str) -> None:
        console.print(f"[yellow]Warning:[/yellow] {msg}")

    # Resolve config env_file paths relative to config file directory
    config_dir = Path(config_path).parent
    resolved_config_env_files = [str(config_dir / env_file_path) for env_file_path in cfg.env_file]

    # Merge config and CLI env files (CLI takes precedence)
    env_files = resolved_config_env_files + (env_file or [])

    try:
        secrets = collect_env_vars(
            env_args=env,
            env_files=env_files if env_files else None,
            on_warning=warn,
        )
    except EnvParseError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    # Warn if wandb is configured but no API key
    if (cfg.wandb.entity or cfg.wandb.project) and "WANDB_API_KEY" not in secrets:
        console.print(
            "[yellow]Warning:[/yellow] W&B config detected but no API key provided.\n"
            "  Set via: -e WANDB_API_KEY=... or --env-file\n"
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
        if cfg.sampling.temperature is not None:
            console.print(f"  Temperature: {cfg.sampling.temperature}")
        if cfg.learning_rate is not None:
            console.print(f"  Learning Rate: {cfg.learning_rate}")
        if cfg.lora_alpha is not None:
            console.print(f"  LoRA Alpha: {cfg.lora_alpha}")
        if cfg.oversampling_factor is not None:
            console.print(f"  Oversampling Factor: {cfg.oversampling_factor}")
        if cfg.max_async_level is not None:
            console.print(f"  Max Async Level: {cfg.max_async_level}")
        if cfg.wandb.project:
            console.print(f"  W&B Project: {cfg.wandb.project}")
        if cfg.eval.env:
            console.print(f"  Eval Environments: {', '.join(e.id for e in cfg.eval.env)}")
        if cfg.val.num_examples is not None:
            console.print(f"  Val Examples: {cfg.val.num_examples}")
        if secrets:
            console.print(f"  Secrets: {', '.join(secrets.keys())}")
        if app_config.team_id:
            console.print(f"  Team: {app_config.team_id}")
        console.print()

        # Build eval config if provided
        eval_config = None
        if cfg.eval.env:
            eval_environments = []
            for e in cfg.eval.env:
                env_cfg: Dict[str, Any] = {"id": e.id}
                if e.name is not None:
                    env_cfg["name"] = e.name
                if e.args:
                    env_cfg["args"] = e.args
                if e.num_examples is not None:
                    env_cfg["num_examples"] = e.num_examples
                if e.rollouts_per_example is not None:
                    env_cfg["rollouts_per_example"] = e.rollouts_per_example
                eval_environments.append(env_cfg)
            eval_config = {"environments": eval_environments}
            if cfg.eval.interval is not None:
                eval_config["interval"] = cfg.eval.interval
            if cfg.eval.num_examples is not None:
                eval_config["num_examples"] = cfg.eval.num_examples
            if cfg.eval.rollouts_per_example is not None:
                eval_config["rollouts_per_example"] = cfg.eval.rollouts_per_example
            if cfg.eval.eval_base_model is not None:
                eval_config["eval_base_model"] = cfg.eval.eval_base_model

        # Build val config if provided
        val_config = None
        has_val_config = (
            cfg.val.num_examples is not None
            or cfg.val.rollouts_per_example is not None
            or cfg.val.interval is not None
        )
        if has_val_config:
            val_config = {}
            if cfg.val.num_examples is not None:
                val_config["num_examples"] = cfg.val.num_examples
            if cfg.val.rollouts_per_example is not None:
                val_config["rollouts_per_example"] = cfg.val.rollouts_per_example
            if cfg.val.interval is not None:
                val_config["interval"] = cfg.val.interval

        # Build buffer config if provided
        buffer_config = None
        buffer_fields = [
            ("easy_threshold", cfg.buffer.easy_threshold),
            ("hard_threshold", cfg.buffer.hard_threshold),
            ("easy_fraction", cfg.buffer.easy_fraction),
            ("hard_fraction", cfg.buffer.hard_fraction),
            ("online_difficulty_filtering", cfg.buffer.online_difficulty_filtering),
            ("env_ratios", cfg.buffer.env_ratios),
            ("skip_verification", cfg.buffer.skip_verification),
            ("seed", cfg.buffer.seed),
        ]
        if any(v is not None for _, v in buffer_fields):
            buffer_config = {k: v for k, v in buffer_fields if v is not None}

        # Create the run
        run = rl_client.create_run(
            model_name=cfg.model,
            environments=[{"id": e.id, "name": e.name, "args": e.args} for e in cfg.env],
            rollouts_per_example=cfg.rollouts_per_example,
            max_steps=cfg.max_steps,
            max_tokens=cfg.sampling.max_tokens,
            temperature=cfg.sampling.temperature,
            batch_size=cfg.batch_size,
            trajectory_strategy=cfg.trajectory_strategy,
            name=cfg.name,
            wandb_entity=cfg.wandb.entity,
            wandb_project=cfg.wandb.project,
            wandb_run_name=cfg.wandb.name,
            secrets=secrets if secrets else None,
            team_id=app_config.team_id,
            eval_config=eval_config,
            val_config=val_config,
            buffer_config=buffer_config,
            learning_rate=cfg.learning_rate,
            lora_alpha=cfg.lora_alpha,
            oversampling_factor=cfg.oversampling_factor,
            max_async_level=cfg.max_async_level,
        )

        if output == "json":
            output_data_as_json({"run": run.model_dump()}, console)
            return

        console.print("[green]✓ Run created successfully![/green]")

        dashboard_url = f"{app_config.frontend_url}/dashboard/training/{run.id}"
        console.print("\n[cyan]Monitor run at:[/cyan]")
        console.print(f"  [link={dashboard_url}]{dashboard_url}[/link]")

        console.print("\n[dim]View logs with:[/dim]")
        console.print(f"  prime rl logs {run.id} -f")

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
    console.print(f"\n[dim]Run with:[/dim] prime rl run {output_path}")
