"""RL (Reinforcement Learning) training commands."""

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import toml
import typer
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError
from rich.console import Console
from rich.table import Table
from typer.core import TyperGroup

from prime_cli.core import Config

from ..api.rl import RLClient, RLRun
from ..client import APIClient, APIError, ValidationError
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

# env_files = ["secrets.env"] # optional file(s) for secrets

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
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str | None = None
    args: Dict[str, Any] = Field(default_factory=dict)

    def to_api_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"id": self.id}
        if self.name is not None:
            result["name"] = self.name
        if self.args:
            result["args"] = self.args
        return result


class EvalEnvConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str | None = None
    args: Dict[str, Any] = Field(default_factory=dict)
    num_examples: int | None = None
    rollouts_per_example: int | None = None

    def to_api_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"id": self.id}
        if self.name is not None:
            result["name"] = self.name
        if self.args:
            result["args"] = self.args
        if self.num_examples is not None:
            result["num_examples"] = self.num_examples
        if self.rollouts_per_example is not None:
            result["rollouts_per_example"] = self.rollouts_per_example
        return result


class SamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_tokens: int | None = None
    temperature: float | None = None


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    interval: int | None = None
    num_examples: int | None = None
    rollouts_per_example: int | None = None
    eval_base_model: bool | None = None
    env: List[EvalEnvConfig] = Field(default_factory=list)

    def to_api_dict(self) -> Dict[str, Any] | None:
        if not self.env:
            return None
        result: Dict[str, Any] = {"environments": [e.to_api_dict() for e in self.env]}
        if self.interval is not None:
            result["interval"] = self.interval
        if self.num_examples is not None:
            result["num_examples"] = self.num_examples
        if self.rollouts_per_example is not None:
            result["rollouts_per_example"] = self.rollouts_per_example
        if self.eval_base_model is not None:
            result["eval_base_model"] = self.eval_base_model
        return result


class ValConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_examples: int | None = None
    rollouts_per_example: int | None = None
    interval: int | None = None

    def to_api_dict(self) -> Dict[str, Any] | None:
        result: Dict[str, Any] = {}
        if self.num_examples is not None:
            result["num_examples"] = self.num_examples
        if self.rollouts_per_example is not None:
            result["rollouts_per_example"] = self.rollouts_per_example
        if self.interval is not None:
            result["interval"] = self.interval
        return result if result else None


class BufferConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    easy_threshold: float | None = None
    hard_threshold: float | None = None
    easy_fraction: float | None = None
    hard_fraction: float | None = None
    online_difficulty_filtering: bool | None = None
    env_ratios: List[float] | None = None
    skip_verification: bool | None = None
    seed: int | None = None

    def to_api_dict(self) -> Dict[str, Any] | None:
        result: Dict[str, Any] = {}
        if self.easy_threshold is not None:
            result["easy_threshold"] = self.easy_threshold
        if self.hard_threshold is not None:
            result["hard_threshold"] = self.hard_threshold
        if self.easy_fraction is not None:
            result["easy_fraction"] = self.easy_fraction
        if self.hard_fraction is not None:
            result["hard_fraction"] = self.hard_fraction
        if self.online_difficulty_filtering is not None:
            result["online_difficulty_filtering"] = self.online_difficulty_filtering
        if self.env_ratios is not None:
            result["env_ratios"] = self.env_ratios
        if self.skip_verification is not None:
            result["skip_verification"] = self.skip_verification
        if self.seed is not None:
            result["seed"] = self.seed
        return result if result else None


class WandbConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity: str | None = None
    project: str | None = None
    name: str | None = None


class RLConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    model: str
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
    env_file: List[str] = Field(default_factory=list)  # deprecated, use env_files
    env_files: List[str] = Field(default_factory=list)


def _format_validation_errors(errors: list[dict]) -> list[str]:
    """Format Pydantic validation errors into user-friendly messages."""
    messages = []
    for error in errors:
        loc = ".".join(str(x) for x in error["loc"])
        msg = error["msg"]
        # Clean up common Pydantic message prefixes
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, ") :]
        messages.append(f"{loc}: {msg}")
    return messages


def load_config(path: str) -> RLConfig:
    """Load config from TOML file."""
    p = Path(path)
    if not p.exists():
        console.print(f"[red]Error:[/red] Config file not found: {path}")
        raise typer.Exit(1)
    try:
        data = toml.load(p)
    except toml.TomlDecodeError as e:
        console.print(f"[red]Error:[/red] Invalid TOML in {path}: {e}")
        raise typer.Exit(1)

    try:
        return RLConfig.model_validate(data)
    except PydanticValidationError as e:
        console.print(f"[red]Error:[/red] Invalid config in {path}:\n")
        for msg in _format_validation_errors(e.errors()):
            console.print(f"  [red]•[/red] {msg}")
        console.print()
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


@app.command("run", rich_help_panel="Commands")
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
    skip_action_check: bool = typer.Option(
        False,
        "--skip-action-check",
        help="Skip action status check and run even if environment action failed.",
    ),
) -> None:
    """Start an RL training run from a config file.

    Example:

        prime rl run rl.toml
    """
    validate_output_format(output, console)

    console.print(f"[dim]Loading config from {config_path}[/dim]\n")
    cfg = load_config(config_path)

    # Collect secrets from all sources
    def warn(msg: str) -> None:
        console.print(f"[yellow]Warning:[/yellow] {msg}")

    # Resolve config env file paths relative to config file directory
    config_dir = Path(config_path).parent
    config_env_files = cfg.env_file + cfg.env_files  # support both, env_files takes precedence
    resolved_config_env_files = [str(config_dir / p) for p in config_env_files]

    # Merge config and CLI env files (CLI takes precedence)
    all_env_files = resolved_config_env_files + (env_file or [])

    try:
        secrets = collect_env_vars(
            env_args=env,
            env_files=all_env_files if all_env_files else None,
            on_warning=warn,
        )
    except EnvParseError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    # Validate WANDB_API_KEY is present when W&B monitoring is configured
    wandb_configured = cfg.wandb.entity or cfg.wandb.project
    if wandb_configured and (not secrets or "WANDB_API_KEY" not in secrets):
        console.print("[red]Configuration Error:[/red]")
        console.print("  WANDB_API_KEY is required when W&B monitoring is configured.\n")
        console.print("Provide it via:")
        console.print('  - env_files in your config: env_files = ["secrets.env"]')
        console.print("  - CLI flag: --env-file secrets.env")
        console.print("  - CLI flag: -e WANDB_API_KEY=your-key")
        console.print(
            "  - Environment variable: export WANDB_API_KEY=... && prime rl ... -e WANDB_API_KEY"
        )
        raise typer.Exit(1)

    try:
        api_client = APIClient()
        rl_client = RLClient(api_client)
        app_config = Config()

        # Show configuration in organized sections
        console.print("[white]Configuration:[/white]\n")

        # Model & Environment
        console.print("[cyan]Model & Environment[/cyan]")
        console.print(f"  Model:        {cfg.model}")
        console.print(f"  Environments: {', '.join(e.id for e in cfg.env)}")
        if app_config.team_id:
            console.print(f"  Team:         {app_config.team_id}")

        # Training
        console.print("\n[cyan]Training[/cyan]")
        console.print(f"  Max Steps:           {cfg.max_steps}")
        console.print(f"  Batch Size:          {cfg.batch_size}")
        console.print(f"  Rollouts per Example: {cfg.rollouts_per_example}")
        if cfg.learning_rate is not None:
            console.print(f"  Learning Rate:       {cfg.learning_rate}")
        if cfg.lora_alpha is not None:
            console.print(f"  LoRA Alpha:          {cfg.lora_alpha}")
        if cfg.oversampling_factor is not None:
            console.print(f"  Oversampling Factor: {cfg.oversampling_factor}")
        if cfg.max_async_level is not None:
            console.print(f"  Max Async Level:     {cfg.max_async_level}")

        # Sampling
        if cfg.sampling.max_tokens or cfg.sampling.temperature is not None:
            console.print("\n[cyan]Sampling[/cyan]")
            if cfg.sampling.max_tokens:
                console.print(f"  Max Tokens:  {cfg.sampling.max_tokens}")
            if cfg.sampling.temperature is not None:
                console.print(f"  Temperature: {cfg.sampling.temperature}")

        # W&B
        if cfg.wandb.entity or cfg.wandb.project:
            console.print("\n[cyan]Weights & Biases[/cyan]")
            console.print(f"  Project: {cfg.wandb.entity or '?'}/{cfg.wandb.project or '?'}")
            if cfg.wandb.name:
                console.print(f"  Run Name: {cfg.wandb.name}")

        # Eval
        if cfg.eval.env:
            console.print("\n[cyan]Evaluation[/cyan]")
            console.print(f"  Environments: {', '.join(e.id for e in cfg.eval.env)}")
            if cfg.eval.interval:
                console.print(f"  Interval:     {cfg.eval.interval}")

        # Validation
        if cfg.val.num_examples is not None:
            console.print("\n[cyan]Validation[/cyan]")
            console.print(f"  Num Examples: {cfg.val.num_examples}")
            if cfg.val.interval:
                console.print(f"  Interval:     {cfg.val.interval}")

        # Secrets
        if secrets:
            console.print("\n[cyan]Secrets[/cyan]")
            console.print(f"  Keys: {', '.join(secrets.keys())}")

        console.print()

        # Check action status for hub environments
        hub_envs = [e for e in cfg.env if "/" in e.id]
        if hub_envs and not skip_action_check:
            console.print("[dim]Checking Environment Actions...[/dim]")
            failed_envs = []

            for env_config in hub_envs:
                env_id_base = env_config.id.split("@")[0]
                owner, name = env_id_base.split("/", 1)
                try:
                    status_resp = rl_client.get_environment_status(owner, name)
                    action = status_resp.get("action") or {}
                    action_status = action.get("status")

                    if action_status == "FAILED":
                        console.print(f"  [red]✗[/red] {env_config.id} [dim](failed)[/dim]")
                        failed_envs.append(env_config.id)
                    elif action_status == "SUCCESS":
                        console.print(f"  [green]✓[/green] {env_config.id} [dim](success)[/dim]")
                    elif action_status in ("RUNNING", "PENDING"):
                        console.print(
                            f"  [yellow]○[/yellow] {env_config.id} [dim](in progress)[/dim]"
                        )
                    else:
                        console.print(f"  [dim]-[/dim] {env_config.id} [dim](no action)[/dim]")
                except APIError:
                    console.print(f"  [dim]-[/dim] {env_config.id} [dim](could not check)[/dim]")

            if failed_envs:
                console.print("\n[red]Error: Action failed for environments:[/red]\n")
                for env_id in failed_envs:
                    env_id_base = env_id.split("@")[0]
                    owner, name = env_id_base.split("/", 1)
                    url = f"{app_config.frontend_url}/dashboard/environments/{owner}/{name}/actions"
                    console.print(f"  [red]✗[/red] {env_id}")
                    console.print(f"    [link={url}]{url}[/link]\n")

                console.print(
                    "[yellow]This usually means the environment doesn't compile or run, "
                    "or is using an unsupported version of verifiers, so the training run "
                    "will fail.[/yellow]"
                )
                console.print("[dim]To proceed anyway, use --skip-action-check[/dim]")
                raise typer.Exit(1)

            console.print()

        console.print("[dim]Creating RL training run...[/dim]\n")

        # Create the run
        run = rl_client.create_run(
            model_name=cfg.model,
            environments=[e.to_api_dict() for e in cfg.env],
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
            eval_config=cfg.eval.to_api_dict(),
            val_config=cfg.val.to_api_dict(),
            buffer_config=cfg.buffer.to_api_dict(),
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

    except ValidationError as e:
        console.print("[red]Configuration Error:[/red]")
        for err in e.errors:
            loc = err.get("loc", [])
            path = ".".join(str(x) for x in loc if x != "body")
            msg = err.get("msg", "")
            if msg.startswith("Value error, "):
                msg = msg[len("Value error, ") :]
            if path:
                console.print(f"  [yellow]{path}[/yellow]: {msg}")
            else:
                console.print(f"  {msg}")
        raise typer.Exit(1)
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("models", rich_help_panel="Commands")
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


def _list_runs_impl(team: Optional[str], num: int, output: str) -> None:
    """Implementation for listing RL training runs."""
    validate_output_format(output, console)

    if num < 1:
        console.print("[red]Error:[/red] --num must be at least 1")
        raise typer.Exit(1)

    try:
        api_client = APIClient()
        rl_client = RLClient(api_client)
        config = Config()

        team_id = team or config.team_id

        all_runs = rl_client.list_runs(team_id=team_id)
        total_count = len(all_runs)

        # Sort by created_at descending and limit
        all_runs.sort(key=lambda r: r.created_at, reverse=True)
        runs = all_runs[:num]

        if output == "json":
            output_data_as_json(
                {"runs": [r.model_dump() for r in runs], "total": total_count}, console
            )
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

        if total_count > num:
            console.print(
                f"\n[dim]Showing {len(runs)} of {total_count} runs. "
                "Use --num (-n) to see more.[/dim]"
            )
        else:
            console.print(f"\n[dim]Total: {total_count} run(s)[/dim]")

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("list", rich_help_panel="Commands")
@app.command("ls", rich_help_panel="Commands", hidden=True)
def list_runs(
    team: Optional[str] = typer.Option(None, "--team", "-t", help="Filter by team ID"),
    num: int = typer.Option(20, "--num", "-n", help="Number of most recent runs to show"),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List your runs (alias: ls)."""
    _list_runs_impl(team, num, output)


@app.command("get", rich_help_panel="Commands")
def get_run(
    run_id: str = typer.Argument(..., help="Run ID to get details for"),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Get details of a specific run.

    Example:

        prime rl get <run_id>

        prime rl get <run_id> -o json
    """
    validate_output_format(output, console)

    try:
        api_client = APIClient()
        rl_client = RLClient(api_client)

        run = rl_client.get_run(run_id)

        if output == "json":
            output_data_as_json({"run": run.model_dump()}, console)
            return

        # Display run details
        formatted = _format_run_for_display(run)
        status_color = _get_status_color(run.status)

        console.print(f"[bold]Run {run_id}[/bold]\n")
        console.print(f"  Status: [{status_color}]{run.status}[/{status_color}]")
        console.print(f"  Model: [magenta]{run.base_model}[/magenta]")
        console.print(f"  Environments: [green]{formatted['environments']}[/green]")
        console.print(f"  Max Steps: {run.max_steps}")
        console.print(f"  Batch Size: {run.batch_size}")
        console.print(f"  Rollouts per Example: {run.rollouts_per_example}")
        if run.max_tokens:
            console.print(f"  Max Tokens: {run.max_tokens}")
        if run.wandb_project:
            console.print(f"  W&B: {run.wandb_entity or ''}/{run.wandb_project}")
        if run.team_id:
            console.print(f"  Team: {run.team_id}")
        console.print(f"  Created: [dim]{formatted['created_at']}[/dim]")
        if run.started_at:
            console.print(f"  Started: [dim]{run.started_at.strftime('%Y-%m-%d %H:%M')}[/dim]")
        if run.completed_at:
            console.print(f"  Completed: [dim]{run.completed_at.strftime('%Y-%m-%d %H:%M')}[/dim]")
        if run.error_message:
            console.print(f"  Error: [red]{run.error_message}[/red]")

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("stop", rich_help_panel="Commands")
def stop_run(
    run_id: str = typer.Argument(..., help="Run ID to stop"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Stop a run."""
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


@app.command("delete", rich_help_panel="Commands")
def delete_run(
    run_id: str = typer.Argument(..., help="Run ID to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Delete a run."""
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


@app.command("logs", rich_help_panel="Monitoring")
def get_logs(
    run_id: str = typer.Argument(..., help="Run ID to get logs for"),
    tail: int = typer.Option(1000, "--tail", "-n", help="Number of lines to show"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
) -> None:
    """Get logs for a run."""
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


@app.command("init", rich_help_panel="Commands")
def init_config(
    output_path: str = typer.Argument(
        "rl.toml",
        help="Output path for the config file",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing file"),
) -> None:
    """Generate a template config file for a training run.

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


@app.command("metrics", rich_help_panel="Monitoring")
def get_metrics(
    run_id: str = typer.Argument(..., help="Run ID to get metrics for"),
    min_step: Optional[int] = typer.Option(None, "--min-step", help="Minimum step (inclusive)"),
    max_step: Optional[int] = typer.Option(None, "--max-step", help="Maximum step (inclusive)"),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Maximum number of records"),
) -> None:
    """Get training metrics for a run.

    Example:

        prime rl metrics <run_id>

        prime rl metrics <run_id> --min-step 10 --max-step 50

        prime rl metrics <run_id> | jq '.metrics[0]'
    """
    try:
        api_client = APIClient()
        rl_client = RLClient(api_client)

        metrics = rl_client.get_metrics(
            run_id,
            min_step=min_step,
            max_step=max_step,
            limit=limit,
        )

        output_data_as_json({"metrics": metrics}, console)

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("rollouts", rich_help_panel="Monitoring")
def get_rollouts(
    run_id: str = typer.Argument(..., help="Run ID to get rollouts for"),
    step: int = typer.Option(..., "--step", "-s", help="Step number to get rollouts for"),
    page: int = typer.Option(1, "--page", "-p", help="Page number (1-indexed)"),
    limit: int = typer.Option(100, "--limit", "-n", help="Number of samples per page"),
) -> None:
    """Get rollout samples for a run.

    Example:

        prime rl rollouts <run_id> --step 10

        prime rl rollouts <run_id> -s 50 --limit 100 | jq '.samples[0]'
    """
    try:
        api_client = APIClient()
        rl_client = RLClient(api_client)

        result = rl_client.get_rollouts(
            run_id,
            step=step,
            page=page,
            limit=limit,
        )

        output_data_as_json(result, console)

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("progress", rich_help_panel="Monitoring")
def get_progress(
    run_id: str = typer.Argument(..., help="Run ID to get progress for"),
) -> None:
    """Get progress information, including which steps have samples and distributions.

    Example:

        prime rl progress <run_id>

        prime rl progress <run_id> | jq '.latest_step'
    """
    try:
        api_client = APIClient()
        rl_client = RLClient(api_client)

        progress = rl_client.get_progress(run_id)

        output_data_as_json(progress, console)

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("distributions", rich_help_panel="Monitoring")
def get_distributions(
    run_id: str = typer.Argument(..., help="Run ID to get distributions for"),
    distribution_type: Optional[str] = typer.Option(
        None, "--type", "-t", help="Distribution type (defaults to all)"
    ),
    step: Optional[int] = typer.Option(
        None, "--step", "-s", help="Step number (defaults to latest)"
    ),
) -> None:
    """Get reward/advantage distribution histogram for a run.

    Example:

        prime rl distributions <run_id>

        prime rl distributions <run_id> --type rewards --step 50
    """
    try:
        api_client = APIClient()
        rl_client = RLClient(api_client)

        result = rl_client.get_distributions(
            run_id,
            distribution_type=distribution_type,
            step=step,
        )

        output_data_as_json(result, console)

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
