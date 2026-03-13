import json
import re
import time
from functools import wraps
from pathlib import Path
from typing import Any, Optional

import typer
from click.core import ParameterSource
from prime_evals import EvalsAPIError, EvalsClient, InvalidEvaluationError
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table
from typer.core import TyperGroup

from ..client import APIClient, APIError
from ..core import Config
from ..utils import output_data_as_json
from ..utils.display import get_eval_viewer_url
from ..utils.env_metadata import find_environment_metadata
from ..utils.eval_push import load_results_jsonl
from ..utils.hosted_eval import (
    EvalStatus,
    HostedEvalConfig,
    HostedEvalResult,
    clean_logs,
    get_new_log_lines,
)
from ..verifiers_bridge import (
    DEFAULT_ENV_DIR_PATH,
    DEFAULT_MODEL,
    _is_config_target,
    _parse_value_option,
    _resolve_environment_reference,
    _split_owner_and_name,
    is_help_request,
    print_eval_run_help,
    run_eval_passthrough,
    run_eval_tui,
)

console = Console()

HOSTED_LOGS_DEFAULT_TAIL_LINES = 1000
HOSTED_LOGS_DEFAULT_POLL_INTERVAL_SECONDS = 5.0
HOSTED_RUN_DEFAULT_POLL_INTERVAL_SECONDS = 10.0
HOSTED_RUN_DEFAULT_NUM_EXAMPLES = 5
HOSTED_RUN_DEFAULT_ROLLOUTS_PER_EXAMPLE = 3
HOSTED_LOGS_RATE_LIMIT_THRESHOLD = 3
HOSTED_LOGS_RATE_LIMIT_WAIT_SECONDS = 30
HOSTED_LOGS_RETRY_WAIT_SECONDS = 10
HOSTED_LOGS_STATUS_UPDATE_EVERY_POLLS = 6
EVAL_TABLE_MAX_TEXT_WIDTH = 30
EVAL_RUN_EXAMPLE_COMMAND = "prime eval run gsm8k -n 10"
EVAL_HOSTED_LABEL = "HOSTED"
EVAL_LOCAL_LABEL = "LOCAL"


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
            "[OPTIONS] ENVIRONMENT [ARGS]... | COMMAND [ARGS]...",
        )


subcommands_app = typer.Typer()


def handle_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except typer.Exit:
            raise
        except EvalsAPIError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Unexpected error:[/red] {e}")
            raise typer.Exit(1)

    return wrapper


def _validate_output_format(output: str, allowed: list[str]) -> None:
    if output not in allowed:
        console.print(f"[red]Error:[/red] output must be one of: {', '.join(allowed)}")
        raise typer.Exit(1)


def format_output(data: dict, output: str) -> None:
    if output == "json":
        output_data_as_json(data, console)
    else:
        syntax = Syntax(json.dumps(data, indent=2), "json", theme="monokai")
        console.print(syntax)


def _parse_json_option(raw: Optional[str], option_name: str) -> Optional[dict[str, str]]:
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        console.print(f"[red]Error:[/red] invalid {option_name}: {exc}")
        raise typer.Exit(1) from exc

    if type(parsed) is not dict:
        console.print(f"[red]Error:[/red] {option_name} must be a JSON object")
        raise typer.Exit(1)

    for key, value in parsed.items():
        if type(key) is not str or type(value) is not str:
            console.print(
                f"[red]Error:[/red] {option_name} must contain only string keys and values"
            )
            raise typer.Exit(1)

    return parsed


def _fetch_eval_status(client: APIClient, eval_id: str) -> dict[str, Any]:
    return client.get(f"/evaluations/{eval_id}")


def _fetch_logs(client: APIClient, eval_id: str) -> str:
    response = client.get(f"/hosted-evaluations/{eval_id}/logs")
    return response.get("logs") or ""


def _build_hosted_evaluation_payload(config: HostedEvalConfig) -> dict[str, Any]:
    eval_config: dict[str, Any] = {
        "num_examples": config.num_examples,
        "rollouts_per_example": config.rollouts_per_example,
        "allow_sandbox_access": config.allow_sandbox_access,
        "allow_instances_access": config.allow_instances_access,
    }

    if config.env_args:
        eval_config["env_args"] = config.env_args
    if config.timeout_minutes is not None:
        eval_config["timeout_minutes"] = config.timeout_minutes
    if config.custom_secrets:
        eval_config["custom_secrets"] = config.custom_secrets

    payload: dict[str, Any] = {
        "environment_ids": [config.environment_id],
        "inference_model": config.inference_model,
        "eval_config": eval_config,
    }
    if config.name:
        payload["name"] = config.name

    return payload


def _create_hosted_evaluation(config: HostedEvalConfig) -> HostedEvalResult:
    client = APIClient()
    payload = _build_hosted_evaluation_payload(config)

    if client.config.team_id:
        payload["team_id"] = client.config.team_id

    created = client.post("/hosted-evaluations", json=payload)
    evaluation_id = created.get("evaluation_id")

    if not evaluation_id:
        raise APIError(f"Failed to get evaluation ID from response: {created}")

    return HostedEvalResult(
        evaluation_id=evaluation_id,
        status=EvalStatus.PENDING,
        total_samples=0,
        avg_score=None,
        min_score=None,
        max_score=None,
    )


def _print_eval_status(eval_data: dict[str, Any]) -> None:
    status_str, status = _parse_eval_status(eval_data)
    color = status.color if status else "white"

    console.print(f"[{color}]Status: {status_str}[/{color}]")

    error_message = eval_data.get("error_message")
    if error_message:
        console.print(f"[red]Error:[/red] {error_message}")

    viewer_url = eval_data.get("viewer_url")
    if viewer_url:
        console.print(f"[dim]View: {viewer_url}[/dim]")
        return

    eval_id = eval_data.get("evaluation_id")
    if eval_id:
        console.print(f"[dim]View: {get_eval_viewer_url(eval_id)}[/dim]")


def _parse_eval_status(eval_data: dict[str, Any]) -> tuple[str, EvalStatus | None]:
    raw_status = eval_data.get("status")
    status_str = raw_status if type(raw_status) is str else "UNKNOWN"

    try:
        return status_str, EvalStatus(status_str)
    except ValueError:
        return status_str, None


def _handle_log_poll_error(eval_id: str, exc: APIError, consecutive_errors: int) -> None:
    if "404" in str(exc):
        console.print(f"\n[red]Evaluation {eval_id} not found.[/red]")
        raise typer.Exit(1) from exc

    if "429" not in str(exc):
        raise exc

    if consecutive_errors >= HOSTED_LOGS_RATE_LIMIT_THRESHOLD:
        console.print(
            f"[yellow]Rate limited. Waiting {HOSTED_LOGS_RATE_LIMIT_WAIT_SECONDS:.0f}s...[/yellow]"
        )
        time.sleep(HOSTED_LOGS_RATE_LIMIT_WAIT_SECONDS)
    else:
        time.sleep(HOSTED_LOGS_RETRY_WAIT_SECONDS)


def _display_logs_follow(eval_id: str, poll_interval: float) -> None:
    console.print(f"[dim]Watching logs for evaluation {eval_id}... (Ctrl+C to stop)[/dim]\n")

    client = APIClient()
    last_logs = ""
    consecutive_errors = 0
    no_logs_polls = 0

    while True:
        try:
            eval_data = _fetch_eval_status(client, eval_id)
            status_str, status = _parse_eval_status(eval_data)

            if status and status in EvalStatus.terminal_statuses():
                final_logs = clean_logs(_fetch_logs(client, eval_id))
                if final_logs and final_logs != last_logs:
                    for line in get_new_log_lines(last_logs, final_logs):
                        console.print(line)
                    last_logs = final_logs
                console.print()
                _print_eval_status(eval_data)
                if status != EvalStatus.COMPLETED:
                    raise typer.Exit(1)
                return

            raw_logs = _fetch_logs(client, eval_id)
            logs = clean_logs(raw_logs) if raw_logs else ""
            consecutive_errors = 0

            if logs and logs != last_logs:
                for line in get_new_log_lines(last_logs, logs):
                    console.print(line)
                last_logs = logs
                no_logs_polls = 0
            else:
                no_logs_polls += 1

            if no_logs_polls > 0 and no_logs_polls % HOSTED_LOGS_STATUS_UPDATE_EVERY_POLLS == 0:
                console.print(f"[dim]Evaluation status: {status_str} (waiting for logs...)[/dim]")
        except APIError as exc:
            consecutive_errors += 1
            _handle_log_poll_error(eval_id, exc, consecutive_errors)
            continue

        time.sleep(poll_interval)


def _display_logs_once(eval_id: str, tail: int) -> None:
    client = APIClient()
    eval_data = _fetch_eval_status(client, eval_id)
    raw_logs = _fetch_logs(client, eval_id)
    logs = clean_logs(raw_logs) if raw_logs else ""

    if logs:
        for line in logs.splitlines()[-tail:]:
            console.print(line)
    else:
        console.print("[yellow]No logs available.[/yellow]")

    console.print()
    _print_eval_status(eval_data)


def _display_logs(
    eval_id: str,
    tail: int,
    follow: bool,
    poll_interval: float = HOSTED_LOGS_DEFAULT_POLL_INTERVAL_SECONDS,
) -> None:
    try:
        if follow:
            _display_logs_follow(eval_id, poll_interval)
        else:
            _display_logs_once(eval_id, tail)
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped watching logs. Evaluation continues running.[/dim]")
    except typer.Exit:
        raise
    except APIError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc


def _resolve_hosted_environment(
    environment: str,
    *,
    env_dir_path: Optional[str],
    env_path: Optional[str],
) -> tuple[str, str]:
    resolved = _resolve_environment_reference(environment, env_dir_path or DEFAULT_ENV_DIR_PATH)

    if resolved.recommend_push:
        console.print(
            "[red]Error:[/red] hosted evaluations require an environment "
            "that is published to the platform"
        )
        if resolved.platform_slug:
            console.print(
                "[yellow]Publish the latest local changes for "
                f"{resolved.platform_slug} first.[/yellow]"
            )
        else:
            console.print("[yellow]Publish the environment with `prime env push` first.[/yellow]")
        raise typer.Exit(1)

    platform_slug = resolved.platform_slug or resolved.upstream_slug
    if platform_slug is None and env_path:
        metadata = find_environment_metadata(
            env_name=resolved.env_name,
            env_path=Path(env_path),
            module_name=resolved.env_name.replace("-", "_"),
        )
        owner = metadata.get("owner") if metadata else None
        name = metadata.get("name") if metadata else None
        if owner and name:
            platform_slug = f"{owner}/{name}"

    if platform_slug is None:
        console.print(
            "[red]Error:[/red] hosted evaluations require an upstream environment on the platform"
        )
        console.print(
            "[yellow]Use an environment slug or publish the local environment "
            "with `prime env push`.[/yellow]"
        )
        raise typer.Exit(1)

    parts = _split_owner_and_name(platform_slug)
    if parts is None:
        console.print(f"[red]Error:[/red] invalid environment slug: {platform_slug}")
        raise typer.Exit(1)

    owner, env_name = parts
    api_client = APIClient()
    response = api_client.get(f"/environmentshub/{owner}/{env_name}/@latest")
    details = response.get("data", response)
    environment_id = details.get("id")
    if not environment_id:
        console.print(f"[red]Error:[/red] could not resolve environment id for {platform_slug}")
        raise typer.Exit(1)

    console.print(f"[dim]Using hosted environment {platform_slug}[/dim]")
    return platform_slug, str(environment_id)


@subcommands_app.command("list")
@handle_errors
def list_evals(
    output: str = typer.Option("table", "--output", "-o", help="table|json"),
    num: int = typer.Option(20, "--num", "-n", help="Items per page"),
    page: int = typer.Option(1, "--page", "-p", help="Page number"),
    env: Optional[str] = typer.Option(
        None,
        "--env",
        "--env-name",
        "-e",
        help="Filter by environment (e.g., 'gsm8k' or 'owner/gsm8k')",
    ),
) -> None:
    """List evaluations."""
    _validate_output_format(output, ["table", "json"])

    if num < 1 or page < 1:
        console.print("[red]Error:[/red] --num and --page must be at least 1")
        raise typer.Exit(1)

    try:
        api_client = APIClient()
        config = Config()
        client = EvalsClient(api_client)

        skip = (page - 1) * num
        data = client.list_evaluations(
            env_name=env,
            team_id=config.team_id,
            skip=skip,
            limit=num,
        )

        if output == "json":
            output_data_as_json(data, console)
            return

        evals = data.get("evaluations", [])

        if not evals:
            if page > 1:
                console.print("[yellow]No more results.[/yellow]")
            else:
                console.print("[yellow]No evaluations found.[/yellow]")
            return

        table = Table(title="Evaluations")
        table.add_column("ID", style="cyan")
        table.add_column("Environment", style="blue")
        table.add_column("Model", style="magenta")
        table.add_column("Status", style="yellow")
        table.add_column("Type", style="green", justify="center")
        table.add_column("Examples", style="dim", justify="right")
        table.add_column("Rollouts", style="dim", justify="right")

        for e in evals:
            eval_id = str(e.get("evaluation_id", e.get("id", "")))
            metadata = e.get("metadata", {})
            num_examples = metadata.get("num_examples", "-")
            rollouts_per_example = metadata.get("rollouts_per_example", "-")

            env_name = "-"
            environment_names = e.get("environment_names", [])
            if environment_names and len(environment_names) > 0:
                env_name = environment_names[0]

            is_hosted = bool(e.get("is_hosted"))
            execution_mode = EVAL_HOSTED_LABEL if is_hosted else EVAL_LOCAL_LABEL

            table.add_row(
                eval_id if eval_id else "",
                str(env_name)[:EVAL_TABLE_MAX_TEXT_WIDTH],
                str(e.get("model_name", ""))[:EVAL_TABLE_MAX_TEXT_WIDTH],
                str(e.get("status", "")),
                execution_mode,
                str(num_examples),
                str(rollouts_per_example),
            )

        console.print(table)
        total = data.get("total", 0)
        if total > page * num:
            console.print(
                f"\n[yellow]Showing page {page} of results. "
                f"Use --page {page + 1} to see more.[/yellow]"
            )
        else:
            console.print(f"\n[dim]Total: {total} evaluation(s)[/dim]")

    except EvalsAPIError as e:
        console.print(f"[red]API Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print(
            "[yellow]Response may contain invalid data. "
            "Try --output json to see raw response.[/yellow]"
        )
        raise typer.Exit(1)


@subcommands_app.command("get")
@handle_errors
def get_eval(
    eval_id: str = typer.Argument(..., help="The ID of the evaluation to retrieve"),
    output: str = typer.Option("json", "--output", "-o", help="json|pretty"),
) -> None:
    _validate_output_format(output, ["json", "pretty"])

    api_client = APIClient()
    client = EvalsClient(api_client)
    data = client.get_evaluation(eval_id)
    format_output(data, output)


@subcommands_app.command("samples")
@handle_errors
def get_samples(
    eval_id: str = typer.Argument(..., help="The ID of the evaluation"),
    page: int = typer.Option(1, "--page", "-p", help="Page number"),
    num: int = typer.Option(100, "--num", "-n", help="Items per page"),
    output: str = typer.Option("json", "--output", "-o", help="json|pretty"),
) -> None:
    _validate_output_format(output, ["json", "pretty"])

    api_client = APIClient()
    client = EvalsClient(api_client)
    data = client.get_samples(eval_id, page=page, limit=num)
    format_output(data, output)


def _load_eval_directory(directory: Path) -> dict:
    with open(directory / "metadata.json") as f:
        metadata = json.load(f)

    env_field = metadata.get("env_id") or metadata.get("env")
    if not env_field or "model" not in metadata:
        raise ValueError(
            f"Missing required 'env_id' or 'model' field in {directory / 'metadata.json'}"
        )

    results = load_results_jsonl(directory / "results.jsonl")

    for sample in results:
        if "id" in sample and "example_id" not in sample:
            sample["example_id"] = sample["id"]

    avg_pattern = re.compile(r"^avg_(.+)$")
    metrics = {}
    metadata_copy = {}
    for key, value in metadata.items():
        if match := avg_pattern.match(key):
            metrics[match.group(1)] = value
        else:
            metadata_copy[key] = value

    return {
        "eval_name": f"{env_field}-{metadata['model']}",
        "model_name": metadata["model"],
        "env": env_field,
        "metrics": metrics,
        "metadata": metadata_copy,
        "results": results,
    }


def _has_eval_files(directory: Path) -> bool:
    return (directory / "metadata.json").exists() and (directory / "results.jsonl").exists()


def _validate_eval_path(path_str: str) -> Path:
    """Validate and return the evaluation directory path."""
    path = Path(path_str)

    if path.is_file():
        # Auto-correct: if user passed metadata.json or results.jsonl, use parent directory
        if path.name in ("metadata.json", "results.jsonl"):
            parent = path.parent
            if _has_eval_files(parent):
                return parent
            raise ValueError(
                f"Directory '{parent}' must contain both metadata.json and results.jsonl"
            )
        raise ValueError(
            f"Expected a directory path, but got file: {path}\n"
            f"Pass a directory containing metadata.json and results.jsonl"
        )

    if path.is_dir():
        if _has_eval_files(path):
            return path

        has_metadata = (path / "metadata.json").exists()
        has_results = (path / "results.jsonl").exists()
        if has_metadata and not has_results:
            raise ValueError(f"Directory '{path}' is missing results.jsonl")
        elif has_results and not has_metadata:
            raise ValueError(f"Directory '{path}' is missing metadata.json")
        else:
            raise ValueError(f"Directory '{path}' is missing both metadata.json and results.jsonl")

    raise FileNotFoundError(f"Path not found: {path}")


def _discover_eval_outputs() -> list[Path]:
    outputs_dir = Path("outputs/evals")
    if not outputs_dir.exists():
        return []

    eval_dirs = []
    for env_dir in outputs_dir.iterdir():
        if not env_dir.is_dir():
            continue
        for run_dir in env_dir.iterdir():
            if run_dir.is_dir() and _has_eval_files(run_dir):
                eval_dirs.append(run_dir)

    return sorted(eval_dirs)


def _push_single_eval(
    config_path: str,
    env_slug: Optional[str],
    run_id: Optional[str],
    eval_id: Optional[str],
    is_public: bool = False,
) -> str:
    path = _validate_eval_path(config_path)
    eval_data = _load_eval_directory(path)
    console.print(f"[blue]✓ Loaded eval data:[/blue] {path}")

    detected_env = eval_data.get("env_id") or eval_data.get("env")
    if not env_slug and detected_env and not run_id and not eval_id:
        env_slug = detected_env

    environments = None
    if env_slug and not run_id and not eval_id:
        # Determine if env_slug is a slug (owner/name) or a name
        # Use appropriate key so _resolve_environments can properly resolve it
        if "/" in env_slug:
            # It's a slug (owner/name format)
            environments = [{"slug": env_slug}]
        else:
            # It's a name (will be resolved by _resolve_environments)
            environments = [{"name": env_slug}]

    console.print()

    api_client = APIClient()
    client = EvalsClient(api_client)

    if eval_id:
        console.print(f"[blue]Checking evaluation:[/blue] {eval_id}")
        try:
            client.get_evaluation(eval_id)
            console.print("[green]✓ Found existing evaluation[/green]")

            console.print("[blue]Updating evaluation...[/blue]")
            client.update_evaluation(
                evaluation_id=eval_id,
                name=eval_data.get("eval_name"),
                model_name=eval_data.get("model_name"),
                framework=eval_data.get("metadata", {}).get("framework", "verifiers"),
                task_type=eval_data.get("metadata", {}).get("task_type"),
                metadata=eval_data.get("metadata"),
                metrics=eval_data.get("metrics"),
                tags=eval_data.get("tags", []),
            )
            console.print(f"[green]✓ Updated evaluation:[/green] {eval_id}")
        except Exception as e:
            console.print(f"[red]Error:[/red] Could not update evaluation {eval_id}: {e}")
            raise
        console.print()
    else:
        console.print("[blue]Creating evaluation...[/blue]")
        create_response = client.create_evaluation(
            name=eval_data["eval_name"],
            environments=environments,
            run_id=run_id,
            model_name=eval_data.get("model_name"),
            framework=eval_data.get("metadata", {}).get("framework", "verifiers"),
            task_type=eval_data.get("metadata", {}).get("task_type"),
            metadata=eval_data.get("metadata"),
            metrics=eval_data.get("metrics"),
            tags=eval_data.get("tags", []),
            is_public=is_public,
        )

        eval_id = create_response.get("evaluation_id")
        if not eval_id:
            raise ValueError("Failed to get evaluation ID from response")

        console.print(f"[green]✓ Created evaluation:[/green] {eval_id}")
        console.print()

    results = eval_data.get("results", [])
    if results:
        console.print(f"[blue]Pushing {len(results)} samples...[/blue]")
        client.push_samples(eval_id, results)
        console.print("[green]✓ Samples pushed successfully[/green]")
        console.print()

    console.print("[blue]Finalizing evaluation...[/blue]")
    client.finalize_evaluation(eval_id, metrics=eval_data.get("metrics"))
    console.print("[green]✓ Evaluation finalized[/green]")
    console.print()

    console.print("[green]✓ Success[/green]")
    console.print(f"[blue]Evaluation ID:[/blue] {eval_id}")
    console.print()
    console.print("[dim]View your evaluation:[/dim]")
    console.print(f"  prime eval get {eval_id}")
    console.print(f"  prime eval samples {eval_id}")

    return eval_id


@subcommands_app.command("tui")
def tui_cmd(
    env_dir: Optional[str] = typer.Option(
        None, "--env-dir", "-e", help="Path to environments directory"
    ),
    outputs_dir: Optional[str] = typer.Option(
        None, "--outputs-dir", "-o", help="Path to outputs directory"
    ),
) -> None:
    """Launch TUI for viewing eval results."""
    run_eval_tui(env_dir=env_dir, outputs_dir=outputs_dir)


@subcommands_app.command("push")
@handle_errors
def push_eval(
    config_path: Optional[str] = typer.Argument(
        None,
        help=(
            "Path to eval directory containing metadata.json and results.jsonl. "
            "If not provided, auto-discovers from outputs/evals/"
        ),
    ),
    env_id: Optional[str] = typer.Option(
        None,
        "--env",
        "--env-id",
        "-e",
        help="Environment name (e.g., 'gsm8k' or 'owner/gsm8k')",
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        "-r",
        help="Link to existing training run id",
    ),
    eval_id: Optional[str] = typer.Option(
        None,
        "--eval",
        "--eval-id",
        help="Push to existing evaluation id",
    ),
    output: str = typer.Option("pretty", "--output", "-o", help="json|pretty"),
    is_public: bool = typer.Option(
        False,
        "--public",
        help="Make the pushed evaluation public. Evaluations are private by default.",
    ),
) -> None:
    """Push evaluation data to Prime Evals.

    The directory must contain metadata.json and results.jsonl files.

    \b
    Examples:
        prime eval push                                    # Push current dir or auto-discover
        prime eval push outputs/evals/gsm8k--gpt-4/abc123  # Push specific directory
        prime eval push --env gsm8k                        # Push with environment override
        prime eval push --public                           # Create a public evaluation
        prime eval push --eval xyz789                      # Push to existing evaluation
    """
    try:
        if eval_id and is_public:
            console.print(
                "[red]Error:[/red] The --public flag cannot be used with --eval-id. "
                "Visibility can only be set when creating a new evaluation."
            )
            raise typer.Exit(1)

        if config_path is None and eval_id:
            console.print("[red]Error:[/red] Cannot use --eval-id with auto-discovery")
            console.print()
            console.print("[yellow]Tip:[/yellow] Specify an explicit path when using --eval-id:")
            console.print("  prime eval push /path/to/eval/data --eval-id <eval-id>")
            console.print("  prime eval push outputs/evals/env--model/run-id --eval-id <eval-id>")
            raise typer.Exit(1)

        if config_path is None:
            current_dir = Path(".")
            if _has_eval_files(current_dir):
                result_eval_id = _push_single_eval(".", env_id, run_id, eval_id, is_public)
                if output == "json":
                    console.print()
                    output_data_as_json({"evaluation_id": result_eval_id}, console)
                return

            eval_dirs = _discover_eval_outputs()
            if not eval_dirs:
                console.print("[red]Error:[/red] No evaluation outputs found")
                console.print(
                    "[yellow]Hint:[/yellow] Run from a directory with "
                    "metadata.json and results.jsonl, or from a directory containing outputs/evals/"
                )
                raise typer.Exit(1)

            console.print(f"[blue]Found {len(eval_dirs)} evaluation(s) to push:[/blue]")
            for eval_dir in eval_dirs:
                console.print(f"  - {eval_dir}")
            console.print()

            results = []
            for eval_dir in eval_dirs:
                try:
                    result_eval_id = _push_single_eval(
                        str(eval_dir), env_id, run_id, eval_id, is_public
                    )
                    results.append(
                        {"path": str(eval_dir), "eval_id": result_eval_id, "status": "success"}
                    )
                except Exception as e:
                    console.print(f"[red]Failed to push {eval_dir}:[/red] {e}")
                    results.append({"path": str(eval_dir), "error": str(e), "status": "failed"})
                console.print()

            success_count = sum(1 for r in results if r["status"] == "success")
            console.print(
                f"[blue]Summary:[/blue] {success_count}/{len(eval_dirs)} "
                f"evaluations pushed successfully"
            )

            if output == "json":
                output_data_as_json({"results": results}, console)

            if success_count < len(eval_dirs):
                raise typer.Exit(1)

            return

        result_eval_id = _push_single_eval(config_path, env_id, run_id, eval_id, is_public)

        if output == "json":
            console.print()
            output_data_as_json({"evaluation_id": result_eval_id}, console)

    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except json.JSONDecodeError as e:
        console.print(f"[red]Error:[/red] Invalid JSON in metadata.json: {e}")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except InvalidEvaluationError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print()
        console.print("[yellow]Tip:[/yellow] You must provide one of:")
        console.print("  --eval <eval_id>     (to update an existing evaluation)")
        console.print("  --run-id <run_id>    (to link to an existing training run)")
        console.print("  --env <env>          (environment name, e.g., 'gsm8k' or 'owner/gsm8k')")
        console.print("  [or ensure 'env' or 'env_id' is set in metadata.json]")
        raise typer.Exit(1)
    except KeyError as e:
        console.print(f"[red]Error:[/red] Missing required field: {e}")
        console.print(
            "[yellow]Hint:[/yellow] metadata.json must contain 'env' (or 'env_id') and 'model'"
        )
        raise typer.Exit(1)
    except EvalsAPIError as e:
        console.print(f"[red]API Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


app = typer.Typer(
    cls=DefaultGroup,
    help=(
        "Run evaluations or manage results (list, get, push, samples).\n\n"
        "By default, 'prime eval <environment>' runs 'prime eval run <environment>'."
    ),
    no_args_is_help=True,
)

app.add_typer(subcommands_app, name="")


@app.command("logs", no_args_is_help=True)
def logs_cmd(
    eval_id: str = typer.Argument(..., help="Evaluation id to get logs for"),
    tail: int = typer.Option(
        HOSTED_LOGS_DEFAULT_TAIL_LINES,
        "--tail",
        "-n",
        help="Number of lines to show",
    ),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
    poll_interval: float = typer.Option(
        HOSTED_LOGS_DEFAULT_POLL_INTERVAL_SECONDS,
        "--poll-interval",
        help="Polling interval in seconds when following logs",
    ),
) -> None:
    """Get logs for a hosted evaluation."""
    _display_logs(eval_id, tail, follow, poll_interval=poll_interval)


@app.command("stop", no_args_is_help=True)
def stop_cmd(
    eval_id: str = typer.Argument(..., help="Evaluation id to stop"),
) -> None:
    """Stop a running hosted evaluation."""
    try:
        client = APIClient()
        result = client.patch(f"/hosted-evaluations/{eval_id}/cancel")
        message = result.get("message") or f"Evaluation {eval_id} cancelled."
        console.print(f"[green]✓ {message}[/green]")
        console.print(f"[dim]View results:[/dim] {get_eval_viewer_url(eval_id)}")
    except APIError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc


@app.command(
    "run",
    help="Run an evaluation with API models (default provider = Prime Inference)",
    no_args_is_help=True,
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
        "help_option_names": [],
    },
)
def run_eval_cmd(
    ctx: typer.Context,
    environment: Optional[str] = typer.Argument(
        None,
        help="Environment name/slug or TOML config path",
    ),
    skip_upload: bool = typer.Option(
        False,
        "--skip-upload",
        help="Skip uploading results to Prime Evals Hub (results are uploaded by default)",
    ),
    env_path: Optional[str] = typer.Option(
        None,
        "--env-path",
        help=(
            "Path to the environment directory "
            "(used to locate .prime/.env-metadata.json for upstream resolution)"
        ),
    ),
    hosted: bool = typer.Option(
        False,
        "--hosted",
        help="Run the evaluation on the platform instead of locally",
    ),
    poll_interval: float = typer.Option(
        HOSTED_RUN_DEFAULT_POLL_INTERVAL_SECONDS,
        "--poll-interval",
        help="Polling interval in seconds for hosted evaluation status",
    ),
    follow: bool = typer.Option(
        False,
        "--follow",
        help="Follow hosted evaluation status and stream logs until completion",
    ),
    timeout_minutes: Optional[int] = typer.Option(
        None,
        "--timeout-minutes",
        help="Timeout in minutes for hosted evaluation",
    ),
    allow_sandbox_access: bool = typer.Option(
        False,
        "--allow-sandbox-access",
        help="Allow sandbox read/write access for hosted evaluations",
    ),
    allow_instances_access: bool = typer.Option(
        False,
        "--allow-instances-access",
        help="Allow instance creation and management for hosted evaluations",
    ),
    custom_secrets: Optional[str] = typer.Option(
        None,
        "--custom-secrets",
        help='Custom secrets for hosted eval as JSON (e.g. \'{"API_KEY":"xxx"}\')',
    ),
    eval_name: Optional[str] = typer.Option(
        None,
        "--eval-name",
        help="Custom name for the hosted evaluation",
    ),
) -> None:
    """Run an evaluation with local-first environment resolution."""
    passthrough_args = list(ctx.args)

    if is_help_request(environment or "", passthrough_args):
        print_eval_run_help()
        raise typer.Exit(0)

    if environment is None:
        console.print("[red]Error:[/red] Missing argument 'ENVIRONMENT'.")
        console.print(f"[dim]Example: {EVAL_RUN_EXAMPLE_COMMAND}[/dim]")
        raise typer.Exit(2)

    if environment.startswith("-"):
        console.print("[red]Error:[/red] Environment/config must be the first argument.")
        console.print(f"[dim]Example: {EVAL_RUN_EXAMPLE_COMMAND}[/dim]")
        raise typer.Exit(2)

    env_dir_path = _parse_value_option(passthrough_args, "--env-dir-path", "-p")
    poll_interval_was_provided = (
        ctx.get_parameter_source("poll_interval") == ParameterSource.COMMANDLINE
    )

    if not hosted:
        hosted_only_args = {
            "--follow": follow,
            "--poll-interval": poll_interval_was_provided,
            "--timeout-minutes": timeout_minutes is not None,
            "--allow-sandbox-access": allow_sandbox_access,
            "--allow-instances-access": allow_instances_access,
            "--custom-secrets": custom_secrets is not None,
            "--eval-name": eval_name is not None,
        }
        used_hosted_only_args = [flag for flag, used in hosted_only_args.items() if used]
        if used_hosted_only_args:
            console.print(
                "[red]Error:[/red] hosted-only options require `--hosted`: "
                + ", ".join(used_hosted_only_args)
            )
            raise typer.Exit(1)

    if hosted:
        if _is_config_target(environment):
            console.print(
                "[red]Error:[/red] hosted evaluations require a single environment, "
                "not a TOML config"
            )
            raise typer.Exit(1)

        try:
            platform_slug, environment_id = _resolve_hosted_environment(
                environment,
                env_dir_path=env_dir_path,
                env_path=env_path,
            )
        except APIError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1) from exc
        model = _parse_value_option(passthrough_args, "--model", "-m") or DEFAULT_MODEL
        raw_num_examples = _parse_value_option(passthrough_args, "--num-examples", "-n")
        raw_rollouts = _parse_value_option(
            passthrough_args,
            "--rollouts-per-example",
            "-r",
        )
        raw_env_args = _parse_value_option(passthrough_args, "--env-args", "")

        try:
            num_examples = (
                int(raw_num_examples)
                if raw_num_examples is not None
                else HOSTED_RUN_DEFAULT_NUM_EXAMPLES
            )
            rollouts_per_example = (
                int(raw_rollouts)
                if raw_rollouts is not None
                else HOSTED_RUN_DEFAULT_ROLLOUTS_PER_EXAMPLE
            )
        except ValueError as exc:
            console.print(
                "[red]Error:[/red] --num-examples and --rollouts-per-example must be integers"
            )
            raise typer.Exit(1) from exc

        if num_examples < -1 or rollouts_per_example < 1:
            console.print(
                "[red]Error:[/red] --num-examples must be >= -1 and "
                "--rollouts-per-example must be >= 1"
            )
            raise typer.Exit(1)

        hosted_config = HostedEvalConfig(
            environment_id=environment_id,
            inference_model=model,
            num_examples=num_examples,
            rollouts_per_example=rollouts_per_example,
            env_args=_parse_json_option(raw_env_args, "--env-args"),
            name=eval_name,
            timeout_minutes=timeout_minutes,
            allow_sandbox_access=allow_sandbox_access,
            allow_instances_access=allow_instances_access,
            custom_secrets=_parse_json_option(custom_secrets, "--custom-secrets"),
        )

        try:
            result = _create_hosted_evaluation(hosted_config)
        except APIError as exc:
            console.print(f"[red]Hosted evaluation failed:[/red] {exc}")
            raise typer.Exit(1) from exc

        if follow:
            console.print("[green]✓ Hosted evaluation started[/green]")
            console.print(f"[cyan]Environment:[/cyan] {platform_slug}")
            console.print(f"[cyan]Evaluation ID:[/cyan] {result.evaluation_id}")
            console.print()
            _display_logs(
                result.evaluation_id,
                tail=HOSTED_LOGS_DEFAULT_TAIL_LINES,
                follow=True,
                poll_interval=poll_interval,
            )
            return

        console.print("[green]✓ Hosted evaluation started[/green]")
        console.print(f"[cyan]Environment:[/cyan] {platform_slug}")
        console.print(f"[cyan]Evaluation ID:[/cyan] {result.evaluation_id}")
        console.print(f"[green]View results:[/green] {get_eval_viewer_url(result.evaluation_id)}")
        console.print("[dim]View logs:[/dim] prime eval logs " + result.evaluation_id + " -f")
        return

    run_eval_passthrough(
        environment=environment,
        passthrough_args=passthrough_args,
        skip_upload=skip_upload,
        env_path=env_path,
    )
