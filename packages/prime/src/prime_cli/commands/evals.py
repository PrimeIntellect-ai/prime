import asyncio
import json
import re
import time
from functools import wraps
from pathlib import Path
from typing import List, Optional

import typer
from prime_evals import EvalsAPIError, EvalsClient, InvalidEvaluationError
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table
from typer.core import TyperGroup

from ..client import APIClient
from ..core import APIError, AsyncAPIClient
from ..utils import output_data_as_json
from ..utils.display import get_eval_viewer_url
from ..utils.eval_push import load_results_jsonl
from ..utils.hosted_eval import (
    clean_logs,
    get_evaluation,
    get_evaluation_logs,
    get_new_log_lines,
    stop_hosted_evaluation,
)
from ..utils.schemas import EvalStatus
from .env import run_eval

console = Console()


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


@subcommands_app.command("list")
@handle_errors
def list_evals(
    output: str = typer.Option("table", "--output", "-o", help="table|json"),
    skip: int = typer.Option(0, "--skip", help="Number of records to skip"),
    limit: int = typer.Option(50, "--limit", help="Maximum number of records to return"),
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

    try:
        api_client = APIClient()
        client = EvalsClient(api_client)

        data = client.list_evaluations(
            env_name=env,
            skip=skip,
            limit=limit,
        )

        if output == "json":
            output_data_as_json(data, console)
            return

        evals = data.get("evaluations", [])

        if not evals:
            console.print("[yellow]No evaluations found.[/yellow]")
            return

        table = Table(title="Evaluations", show_lines=False)
        table.add_column("Id", style="cyan", no_wrap=True)
        table.add_column("Environment", style="blue")
        table.add_column("Model", style="magenta")
        table.add_column("Status", style="yellow")
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

            table.add_row(
                eval_id if eval_id else "",
                str(env_name)[:30],
                str(e.get("model_name", ""))[:30],
                str(e.get("status", "")),
                str(num_examples),
                str(rollouts_per_example),
            )

        console.print(table)
        total = data.get("total", 0)
        if evals:
            console.print(f"[dim]Total: {total} | Showing {skip + 1}-{skip + len(evals)}[/dim]")
        else:
            console.print(f"[dim]Total: {total}[/dim]")

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
    eval_id: str = typer.Argument(..., help="The id of the evaluation to retrieve"),
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
    eval_id: str = typer.Argument(..., help="The id of the evaluation"),
    page: int = typer.Option(1, "--page", "-p", help="Page number"),
    limit: int = typer.Option(100, "--num", "-n", help="Samples per page"),
    output: str = typer.Option("json", "--output", "-o", help="json|pretty"),
) -> None:
    _validate_output_format(output, ["json", "pretty"])

    api_client = APIClient()
    client = EvalsClient(api_client)
    data = client.get_samples(eval_id, page=page, limit=limit)
    format_output(data, output)


def _fetch_logs(eval_id: str) -> str:
    async def _fetch():
        async with AsyncAPIClient() as client:
            return await get_evaluation_logs(client, eval_id)

    return asyncio.run(_fetch())


def _fetch_eval_status(eval_id: str) -> dict:
    async def _fetch():
        async with AsyncAPIClient() as client:
            return await get_evaluation(client, eval_id)

    return asyncio.run(_fetch())


def _print_eval_status(eval_data: dict) -> None:
    status_str = eval_data.get("status")
    try:
        status = EvalStatus(status_str)
        color = status.color
    except ValueError:
        color = "white"

    eval_id = eval_data.get("evaluation_id", eval_data.get("id", ""))
    console.print(f"[{color}]Status: {status_str}[/{color}]")

    error_message = eval_data.get("error_message")
    if error_message:
        console.print(f"[red]Error: {error_message}[/red]")

    if eval_id:
        viewer_url = get_eval_viewer_url(eval_id, eval_data.get("viewer_url"))
        console.print(f"[dim]View: {viewer_url}[/dim]")


def _display_logs(eval_id: str, tail: int, follow: bool) -> None:
    try:
        if follow:
            console.print(
                f"[dim]Watching logs for evaluation {eval_id}... (Ctrl+C to stop)[/dim]\n"
            )
            last_logs = ""
            consecutive_errors = 0
            no_logs_polls = 0

            while True:
                try:
                    eval_data = _fetch_eval_status(eval_id)
                    status_str = eval_data.get("status")
                    try:
                        status = EvalStatus(status_str)
                    except ValueError:
                        status = None

                    if status and status in EvalStatus.terminal_statuses():
                        console.print()
                        _print_eval_status(eval_data)
                        if status != EvalStatus.COMPLETED:
                            raise typer.Exit(1)
                        return

                    raw_logs = _fetch_logs(eval_id)
                    logs = clean_logs(raw_logs) if raw_logs else ""
                    consecutive_errors = 0

                    if logs and logs != last_logs:
                        for line in get_new_log_lines(last_logs, logs):
                            console.print(line)
                        last_logs = logs
                        no_logs_polls = 0
                    elif not logs:
                        no_logs_polls += 1

                    if no_logs_polls > 0 and no_logs_polls % 6 == 0:
                        console.print(
                            f"[dim]Evaluation status: {status_str} (waiting for logs...)[/dim]"
                        )

                except APIError as e:
                    consecutive_errors += 1
                    err_str = str(e)
                    if "404" in err_str:
                        console.print(
                            f"\n[red]Evaluation {eval_id} not found or has been cancelled.[/red]"
                        )
                        raise typer.Exit(1)
                    if "429" in err_str:
                        if consecutive_errors >= 3:
                            console.print("[yellow]Rate limited. Waiting 30s...[/yellow]")
                            time.sleep(30)
                        else:
                            time.sleep(10)
                        continue
                    raise

                time.sleep(5)
        else:
            eval_data = _fetch_eval_status(eval_id)
            raw_logs = _fetch_logs(eval_id)
            logs = clean_logs(raw_logs) if raw_logs else ""

            if logs:
                lines = logs.splitlines()
                if len(lines) > tail:
                    lines = lines[-tail:]
                for line in lines:
                    console.print(line)
            else:
                console.print("[yellow]No logs available.[/yellow]")

            console.print()
            _print_eval_status(eval_data)

    except KeyboardInterrupt:
        console.print("\n[dim]Stopped watching logs. Evaluation continues running.[/dim]")
        console.print(f"[dim]To stop the evaluation: prime eval stop {eval_id}[/dim]")
    except typer.Exit:
        raise
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


def _load_verifiers_format(directory: Path) -> dict:
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
) -> str:
    path = _validate_eval_path(config_path)
    eval_data = _load_verifiers_format(path)
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
        )

        eval_id = create_response.get("evaluation_id")
        if not eval_id:
            raise ValueError("Failed to get evaluation id from response")

        console.print(f"[green]✓ Created evaluation:[/green] {eval_id}")
        console.print()

    results = eval_data.get("results", [])
    if results:
        console.print(f"[blue]Pushing {len(results)} samples...[/blue]")
        client.push_samples(eval_id, results)
        console.print("[green]✓ Samples pushed successfully[/green]")
        console.print()

    console.print("[blue]Finalizing evaluation...[/blue]")
    finalize_response = client.finalize_evaluation(eval_id, metrics=eval_data.get("metrics"))
    console.print("[green]✓ Evaluation finalized[/green]")
    console.print()

    console.print("[green]✓ Success[/green]")
    console.print(f"[blue]Evaluation id:[/blue] {eval_id}")
    console.print()

    viewer_url = get_eval_viewer_url(eval_id, finalize_response.get("viewer_url"))
    console.print(f"[green]View results at:[/green] {viewer_url}")
    console.print()

    console.print("[dim]CLI commands:[/dim]")
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
    """Launch TUI for viewing eval results (passthrough to vf-tui)."""
    from verifiers.scripts.tui import VerifiersTUI

    env_path = env_dir or "./environments"
    outputs_path = outputs_dir or "./outputs"
    tui_app = VerifiersTUI(env_dir_path=env_path, outputs_dir_path=outputs_path)
    tui_app.run()


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
) -> None:
    """Push evaluation data to Prime Evals.

    The directory must contain metadata.json and results.jsonl files.

    \b
    Examples:
        prime eval push                                    # Push current dir or auto-discover
        prime eval push outputs/evals/gsm8k--gpt-4/abc123  # Push specific directory
        prime eval push --env gsm8k                        # Push with environment override
        prime eval push --eval xyz789                      # Push to existing evaluation
    """
    try:
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
                result_eval_id = _push_single_eval(".", env_id, run_id, eval_id)
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
                    result_eval_id = _push_single_eval(str(eval_dir), env_id, run_id, eval_id)
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

        result_eval_id = _push_single_eval(config_path, env_id, run_id, eval_id)

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
    tail: int = typer.Option(1000, "--tail", "-n", help="Number of lines to show"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
) -> None:
    """Get logs for a hosted evaluation."""
    _display_logs(eval_id, tail, follow)


@app.command("pull", no_args_is_help=True)
def pull_cmd(
    eval_id: str = typer.Argument(..., help="Evaluation id to pull"),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory (defaults to outputs/evals/<env>--<model>/<eval_id>)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing files",
    ),
) -> None:
    """Pull evaluation results in verifiers format.

    Downloads evaluation metadata and results from the platform
    and saves them in the verifiers format (metadata.json + results.jsonl).

    \b
    Examples:
        prime eval pull abc123                           # Pull to default location
        prime eval pull abc123 -o ./my-results           # Pull to custom directory
        prime eval pull abc123 -f                        # Overwrite existing files
    """
    try:
        api_client = APIClient()

        console.print(f"[blue]Fetching evaluation {eval_id}...[/blue]")

        try:
            eval_data = _fetch_eval_status(eval_id)
            status = eval_data.get("status")
            if status != "COMPLETED":
                console.print(f"[red]Cannot pull evaluation with status {status}.[/red]")
                console.print("[dim]Only completed evaluations can be pulled.[/dim]")
                raise typer.Exit(1)
        except APIError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

        try:
            export_data = api_client.get(f"/evaluations/{eval_id}/export")
        except APIError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

        metadata = export_data.get("metadata", {})
        results = export_data.get("results", [])

        # Determine output directory
        if output_dir:
            out_path = Path(output_dir)
        else:
            env_id = metadata.get("env_id", "unknown")
            model = metadata.get("model", "unknown")
            # Sanitize for path
            env_safe = re.sub(r"[^\w\-]", "_", str(env_id))
            model_safe = re.sub(r"[^\w\-]", "_", str(model))
            out_path = Path(f"outputs/evals/{env_safe}--{model_safe}/{eval_id}")

        # Check if directory exists and has files
        if out_path.exists() and not force:
            if (out_path / "metadata.json").exists() or (out_path / "results.jsonl").exists():
                console.print(
                    f"[yellow]Warning:[/yellow] Output directory already exists: {out_path}"
                )
                console.print("[yellow]Use --force to overwrite existing files[/yellow]")
                raise typer.Exit(1)

        # Create directory
        out_path.mkdir(parents=True, exist_ok=True)

        # Write metadata.json
        metadata_path = out_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        console.print(f"[green]✓[/green] Wrote {metadata_path}")

        # Write results.jsonl
        results_path = out_path / "results.jsonl"
        with open(results_path, "w") as f:
            for result in results:
                f.write(json.dumps(result, default=str) + "\n")
        console.print(f"[green]✓[/green] Wrote {results_path} ({len(results)} samples)")

        console.print()
        console.print(f"[green]✓ Successfully pulled evaluation to:[/green] {out_path}")

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("stop", no_args_is_help=True)
def stop_cmd(
    eval_id: str = typer.Argument(..., help="Evaluation id to stop"),
) -> None:
    """Stop a running hosted evaluation."""
    stop_hosted_evaluation(eval_id)


@app.command(
    "run",
    help="Run an evaluation with API models (default provider = Prime Inference)",
    no_args_is_help=True,
)
def run_eval_cmd(
    environment: str = typer.Argument(
        ...,
        help="Environment name (e.g. 'wordle') or slug (e.g. 'primeintellect/wordle')",
    ),
    model: str = typer.Option(
        "openai/gpt-4.1-mini",
        "--model",
        "-m",
        help=(
            "Model to use (e.g. 'openai/gpt-4.1-mini', 'prime-intellect/intellect-3', "
            "see 'prime inference models' for available models)"
        ),
    ),
    num_examples: Optional[int] = typer.Option(
        None, "--num-examples", "-n", help="Number of examples"
    ),
    rollouts_per_example: Optional[int] = typer.Option(
        None, "--rollouts-per-example", "-r", help="Rollouts per example"
    ),
    max_concurrent: Optional[int] = typer.Option(
        32, "--max-concurrent", "-c", help="Max concurrent requests"
    ),
    max_concurrent_generation: Optional[int] = typer.Option(
        None, "--max-concurrent-generation", help="Max concurrent generation requests"
    ),
    max_concurrent_scoring: Optional[int] = typer.Option(
        None, "--max-concurrent-scoring", help="Max concurrent scoring requests"
    ),
    max_tokens: Optional[int] = typer.Option(
        None, "--max-tokens", "-t", help="Max tokens to generate (unset → model default)"
    ),
    temperature: Optional[float] = typer.Option(None, "--temperature", "-T", help="Temperature"),
    sampling_args: Optional[str] = typer.Option(
        None,
        "--sampling-args",
        "-S",
        help='Sampling args as JSON, e.g. \'{"enable_thinking": false, "max_tokens": 256}\'',
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    no_interleave_scoring: bool = typer.Option(
        False, "--no-interleave-scoring", "-N", help="Disable interleaving of scoring"
    ),
    state_columns: Optional[str] = typer.Option(
        None,
        "--state-columns",
        "-C",
        help="Comma-separated list of state columns to save (e.g., 'turn,timing')",
    ),
    save_results: bool = typer.Option(False, "--save-results", "-s", help="Save results to disk"),
    save_every: int = typer.Option(-1, "--save-every", "-f", help="Save dataset every n rollouts"),
    independent_scoring: bool = typer.Option(
        False,
        "--independent-scoring",
        "-R",
        help="Score each rollout individually instead of scoring by group",
    ),
    save_to_hf_hub: bool = typer.Option(False, "--save-to-hf-hub", "-H", help="Save to HF Hub"),
    hf_hub_dataset_name: Optional[str] = typer.Option(
        None, "--hf-hub-dataset-name", "-D", help="HF Hub dataset name"
    ),
    env_args: Optional[str] = typer.Option(
        None, "--env-args", "-a", help='Environment args as JSON, e.g. \'{"key":"value"}\''
    ),
    extra_env_kwargs: Optional[str] = typer.Option(
        None,
        "--extra-env-kwargs",
        "-x",
        help='Extra environment kwargs as JSON, e.g. \'{"key":"value"}\'',
    ),
    env_dir_path: Optional[str] = typer.Option(
        None, "--env-dir-path", "-p", help="Path to environments directory"
    ),
    api_key_var: Optional[str] = typer.Option(
        None, "--api-key-var", "-k", help="Override api key variable instead of using PRIME_API_KEY"
    ),
    api_base_url: Optional[str] = typer.Option(
        None,
        "--api-base-url",
        "-b",
        help=(
            "Override API base URL variable instead of using Prime Inference, should end in '/v1'"
        ),
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
    endpoints_path: Optional[str] = typer.Option(
        None,
        "--endpoints-path",
        "-e",
        help="Path to endpoints.py file with custom endpoint configurations",
    ),
    header: Optional[List[str]] = typer.Option(
        None,
        "--header",
        help="Extra HTTP header for inference API ('Name: Value'). Repeatable.",
    ),
    hosted: bool = typer.Option(
        False,
        "--hosted",
        help="Run evaluation on the platform instead of locally",
    ),
    poll_interval: float = typer.Option(
        10.0,
        "--poll-interval",
        help="Polling interval in seconds for hosted evaluation status",
    ),
    follow: bool = typer.Option(
        False,
        "--follow",
        help="Follow hosted evaluation and stream logs until completion",
    ),
    timeout_minutes: Optional[int] = typer.Option(
        None,
        "--timeout-minutes",
        help="Timeout in minutes for hosted evaluation (default: 120, max: 1440)",
    ),
    allow_sandbox_access: bool = typer.Option(
        False,
        "--allow-sandbox-access",
        help="Allow sandbox read/write access for hosted evaluations",
    ),
    allow_instances_access: bool = typer.Option(
        False,
        "--allow-instances-access",
        help="Allow pod/instance creation and management for hosted evaluations",
    ),
    custom_secrets: Optional[str] = typer.Option(
        None,
        "--custom-secrets",
        help='Custom secrets for hosted eval as JSON (e.g., \'{"API_KEY": "xxx"}\')',
    ),
    eval_name: Optional[str] = typer.Option(
        None,
        "--eval-name",
        help="Custom name for the hosted evaluation",
    ),
) -> None:
    """
    Run verifiers' vf-eval with Prime Inference (local) or on the platform (--hosted).

    \b
    Examples:
        prime eval run primeintellect/wordle -m openai/gpt-4.1-mini -n 5
        prime eval run wordle -m openai/gpt-4.1-mini -n 2 -r 3 -t 1024 -T 0.7
        prime eval run primeintellect/gsm8k --hosted -m openai/gpt-4.1-mini -n 10
        prime eval run primeintellect/gsm8k --hosted -f  # Follow logs until completion
    """
    run_eval(
        environment=environment,
        model=model,
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
        max_concurrent=max_concurrent,
        max_concurrent_generation=max_concurrent_generation,
        max_concurrent_scoring=max_concurrent_scoring,
        max_tokens=max_tokens,
        temperature=temperature,
        sampling_args=sampling_args,
        verbose=verbose,
        no_interleave_scoring=no_interleave_scoring,
        state_columns=state_columns,
        save_results=save_results,
        save_every=save_every,
        independent_scoring=independent_scoring,
        save_to_hf_hub=save_to_hf_hub,
        hf_hub_dataset_name=hf_hub_dataset_name,
        env_args=env_args,
        extra_env_kwargs=extra_env_kwargs,
        env_dir_path=env_dir_path,
        api_key_var=api_key_var,
        api_base_url=api_base_url,
        skip_upload=skip_upload,
        env_path=env_path,
        endpoints_path=endpoints_path,
        headers=header,
        hosted=hosted,
        poll_interval=poll_interval,
        follow=follow,
        timeout_minutes=timeout_minutes,
        allow_sandbox_access=allow_sandbox_access,
        allow_instances_access=allow_instances_access,
        custom_secrets=custom_secrets,
        eval_name=eval_name,
    )
