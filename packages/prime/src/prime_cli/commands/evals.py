import json
import re
from functools import wraps
from pathlib import Path
from typing import Optional

import typer
from prime_core import APIClient
from prime_evals import EvalsAPIError, EvalsClient, InvalidEvaluationError
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from ..utils import output_data_as_json

app = typer.Typer(
    help="Manage evaluations (push, list, and view evaluation results)",
    no_args_is_help=True,
)
console = Console()


def handle_errors(func):
    """Decorator to handle common errors in eval commands."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
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


@app.command("list")
@handle_errors
def list_evals(
    output: str = typer.Option("table", "--output", "-o", help="table|json"),
    skip: int = typer.Option(0, "--skip", help="Number of records to skip"),
    limit: int = typer.Option(50, "--limit", help="Maximum number of records to return"),
) -> None:
    """List evaluations."""
    _validate_output_format(output, ["table", "json"])

    try:
        api_client = APIClient()
        client = EvalsClient(api_client)
        data = client.list_evaluations(
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

        table = Table(title="Evaluations")
        table.add_column("ID", style="cyan")
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


@app.command("get")
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


@app.command("samples")
@handle_errors
def get_samples(
    eval_id: str = typer.Argument(..., help="The ID of the evaluation"),
    page: int = typer.Option(1, "--page", "-p", help="Page number"),
    limit: int = typer.Option(100, "--limit", "-l", help="Samples per page"),
    output: str = typer.Option("json", "--output", "-o", help="json|pretty"),
) -> None:
    _validate_output_format(output, ["json", "pretty"])

    api_client = APIClient()
    client = EvalsClient(api_client)
    data = client.get_samples(eval_id, page=page, limit=limit)
    format_output(data, output)


def _load_verifiers_format(directory: Path) -> dict:
    with open(directory / "metadata.json") as f:
        metadata = json.load(f)

    if "env" not in metadata or "model" not in metadata:
        raise ValueError(
            f"Missing required 'env' or 'model' field in {directory / 'metadata.json'}"
        )

    results = []
    with open(directory / "results.jsonl") as f:
        for line in f:
            if line := line.strip():
                try:
                    sample = json.loads(line)
                    if "id" in sample and "example_id" not in sample:
                        sample["example_id"] = sample["id"]
                    results.append(sample)
                except json.JSONDecodeError:
                    continue

    avg_pattern = re.compile(r"^avg_(.+)$")
    metrics = {}
    metadata_copy = {}
    for key, value in metadata.items():
        if match := avg_pattern.match(key):
            metrics[match.group(1)] = value
        else:
            metadata_copy[key] = value

    return {
        "eval_name": f"{metadata['env']}-{metadata['model']}",
        "model_name": metadata["model"],
        "env": metadata["env"],
        "metrics": metrics,
        "metadata": metadata_copy,
        "results": results,
    }


def _has_verifiers_files(directory: Path) -> bool:
    return (directory / "metadata.json").exists() and (directory / "results.jsonl").exists()


def _detect_format(path_str: str) -> tuple[str, Path]:
    path = Path(path_str)

    if path.is_file():
        return ("json", path)
    if path.is_dir():
        if _has_verifiers_files(path):
            return ("verifiers", path)
        raise ValueError(f"Directory {path} missing metadata.json or results.jsonl")
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
            if run_dir.is_dir() and _has_verifiers_files(run_dir):
                eval_dirs.append(run_dir)

    return sorted(eval_dirs)


def _push_single_eval(
    config_path: str,
    env_id: Optional[str],
    run_id: Optional[str],
) -> str:
    format_type, path = _detect_format(config_path)

    if format_type == "json":
        with open(path, "r") as f:
            eval_data = json.load(f)
        console.print(f"[blue]✓ Loaded eval data (JSON format):[/blue] {path}")
    else:
        eval_data = _load_verifiers_format(path)
        console.print(f"[blue]✓ Loaded eval data (verifiers format):[/blue] {path}")

    detected_env = eval_data.get("env")
    if not env_id and detected_env and not run_id:
        env_id = detected_env

    environments = None
    if env_id and not run_id:
        environments = [{"id": env_id}]

    console.print()

    api_client = APIClient()
    client = EvalsClient(api_client)

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


@app.command("push")
@handle_errors
def push_eval(
    config_path: Optional[str] = typer.Argument(
        None,
        help=(
            "Path to eval config JSON file or directory with metadata.json/results.jsonl. "
            "If not provided, auto-discovers from outputs/evals/"
        ),
    ),
    env_id: Optional[str] = typer.Option(
        None,
        "--env",
        "-e",
        help="Environment name (e.g., 'gsm8k' or 'd42me/gsm8k')",
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        "-r",
        help="Link to existing training run id",
    ),
    output: str = typer.Option("pretty", "--output", "-o", help="json|pretty"),
) -> None:
    """Push evaluation data to Prime Evals.

    Supports JSON format, verifiers format directory, or auto-discovery.

    Examples:
        prime eval push                                    # Push current dir or auto-discover
        prime eval push outputs/evals/gsm8k--gpt-4/abc123  # Push specific directory
        prime eval push eval.json --run-id abc123          # Push JSON file
    """
    try:
        if config_path is None:
            current_dir = Path(".")
            if _has_verifiers_files(current_dir):
                eval_id = _push_single_eval(".", env_id, run_id)
                if output == "json":
                    console.print()
                    output_data_as_json({"evaluation_id": eval_id}, console)
                return

            eval_dirs = _discover_eval_outputs()
            if not eval_dirs:
                console.print("[red]Error:[/red] No evaluation outputs found")
                console.print(
                    "[yellow]Hint:[/yellow] Run from a directory with "
                    "metadata.json/results.jsonl or outputs/evals/"
                )
                raise typer.Exit(1)

            console.print(f"[blue]Found {len(eval_dirs)} evaluation(s) to push:[/blue]")
            for eval_dir in eval_dirs:
                console.print(f"  - {eval_dir}")
            console.print()

            results = []
            for eval_dir in eval_dirs:
                try:
                    eval_id = _push_single_eval(str(eval_dir), env_id, run_id)
                    results.append({"path": str(eval_dir), "eval_id": eval_id, "status": "success"})
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

        eval_id = _push_single_eval(config_path, env_id, run_id)

        if output == "json":
            console.print()
            output_data_as_json({"evaluation_id": eval_id}, console)

    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except json.JSONDecodeError as e:
        console.print(f"[red]Error:[/red] Invalid JSON: {e}")
        raise typer.Exit(1)
    except InvalidEvaluationError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print()
        console.print("[yellow]Tip:[/yellow] You must provide one of:")
        console.print("  --run-id <run_id>  (to link to an existing training run)")
        console.print("  --env <env>        (environment name, e.g., 'gsm8k' or 'd42me/gsm8k')")
        console.print("  [or use verifiers format with 'env' in metadata.json for auto-detection]")
        raise typer.Exit(1)
    except KeyError as e:
        console.print(f"[red]Error:[/red] Missing required field in config: {e}")
        console.print("[yellow]Hint:[/yellow] See examples/eval_example.json for required fields")
        raise typer.Exit(1)
    except EvalsAPIError as e:
        console.print(f"[red]API Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
