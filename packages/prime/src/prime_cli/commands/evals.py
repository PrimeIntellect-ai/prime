import json
import re
from functools import wraps
from pathlib import Path
from typing import Optional

import typer
from prime_evals import EvalsAPIError, EvalsClient, InvalidEvaluationError
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table
from typer.core import TyperGroup

from ..client import APIClient
from ..utils import output_data_as_json
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

    env_field = metadata.get("env_id") or metadata.get("env")
    if not env_field or "model" not in metadata:
        raise ValueError(
            f"Missing required 'env_id' or 'model' field in {directory / 'metadata.json'}"
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
        "eval_name": f"{env_field}-{metadata['model']}",
        "model_name": metadata["model"],
        "env": env_field,
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
    env_slug: Optional[str],
    run_id: Optional[str],
    eval_id: Optional[str],
) -> str:
    format_type, path = _detect_format(config_path)

    if format_type == "json":
        with open(path, "r") as f:
            eval_data = json.load(f)
        console.print(f"[blue]✓ Loaded eval data (JSON format):[/blue] {path}")
    else:
        eval_data = _load_verifiers_format(path)
        console.print(f"[blue]✓ Loaded eval data (verifiers format):[/blue] {path}")

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


@subcommands_app.command("push")
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

    Supports JSON format, verifiers format directory, or auto-discovery.

    Examples:
        prime eval push                                    # Push current dir or auto-discover
        prime eval push outputs/evals/gsm8k--gpt-4/abc123  # Push specific directory
        prime eval push eval.json --run-id abc123          # Push JSON file with run_id
        prime eval push eval.json --eval xyz789            # Push to existing evaluation
        prime eval push --env gsm8k                        # Push with environment
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
            if _has_verifiers_files(current_dir):
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
        console.print(f"[red]Error:[/red] Invalid JSON: {e}")
        raise typer.Exit(1)
    except InvalidEvaluationError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print()
        console.print("[yellow]Tip:[/yellow] You must provide one of:")
        console.print("  --eval <eval_id>     (to update an existing evaluation)")
        console.print("  --run-id <run_id>    (to link to an existing training run)")
        console.print("  --env <env>          (environment name, e.g., 'gsm8k' or 'owner/gsm8k')")
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


app = typer.Typer(
    cls=DefaultGroup,
    help=(
        "Run evaluations or manage results (list, get, push, samples).\n\n"
        "By default, 'prime eval <environment>' runs 'prime eval run <environment>'."
    ),
    no_args_is_help=True,
)

app.add_typer(subcommands_app, name="")


@app.command(
    "run",
    no_args_is_help=True,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def run_eval_cmd(
    ctx: typer.Context,
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
        5, "--num-examples", "-n", help="Number of examples"
    ),
    rollouts_per_example: Optional[int] = typer.Option(
        3, "--rollouts-per-example", "-r", help="Rollouts per example"
    ),
    max_concurrent: Optional[int] = typer.Option(
        32, "--max-concurrent", "-c", help="Max concurrent requests"
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
    save_results: bool = typer.Option(True, "--save-results", "-s", help="Save results to disk"),
    save_every: int = typer.Option(1, "--save-every", "-f", help="Save dataset every n rollouts"),
    save_to_hf_hub: bool = typer.Option(False, "--save-to-hf-hub", "-H", help="Save to HF Hub"),
    hf_hub_dataset_name: Optional[str] = typer.Option(
        None, "--hf-hub-dataset-name", "-D", help="HF Hub dataset name"
    ),
    env_args: Optional[str] = typer.Option(
        None, "--env-args", "-a", help='Environment args as JSON, e.g. \'{"key":"value"}\''
    ),
    api_key_var: Optional[str] = typer.Option(
        None, "--api-key-var", "-k", help="override api key variable instead of using PRIME_API_KEY"
    ),
    api_base_url: Optional[str] = typer.Option(
        None,
        "--api-base-url",
        "-b",
        help=(
            "override api base url variable instead of using prime inference url, "
            "should end in '/v1'"
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
) -> None:
    """
    Run verifiers' vf-eval with Prime Inference.

    Examples:
       prime eval primeintellect/wordle -m openai/gpt-4.1-mini -n 5
       prime eval wordle -m openai/gpt-4.1-mini -n 2 -r 3 -t 1024 -T 0.7
    """
    run_eval(
        environment=environment,
        model=model,
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
        max_concurrent=max_concurrent,
        max_tokens=max_tokens,
        temperature=temperature,
        sampling_args=sampling_args,
        verbose=verbose,
        save_results=save_results,
        save_every=save_every,
        save_to_hf_hub=save_to_hf_hub,
        hf_hub_dataset_name=hf_hub_dataset_name,
        env_args=env_args,
        api_key_var=api_key_var,
        api_base_url=api_base_url,
        skip_upload=skip_upload,
        env_path=env_path,
    )
