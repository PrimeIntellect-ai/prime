import asyncio
import json
from functools import wraps
from pathlib import Path
from typing import List, Optional

import typer
from openai import AsyncOpenAI
from prime_core import APIClient, Config
from prime_evals import EvalsAPIError, EvalsClient, InvalidEvaluationError
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from ..utils import output_data_as_json, validate_output_format
from ..utils.eval_runner import run_and_push_eval

app = typer.Typer(
    help="Run and manage Prime Evals (in closed beta, requires prime eval permissions)",
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


def format_output(data: dict, output: str) -> None:
    """Format and print output based on the output format."""
    if output == "json":
        output_data_as_json(data, console)
    else:
        syntax = Syntax(json.dumps(data, indent=2), "json", theme="monokai")
        console.print(syntax)


@app.command("run")
@handle_errors
def run_eval(
    environments: List[str] = typer.Argument(..., help="Environment IDs to evaluate"),
    model: str = typer.Option(
        "meta-llama/llama-3.1-70b-instruct",
        "--model",
        "-m",
        help="Model to use (e.g. 'meta-llama/llama-3.1-70b-instruct')",
    ),
    num_examples: int = typer.Option(
        5, "--num-examples", "-n", help="Number of examples per environment"
    ),
    rollouts_per_example: int = typer.Option(
        3, "--rollouts-per-example", "-r", help="Rollouts per example"
    ),
    max_concurrent: int = typer.Option(
        32, "--max-concurrent", "-c", help="Max concurrent requests"
    ),
    max_tokens: Optional[int] = typer.Option(
        None, "--max-tokens", "-t", help="Max tokens to generate"
    ),
    temperature: Optional[float] = typer.Option(None, "--temperature", "-T", help="Temperature"),
    sampling_args: Optional[str] = typer.Option(
        None,
        "--sampling-args",
        "-S",
        help="Sampling args as JSON, e.g. '{\"enable_thinking\": false}'",
    ),
    save_to_hub: bool = typer.Option(
        True, "--save-to-hub/--no-save-to-hub", help="Save results to Environment Hub"
    ),
    eval_name: Optional[str] = typer.Option(
        None, "--eval-name", help="Evaluation name (auto-generated if not provided)"
    ),
    env_args: Optional[str] = typer.Option(
        None, "--env-args", "-a", help='Environment args as JSON, e.g. \'{"key":"value"}\''
    ),
    api_base_url: Optional[str] = typer.Option(
        None,
        "--api-base-url",
        "-b",
        help="Override API base URL (defaults to Prime Inference)",
    ),
    env_dir_path: str = typer.Option(
        "./environments", "--env-dir-path", "-p", help="Path to environments directory"
    ),
) -> None:
    """Run evaluation on one or more environments using Prime Inference.

    Evaluates environments and optionally pushes results to Environment Hub.

    Examples:
        # Single environment
        prime evals run gsm8k -n 10 -m meta-llama/llama-3.1-70b-instruct

        # Multiple environments
        prime evals run gsm8k wordle math500 -n 100 -r 5

        # Without Hub push
        prime evals run gsm8k --no-push-to-hub
    """
    config = Config()

    api_key = config.api_key
    if not api_key:
        console.print(
            "[red]No API key configured.[/red] "
            "Run [bold]prime login[/bold] or [bold]prime config set-api-key[/bold]."
        )
        raise typer.Exit(1)

    if api_base_url:
        chosen_base = api_base_url.rstrip("/")
    else:
        inference_base_url = (config.inference_url or "").strip()
        if not inference_base_url:
            console.print(
                "[red]Inference URL not configured.[/red] Check [bold]prime config view[/bold]."
            )
            raise typer.Exit(1)
        chosen_base = inference_base_url.rstrip("/")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=chosen_base,
    )

    env_args_dict = {}
    if env_args:
        try:
            parsed_args = json.loads(env_args)
            # If single dict, apply to all envs
            is_flat_dict = isinstance(parsed_args, dict) and not any(
                env_id in parsed_args for env_id in environments
            )
            if is_flat_dict:
                env_args_dict = {env_id: parsed_args for env_id in environments}
            else:
                env_args_dict = parsed_args
        except json.JSONDecodeError as e:
            console.print(f"[red]Invalid JSON in --env-args:[/red] {e}")
            raise typer.Exit(1)

    merged_sampling_args = {}
    if sampling_args:
        try:
            merged_sampling_args.update(json.loads(sampling_args))
        except json.JSONDecodeError as e:
            console.print(f"[red]Invalid JSON in --sampling-args:[/red] {e}")
            raise typer.Exit(1)

    if max_tokens is not None:
        merged_sampling_args["max_tokens"] = max_tokens
    if temperature is not None:
        merged_sampling_args["temperature"] = temperature

    console.print(f"[blue]Running evaluation on {len(environments)} environment(s)...[/blue]")
    console.print(f"[dim]Model: {model}[/dim]")
    console.print(f"[dim]Environments: {', '.join(environments)}[/dim]")
    console.print()

    try:
        result = asyncio.run(
            run_and_push_eval(
                environments=environments,
                model=model,
                client=client,
                num_examples=num_examples,
                rollouts_per_example=rollouts_per_example,
                max_concurrent=max_concurrent,
                env_args_dict=env_args_dict,
                sampling_args=merged_sampling_args if merged_sampling_args else None,
                save_to_hub=save_to_hub,
                eval_name=eval_name,
                framework="prime-cli",
                env_dir_path=env_dir_path,
            )
        )

        console.print("[green]✓ Evaluation complete![/green]")
        console.print()

        table = Table(title="Evaluation Results")
        table.add_column("Environment", style="cyan")
        table.add_column("Avg Reward", style="green")
        table.add_column("Std Reward", style="yellow")
        table.add_column("Samples", style="white")
        if save_to_hub:
            table.add_column("Eval ID", style="magenta")

        for env_id in environments:
            metrics = result["metrics"][env_id]
            row = [
                env_id,
                f"{metrics['avg_reward']:.3f}",
                f"{metrics['std_reward']:.3f}",
                str(metrics["num_samples"]),
            ]
            if save_to_hub and env_id in result["eval_ids"]:
                row.append(result["eval_ids"][env_id][:16] + "...")

            table.add_row(*row)

        console.print(table)

        console.print()
        for env_id in environments:
            metrics = result["metrics"][env_id]
            other_metrics = {
                k: v for k, v in metrics.items() if k.startswith("avg_") and k != "avg_reward"
            }
            if other_metrics:
                console.print(f"[cyan]{env_id}[/cyan] additional metrics:")
                for metric_name, value in other_metrics.items():
                    display_name = metric_name.replace("avg_", "")
                    console.print(f"  {display_name}: {value:.3f}")

        if save_to_hub:
            console.print()
            console.print("[dim]View results: prime evals list[/dim]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("list")
@handle_errors
def list_evals(
    environment_id: Optional[str] = typer.Option(
        None, "--environment-id", "-e", help="Filter by environment ID"
    ),
    suite_id: Optional[str] = typer.Option(None, "--suite-id", "-s", help="Filter by suite ID"),
    output: str = typer.Option("table", "--output", "-o", help="table|json"),
    skip: int = typer.Option(0, "--skip", help="Number of records to skip"),
    limit: int = typer.Option(50, "--limit", help="Maximum number of records to return"),
) -> None:
    """List evaluations."""
    validate_output_format(output, console)

    try:
        api_client = APIClient()
        client = EvalsClient(api_client)
        data = client.list_evaluations(
            environment_id=environment_id,
            suite_id=suite_id,
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
        table.add_column("Name", style="green")
        table.add_column("Type", style="blue")
        table.add_column("Model", style="magenta")
        table.add_column("Status", style="yellow")
        table.add_column("Samples", style="white")

        for e in evals:
            eval_id = str(e.get("evaluation_id", e.get("id", "")))
            table.add_row(
                eval_id if eval_id else "",
                str(e.get("name", ""))[:40],
                str(e.get("eval_type", ""))[:20],
                str(e.get("model_name", ""))[:30],
                str(e.get("status", "")),
                str(e.get("total_samples", 0)),
            )

        console.print(table)
        if len(evals) > 0:
            start = skip + 1
            end = skip + len(evals)
            console.print(f"[dim]Total: {data.get('total', 0)} | Showing {start}-{end}[/dim]")
        else:
            console.print(f"[dim]Total: {data.get('total', 0)}[/dim]")

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
    """Get details of a specific evaluation by ID."""
    if output not in ["json", "pretty"]:
        console.print("[red]Error:[/red] output must be 'json' or 'pretty'")
        raise typer.Exit(1)

    try:
        api_client = APIClient()
        client = EvalsClient(api_client)
        data = client.get_evaluation(eval_id)
        format_output(data, output)
    except EvalsAPIError as e:
        console.print(f"[red]API Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("samples")
@handle_errors
def get_samples(
    eval_id: str = typer.Argument(..., help="The ID of the evaluation"),
    page: int = typer.Option(1, "--page", "-p", help="Page number"),
    limit: int = typer.Option(100, "--limit", "-l", help="Samples per page"),
    output: str = typer.Option("json", "--output", "-o", help="json|pretty"),
) -> None:
    """Get samples for a specific evaluation."""
    if output not in ["json", "pretty"]:
        console.print("[red]Error:[/red] output must be 'json' or 'pretty'")
        raise typer.Exit(1)

    try:
        api_client = APIClient()
        client = EvalsClient(api_client)
        data = client.get_samples(eval_id, page=page, limit=limit)
        format_output(data, output)
    except EvalsAPIError as e:
        console.print(f"[red]API Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("push")
@handle_errors
def push_eval(
    config_file: str = typer.Argument(..., help="Path to eval config JSON file"),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        "-r",
        help="Link to existing training run id",
    ),
    env_hub_id: Optional[str] = typer.Option(
        None,
        "--env-hub-id",
        "-e",
        help="Environment Hub id",
    ),
    env_metadata_path: Optional[str] = typer.Option(
        None,
        "--env-metadata",
        help="Path to .env-metadata.json file (overrides default lookup)",
    ),
    output: str = typer.Option("pretty", "--output", "-o", help="json|pretty"),
) -> None:
    """Push evaluation data from a JSON config file.

    The config file should contain:
    - eval_name: Name of the evaluation
    - model_name: Model used
    - dataset: Dataset name
    - metrics: Dictionary of metrics
    - metadata: Dictionary of metadata
    - results: List of result samples

    Either --run-id or --env-hub-id must be provided:
    - Use --run-id to link to an existing training run
    - Use --env-hub-id with a hub environment ID

    Examples:
        prime evals push eval.json --run-id abc123
        prime evals push eval.json --env-hub-id my-env-hub-id
        prime evals push eval.json --env-metadata environments/gsm8k/.env-metadata.json
    """
    try:
        with open(config_file, "r") as f:
            eval_data = json.load(f)

        console.print(f"[blue]✓ Loaded eval data from:[/blue] {config_file}")
        console.print(f"[dim]   Name: {eval_data.get('eval_name', 'N/A')}[/dim]")
        console.print(f"[dim]   Model: {eval_data.get('model_name', 'N/A')}[/dim]")
        console.print(f"[dim]   Dataset: {eval_data.get('dataset', 'N/A')}[/dim]")
        console.print(f"[dim]   Results: {len(eval_data.get('results', []))} samples[/dim]")

        environments = None

        if env_hub_id and not run_id:
            metadata_path = None

            if env_metadata_path:
                metadata_path = Path(env_metadata_path)
            else:
                env_name = env_hub_id.replace("-", "_")
                default_path = Path(f"environments/{env_name}/.env-metadata.json")
                if default_path.exists():
                    metadata_path = default_path

            if metadata_path and metadata_path.exists():
                try:
                    with open(metadata_path) as f:
                        hub_metadata = json.load(f)
                        resolved_env_id = hub_metadata.get("environment_id")
                        if resolved_env_id:
                            short_id = resolved_env_id[:16]
                            console.print(
                                f"[blue]✓ Found environment metadata:[/blue] {short_id}..."
                            )
                            environments = [{"id": resolved_env_id}]
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not load {metadata_path}: {e}[/yellow]")

            if not environments:
                environments = [{"id": env_hub_id}]
                console.print(f"[blue]Using environment hub ID:[/blue] {env_hub_id}")

        console.print()

        api_client = APIClient()
        client = EvalsClient(api_client)

        console.print("[blue]Creating evaluation...[/blue]")
        create_response = client.create_evaluation(
            name=eval_data["eval_name"],
            environments=environments,
            run_id=run_id,
            model_name=eval_data.get("model_name"),
            dataset=eval_data.get("dataset"),
            framework=eval_data.get("metadata", {}).get("framework", "verifiers"),
            task_type=eval_data.get("metadata", {}).get("task_type"),
            metadata=eval_data.get("metadata"),
            metrics=eval_data.get("metrics"),
            tags=eval_data.get("tags", []),
        )

        eval_id = create_response.get("evaluation_id") or create_response.get("id")
        if not eval_id:
            console.print("[red]Error:[/red] Failed to get evaluation ID from response")
            raise typer.Exit(1)

        console.print(f"[green]✓ Created evaluation:[/green] {eval_id}")
        console.print()

        if "results" in eval_data and eval_data["results"]:
            console.print(f"[blue]Pushing {len(eval_data['results'])} samples...[/blue]")
            client.push_samples(eval_id, eval_data["results"])
            console.print("[green]✓ Samples pushed successfully[/green]")
            console.print()

        console.print("[blue]Finalizing evaluation...[/blue]")
        finalize_response = client.finalize_evaluation(eval_id, metrics=eval_data.get("metrics"))
        console.print("[green]✓ Evaluation finalized[/green]")
        console.print()

        console.print("[green]✓ Success[/green]")
        console.print(f"[blue]Evaluation ID:[/blue] {eval_id}")
        console.print()
        console.print("[dim]View your evaluation:[/dim]")
        console.print(f"  prime evals get {eval_id}")
        console.print(f"  prime evals samples {eval_id}")

        if output == "json":
            console.print()
            output_data_as_json(
                {
                    "evaluation_id": eval_id,
                    "create_response": create_response,
                    "finalize_response": finalize_response,
                },
                console,
            )

    except FileNotFoundError:
        console.print(f"[red]Error:[/red] File not found: {config_file}")
        raise typer.Exit(1)
    except json.JSONDecodeError as e:
        console.print(f"[red]Error:[/red] Invalid JSON in config file: {e}")
        raise typer.Exit(1)
    except InvalidEvaluationError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print()
        console.print("[yellow]Tip:[/yellow] Either provide:")
        console.print("  --run-id <run_id>  (to link to an existing run)")
        console.print("  --env-hub-id <env_id>  (for environment from hub)")
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
