import json
from functools import wraps
from typing import Optional

import typer
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from ..api.evals import EvalsAPIError, EvalsClient
from ..utils import output_data_as_json, validate_output_format

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
    else:  # pretty
        syntax = Syntax(json.dumps(data, indent=2), "json", theme="monokai")
        console.print(syntax)


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
        client = EvalsClient()
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
                eval_id[:8] if eval_id else "",
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
        client = EvalsClient()
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
        client = EvalsClient()
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

    See examples/eval_example.json for a complete example.
    """
    try:
        with open(config_file, "r") as f:
            eval_data = json.load(f)

        console.print(f"[blue]📖 Loaded eval data from:[/blue] {config_file}")
        console.print(f"[dim]   Name: {eval_data.get('eval_name', 'N/A')}[/dim]")
        console.print(f"[dim]   Model: {eval_data.get('model_name', 'N/A')}[/dim]")
        console.print(f"[dim]   Dataset: {eval_data.get('dataset', 'N/A')}[/dim]")
        console.print(f"[dim]   Results: {len(eval_data.get('results', []))} samples[/dim]")
        console.print()

        client = EvalsClient()

        console.print("[blue]📤 Creating evaluation...[/blue]")
        create_response = client.create_evaluation(
            name=eval_data["eval_name"],
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

        console.print(f"[green]✅ Created evaluation:[/green] {eval_id}")
        console.print()

        if "results" in eval_data and eval_data["results"]:
            console.print(f"[blue]📤 Pushing {len(eval_data['results'])} samples...[/blue]")
            client.push_samples(eval_id, eval_data["results"])
            console.print("[green]✅ Samples pushed successfully[/green]")
            console.print()

        console.print("[blue]🏁 Finalizing evaluation...[/blue]")
        finalize_response = client.finalize_evaluation(eval_id, metrics=eval_data.get("metrics"))
        console.print("[green]✅ Evaluation finalized[/green]")
        console.print()

        console.print("[green]🎉 Success[/green]")
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
