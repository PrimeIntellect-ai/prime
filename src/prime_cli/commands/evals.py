import json
from functools import wraps
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from ..api.evals import EvalsAPIError, EvalsClient
from ..utils import output_data_as_json, validate_output_format

app = typer.Typer(
    help="Manage Prime Evals",
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
    output: str = typer.Option("table", "--output", "-o", help="table|json"),
) -> None:
    """List all evals from the hub endpoint."""
    validate_output_format(output, console)

    client = EvalsClient()
    data = client.list_evals()

    if output == "json":
        output_data_as_json(data, console)
        return

    # Extract evals list from response
    evals = data.get("data", data.get("evals", data)) if isinstance(data, dict) else data

    if not evals:
        console.print("[yellow]No evals found.[/yellow]")
        return

    table = Table(title="Prime Evals")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Model", style="magenta")
    table.add_column("Dataset", style="blue")
    table.add_column("Metrics", style="yellow")

    for e in evals:
        metrics = e.get("metrics", {})
        metrics_str = (
            ", ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()
            )
            if metrics
            else ""
        )

        table.add_row(
            str(e.get("id", "")),
            str(e.get("eval_name", e.get("name", ""))),
            str(e.get("model_name", e.get("model", ""))),
            str(e.get("dataset", "")),
            metrics_str,
        )

    console.print(table)


@app.command("get")
@handle_errors
def get_eval(
    eval_id: str = typer.Argument(..., help="The ID of the eval to retrieve"),
    output: str = typer.Option("json", "--output", "-o", help="json|pretty"),
) -> None:
    """Get details of a specific eval by ID."""
    if output not in ["json", "pretty"]:
        console.print("[red]Error:[/red] output must be 'json' or 'pretty'")
        raise typer.Exit(1)

    client = EvalsClient()
    data = client.get_eval(eval_id)
    format_output(data, output)


@app.command("push")
@handle_errors
def push_eval(
    file: Optional[str] = typer.Argument(None, help="Path to JSON file with eval data"),
    eval_name: Optional[str] = typer.Option(None, "--name", "-n", help="Name of the evaluation"),
    model_name: Optional[str] = typer.Option(
        None, "--model", "-m", help="Name of the model evaluated"
    ),
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d", help="Name of the dataset used"),
    metrics: Optional[str] = typer.Option(
        None, "--metrics", help="Metrics as JSON string or key=value pairs"
    ),
    output: str = typer.Option("json", "--output", "-o", help="table|json|pretty"),
) -> None:
    """
    Push evaluation results to the hub endpoint.

    Example: prime evals push eval.json
    Example: prime evals push --name my-eval --model gpt-4 --dataset mmlu
    """
    if output not in ["json", "pretty", "table"]:
        console.print("[red]Error:[/red] output must be 'json', 'pretty', or 'table'")
        raise typer.Exit(1)

    client = EvalsClient()

    # Load or construct payload
    if file:
        file_path = Path(file)
        if not file_path.exists():
            console.print(f"[red]Error:[/red] File not found: {file}")
            raise typer.Exit(1)
        with open(file_path, "r") as f:
            payload = json.load(f)
    else:
        if not all([eval_name, model_name, dataset]):
            console.print("[red]Error:[/red] Provide --name, --model, and --dataset")
            raise typer.Exit(1)

        payload = {
            "eval_name": eval_name,
            "model_name": model_name,
            "dataset": dataset,
            "metrics": _parse_metrics(metrics) if metrics else {},
        }

    # Push and display result
    result = client.push_eval(payload)

    if output == "json":
        output_data_as_json(result, console)
    elif output == "pretty":
        format_output(result, output)
    else:  # table
        console.print("[green]✓[/green] Eval pushed successfully!")
        for key, label in [
            ("id", "ID"),
            ("eval_name", "Name"),
            ("model_name", "Model"),
            ("dataset", "Dataset"),
        ]:
            if key in result:
                console.print(
                    f"  {label}: [cyan]{result[key]}[/cyan]"
                    if key == "id"
                    else f"  {label}: {result[key]}"
                )


def _parse_metrics(metrics_str: str) -> dict:
    """Parse metrics from JSON or key=value format."""
    try:
        return json.loads(metrics_str)
    except json.JSONDecodeError:
        metrics_dict = {}
        for pair in metrics_str.split(","):
            if "=" in pair:
                key, value = pair.split("=", 1)
                try:
                    metrics_dict[key.strip()] = float(value.strip())
                except ValueError:
                    metrics_dict[key.strip()] = value.strip()
        return metrics_dict


@app.command("delete")
@handle_errors
def delete_eval(
    eval_id: str = typer.Argument(..., help="The ID of the eval to delete"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Delete an eval by ID."""
    if not yes and not typer.confirm(f"Are you sure you want to delete eval {eval_id}?"):
        console.print("[yellow]Cancelled.[/yellow]")
        raise typer.Exit(0)

    client = EvalsClient()
    result = client.delete_eval(eval_id)

    console.print(f"[green]✓[/green] Eval {eval_id} deleted successfully.")
    if isinstance(result, dict) and result.get("message"):
        console.print(f"  {result['message']}")
