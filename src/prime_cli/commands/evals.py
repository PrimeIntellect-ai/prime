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
            "[yellow]Response may contain invalid data. Try --output json to see raw response.[/yellow]"
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
