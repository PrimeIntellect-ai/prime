from __future__ import annotations

import datetime as _dt

import typer
from rich.console import Console
from rich.table import Table

from ..api.inference import InferenceAPIError, InferenceClient
from ..utils import output_data_as_json, validate_output_format

app = typer.Typer(
    help="Run and manage Prime Inference\n\n"
    "Use `prime env eval` for environment evals with Prime Inference."
)
console = Console()


def _fmt_created(val: str) -> str:
    """Format created timestamp for display"""
    try:
        # if epoch seconds
        ts = int(val)
        return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return val or ""


@app.command("models")
def list_models(
    output: str = typer.Option("table", "--output", "-o", help="table|json"),
) -> None:
    """List available models from Prime Inference (/v1/models)."""
    validate_output_format(output, console)
    try:
        client = InferenceClient()
        data = client.list_models()

        # Expect OpenAI-style: {"object":"list","data":[{"id":..., ...}, ...]}
        models = []
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                models = data["data"]
            elif "models" in data and isinstance(data["models"], list):
                models = data["models"]  # be liberal in what we accept
        elif isinstance(data, list):
            models = data

        if output == "json":
            output_data_as_json(data, console)
            return

        if not models:
            console.print("[yellow]No models returned.[/yellow]")
            return

        table = Table(title="Prime Inference Available Models")
        table.add_column("id", style="cyan")
        table.add_column("created", style="magenta")

        for m in models:
            mid = str(m.get("id", ""))
            created = _fmt_created(str(m.get("created", "")))
            table.add_row(mid, created)

        console.print(table)

    except InferenceAPIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        raise typer.Exit(1)

