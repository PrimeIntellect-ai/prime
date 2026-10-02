"""Display utilities for table and JSON output."""

import json
from typing import Any, Dict, List, Optional, Tuple

import typer
from rich.console import Console
from rich.table import Table

from prime_cli.core import Config


def legacy_output_option(short: bool = True) -> Any:
    """Hidden `--output` option, kept for one release as an alias for `--json`."""
    names = ("--output", "-o") if short else ("--output",)
    return typer.Option(None, *names, hidden=True, help="Deprecated: use --json")


def resolve_output_format(
    as_json: bool, output: Optional[str], console: Console, default: str = "table"
) -> str:
    """Return the output format chosen by `--json` or the deprecated `--output`."""
    if as_json:
        return "json"
    if output is None:
        return default
    if output not in (default, "json"):
        console.print(
            f"[red]Error: Invalid output format '{output}'. "
            f"Supported formats: {default}, json[/red]"
        )
        raise typer.Exit(1)
    return output


def output_data_as_json(data: Any, console: Console) -> None:
    """Output data as formatted JSON.

    `soft_wrap=True` disables Rich's terminal-width wrapping so long string
    values don't get a literal newline injected in the middle and break
    parsers (e.g. `prime train usage` run names regularly exceed 80 chars).
    """
    console.print(
        json.dumps(data, indent=2, default=str),
        markup=False,
        highlight=False,
        soft_wrap=True,
    )


def build_table(title: str, columns: List[Tuple[str, str]], show_lines: bool = True) -> Table:
    """
    Build a Rich table with standard styling.

    Args:
        title: Table title
        columns: List of (header, style) tuples
        show_lines: Whether to show row separator lines
    """
    table = Table(title=title, show_lines=show_lines)
    for header, style in columns:
        table.add_column(header, style=style, no_wrap=(header == "ID"))
    return table


def status_color(status: str, mapping: Dict[str, str], default: str = "white") -> str:
    """Get color for status based on mapping with fallback to default."""
    return mapping.get(status, default)


def get_eval_viewer_url(evaluation_id: str) -> str:
    """Build the dashboard URL for an evaluation."""
    frontend_url = Config().frontend_url.rstrip("/")
    return f"{frontend_url}/dashboard/evaluations/{evaluation_id}"


# Common status color mappings
SANDBOX_STATUS_COLORS = {
    "PENDING": "yellow",
    "PROVISIONING": "yellow",
    "RUNNING": "green",
    "PAUSED": "blue",
    "ERROR": "red",
    "TERMINATED": "white",
    "TIMEOUT": "white",
}

POD_STATUS_COLORS = {
    "ACTIVE": "green",
    "PENDING": "yellow",
    "ERROR": "red",
    "INSTALLING": "yellow",
}

STOCK_STATUS_COLORS = {
    "High": "green",
    "Medium": "yellow",
    "Low": "red",
}

DISK_STATUS_COLORS = {
    "ACTIVE": "green",
    "PROVISIONING": "yellow",
    "PENDING": "yellow",
    "STOPPED": "blue",
    "ERROR": "red",
    "TERMINATED": "white",
}

DEPLOYMENT_STATUS_COLORS = {
    "NOT_DEPLOYED": "white",
    "DEPLOYING": "yellow",
    "DEPLOYED": "green",
    "UNLOADING": "yellow",
    "DEPLOY_FAILED": "red",
    "UNLOAD_FAILED": "red",
}
