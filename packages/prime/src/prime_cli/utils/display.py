"""Display utilities for table and JSON output."""

import json
from typing import Any, Dict, List, Tuple

import typer
from rich.console import Console
from rich.table import Table


def validate_output_format(output: str, console: Console) -> None:
    """Validate that output format is supported."""
    if output not in ["table", "json"]:
        console.print(
            f"[red]Error: Invalid output format '{output}'. Supported formats: table, json[/red]"
        )
        raise typer.Exit(1)


def output_data_as_json(data: Any, console: Console) -> None:
    """Output data as formatted JSON."""
    console.print(json.dumps(data, indent=2, default=str))


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
