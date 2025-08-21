"""Shared utilities for CLI commands."""

import json
from typing import Any, Dict, List

import typer
from rich.console import Console


def validate_output_format(output: str, console: Console) -> None:
    """Validate that output format is supported."""
    if output not in ["table", "json"]:
        console.print(f"[red]Error: Invalid output format '{output}'. Supported formats: table, json[/red]")
        raise typer.Exit(1)


def output_data_as_json(data: Any, console: Console) -> None:
    """Output data as formatted JSON."""
    console.print(json.dumps(data, indent=2, default=str))


def confirm_or_skip(message: str, yes_flag: bool, default: bool = False) -> bool:
    """Show confirmation prompt or skip if --yes flag is provided."""
    if yes_flag:
        return True
    return typer.confirm(message, default=default)
