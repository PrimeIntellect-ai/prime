"""Helpers for commands that persist CLI configuration."""

import typer
from rich.markup import escape

from prime_cli.core import Config

from .plain import get_console


def require_persistent_context() -> None:
    """Reject config writes while a temporary ``--context`` is selected."""
    context = Config().context_override
    if context is None:
        return

    console = get_console(stderr=True)
    safe_context = escape(context)
    console.print(f"[red]Error:[/red] Temporary context '{safe_context}' is read-only.")
    console.print(f"[dim]First run: prime config use {safe_context}; then retry without -c.[/dim]")
    raise typer.Exit(1)
