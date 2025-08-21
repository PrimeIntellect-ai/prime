"""Prompt utilities for user interaction."""

import typer


def confirm_or_skip(message: str, yes_flag: bool, default: bool = False) -> bool:
    """Show confirmation prompt or skip if --yes flag is provided."""
    if yes_flag:
        return True
    return bool(typer.confirm(message, default=default))
