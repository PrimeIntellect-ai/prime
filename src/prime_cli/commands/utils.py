"""Utility functions for sandbox commands."""

import os
from typing import Optional, Tuple

from rich.console import Console

console = Console()





def _parse_cp_arg(arg: str) -> Tuple[Optional[str], str]:
    """Parse cp-style arg: either "<sandbox-id>:<path>" or local path.

    Returns (sandbox_id, path). sandbox_id is None if local.
    """
    if ":" in arg and not arg.startswith(":"):
        sandbox_id, path = arg.split(":", 1)
        if sandbox_id:
            return sandbox_id, path
    return None, arg


def _expand_home_in_path(path: str) -> str:
    """
    Safely expand $HOME to /sandbox-workspace in paths.
    Only allows $HOME expansion for security - no other environment variables.
    """
    if "$HOME" in path:
        # Replace $HOME with the sandbox workspace path
        expanded_path = path.replace("$HOME", "/sandbox-workspace")
        return expanded_path
    return path
