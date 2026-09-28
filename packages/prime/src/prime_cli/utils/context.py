"""Helpers for commands that persist CLI configuration."""

from pathlib import Path
from typing import Optional

import typer
from rich.markup import escape

from prime_cli.core import Config
from prime_cli.core.config import LOCAL_CONTEXT_FILE, read_local_context, write_local_context

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


def local_context_target(config: Config, local: bool, global_: bool) -> Optional[Path]:
    """Directory context file a selection should be written to, or None for global.

    ``--local`` targets ``./.prime/context.json``; ``--global`` the global config.
    Without either, the directory context already in effect (if any) is updated,
    so a selection made inside a pinned directory takes effect there.
    """
    if local and global_:
        get_console(stderr=True).print("[red]Error:[/red] Use either --local or --global.")
        raise typer.Exit(1)
    if global_:
        return None
    if local:
        target = Path.cwd() / LOCAL_CONTEXT_FILE
        if target.is_symlink() or target.parent.is_symlink():
            get_console(stderr=True).print(
                f"[red]Error:[/red] {escape(str(target))} or its directory is a symlink; "
                "replace it with a plain file or directory first."
            )
            raise typer.Exit(1)
        return target
    return config.local_context_file


def update_local_context(path: Path, **fields: Optional[str]) -> None:
    """Merge fields into a directory context file.

    A None value removes the key, except ``team_id``, where null pins the
    personal account.
    """
    data = read_local_context(path) if path.is_file() else {}
    for key, value in fields.items():
        if value is None and key != "team_id":
            data.pop(key, None)
        else:
            data[key] = value
    write_local_context(path, data)


def pin_team(path: Path, team_id: Optional[str], team_name: Optional[str] = None) -> None:
    """Pin a team (None: the personal account) in a directory context file."""
    update_local_context(
        path,
        team_id=team_id or None,
        team_name=team_name if team_id else None,
        team_role=None,
    )


def describe_local_context(path: Path) -> str:
    """Short 'for <dir>' phrase naming the directory a context file applies to."""
    return f"for {escape(str(path.parent.parent))}"


def apply_team(
    config: Config,
    target: Optional[Path],
    team_id: Optional[str],
    team_name: Optional[str] = None,
    team_role: Optional[str] = None,
) -> str:
    """Store a team selection in a directory context file, or else the config.

    Returns a phrase describing where it applies ("" for the global config).
    """
    if target is not None:
        pin_team(target, team_id, team_name)
        return " " + describe_local_context(target)

    if config.local_context_file is None:
        store = config
    else:
        # --global from inside a directory context: bypass it and write the
        # global config (or its current environment) directly.
        store = Config(use_context=False)
    store.set_team(team_id, team_name=team_name, team_role=team_role)
    store.update_current_environment_file()
    if config.team_pinned or config.writes_context:
        get_console().print(
            f"[yellow]Note:[/yellow] {escape(str(config.local_context_file))} "
            "still selects this directory's account."
        )
    return ""
