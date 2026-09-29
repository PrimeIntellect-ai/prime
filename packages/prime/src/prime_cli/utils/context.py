"""Helpers for commands that persist CLI configuration."""

import os
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


def require_loadable_config(notice: bool = False) -> None:
    """Exit readably if the config cannot load (e.g. a broken directory context).

    With ``notice``, also say on stderr when a directory context changes the
    account, since pins can arrive with a cloned repository.
    """
    if os.environ.get("PRIME_CONTEXT"):
        return  # replaces the directory context and was validated already
    try:
        message = Config().local_context_notice()
    except (ValueError, TypeError, AttributeError) as e:
        get_console(stderr=True).print(f"[red]Error:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    disabled = os.environ.get("PRIME_DISABLE_CONTEXT_NOTICE", "").lower() in ("1", "true", "yes")
    if notice and message and not disabled:
        get_console(stderr=True).print(f"[dim]{escape(message)}[/dim]")


def _fail(message: str) -> None:
    get_console(stderr=True).print(f"[red]Error:[/red] {message}")
    raise typer.Exit(1)


def local_context_target(config: Config, local: bool, global_: bool) -> Optional[Path]:
    """Where a selection is written: a directory context file, or None for global.

    ``--local`` targets the nearest git or Lab workspace root (else the current
    directory). Without flags, a directory context already in effect is updated.
    """
    if local and global_:
        _fail("Use either --local or --global.")
    if global_:
        return None
    if not local:
        return config.local_context_file

    home = Path.home().resolve()
    directory = cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if candidate == home:
            break
        if (candidate / ".git").exists() or (candidate / ".prime" / "lab.json").is_file():
            directory = candidate
            break
    if directory == home:
        _fail("--local cannot pin your home directory; ~/.prime is the global config.")
    target = directory / LOCAL_CONTEXT_FILE
    if target.is_symlink() or target.parent.is_symlink():
        _fail(f"{escape(str(target))} or its directory is a symlink.")
    return target


def pin_team(path: Path, team_id: Optional[str], team_name: Optional[str] = None) -> None:
    """Pin a team (None: personal) in a directory context file.

    No role is stored: the file may be shared, and a role is per-user.
    """
    data = read_local_context(path) if path.is_file() else {}
    data.pop("team_role", None)
    data.pop("team_name", None)
    data["team_id"] = team_id or None
    if team_id and team_name:
        data["team_name"] = team_name
    write_local_context(path, data)


def apply_team(
    config: Config,
    target: Optional[Path],
    team_id: Optional[str],
    team_name: Optional[str] = None,
    team_role: Optional[str] = None,
) -> str:
    """Store a team selection; returns " for <dir>" when it went to a directory context."""
    if target is not None:
        pin_team(target, team_id, team_name)
        return f" for {escape(str(target.parent.parent))}"

    # --global inside a directory context writes the global config directly.
    store = config if config.local_context_file is None else Config(use_context=False)
    store.set_team(team_id, team_name=team_name, team_role=team_role)
    store.update_current_environment_file()
    if config.team_pinned or config.writes_context:
        get_console().print(
            f"[yellow]Note:[/yellow] {escape(str(config.local_context_file))} "
            "still selects this directory's account."
        )
    return ""
