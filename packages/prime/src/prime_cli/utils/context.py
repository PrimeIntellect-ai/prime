"""Helpers for commands that persist CLI configuration."""

import os
import subprocess
from pathlib import Path
from typing import Optional

import typer
from rich.markup import escape

from prime_cli.core import Config
from prime_cli.core.config import (
    LOCAL_CONTEXT_FILE,
    find_local_context_file,
    read_local_context,
    write_local_context,
)

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
    if os.environ.get("PRIME_CONTEXT") or find_local_context_file() is None:
        # Nothing to check: without a pin every command behaves as before, even
        # --help with an unwritable or corrupt ~/.prime.
        return
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


def _git(directory: Path, *args: str) -> Optional[str]:
    """stdout of a git command run in ``directory``, or None when it fails."""
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), *args],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def keep_out_of_git(path: Path) -> Optional[str]:
    """Add a pin to the repository's ``.git/info/exclude`` so it is not committed
    by accident. A context pin names a saved context from this user's ~/.prime,
    so a committed one breaks every clone and CI job that lacks it.

    Returns "excluded" when an entry was added, "tracked" when the pin is already
    committed (a deliberate choice, left alone), else None.
    """
    directory = path.parent.parent
    top = _git(directory, "rev-parse", "--show-toplevel")
    if not top:
        return None
    if _git(directory, "ls-files", "--error-unmatch", "--", str(path)) is not None:
        return "tracked"
    if _git(directory, "check-ignore", "-q", "--", str(path)) is not None:
        return None
    exclude = _git(directory, "rev-parse", "--git-path", "info/exclude")
    if not exclude:
        return None
    try:
        pattern = "/" + path.resolve().relative_to(Path(top).resolve()).as_posix()
        exclude_file = directory / exclude
        exclude_file.parent.mkdir(parents=True, exist_ok=True)
        existing = exclude_file.read_text() if exclude_file.is_file() else ""
        separator = "" if not existing or existing.endswith("\n") else "\n"
        exclude_file.write_text(f"{existing}{separator}{pattern}\n")
    except (OSError, ValueError):
        return None
    return "excluded"


def write_pin(path: Path, data: dict) -> None:
    """Write a directory context file, keeping a new one out of git."""
    write_local_context(path, data)
    if keep_out_of_git(path) == "excluded":
        get_console(stderr=True).print(
            f"[dim]Added {escape(str(LOCAL_CONTEXT_FILE))} to .git/info/exclude so it stays "
            "local; 'git add -f' it to share a team pin with the repository.[/dim]"
        )


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
    write_pin(path, data)


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
