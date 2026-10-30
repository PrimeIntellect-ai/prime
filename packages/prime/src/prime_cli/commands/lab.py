"""Removed Lab workspace commands, kept so old invocations explain the removal."""

import typer

from ..utils import PlainTyper, get_console

app = PlainTyper(
    help="Removed: Lab workspaces are no longer set up by the CLI.",
    no_args_is_help=True,
)
console = get_console()

_LAB_REMOVED_NOTE = (
    "[yellow]`prime lab` was removed.[/yellow] It downloaded workspace files that "
    "verifiers no longer publishes. Add verifiers to your own project with "
    "`uv add verifiers`, and start Hosted Training runs from the prime-rl examples: "
    "https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples"
)
_HYGIENE_WORKFLOW_NOTE = (
    "Remove .github/workflows/prime-lab-hygiene.yml from your repository; "
    "this check no longer does anything."
)
_IGNORE_LEGACY_FLAGS = {"allow_extra_args": True, "ignore_unknown_options": True}


def _lab_removed() -> None:
    console.print(_LAB_REMOVED_NOTE)
    raise typer.Exit(1)


@app.command(add_help_option=False, deprecated=True, context_settings=_IGNORE_LEGACY_FLAGS)
def setup() -> None:
    """Removed: Lab workspaces are no longer set up by the CLI."""
    _lab_removed()


@app.command(add_help_option=False, deprecated=True, context_settings=_IGNORE_LEGACY_FLAGS)
def sync() -> None:
    """Removed: Lab workspaces are no longer set up by the CLI."""
    _lab_removed()


@app.command(add_help_option=False, deprecated=True, context_settings=_IGNORE_LEGACY_FLAGS)
def doctor() -> None:
    """Removed: Lab workspaces are no longer set up by the CLI."""
    _lab_removed()


@app.command(deprecated=True, context_settings=_IGNORE_LEGACY_FLAGS)
def hygiene() -> None:
    """Removed: exits 0 so existing prime-lab-hygiene workflows keep passing."""
    console.print(_LAB_REMOVED_NOTE)
    console.print(_HYGIENE_WORKFLOW_NOTE)


@app.command("register-github", deprecated=True, context_settings=_IGNORE_LEGACY_FLAGS)
def register_github() -> None:
    """Removed: Lab workspaces are no longer set up by the CLI."""
    _lab_removed()
