"""Lab workspace commands."""

from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(
    help="Set up and maintain Lab workspaces",
    no_args_is_help=True,
)
console = Console()


@app.command(
    add_help_option=False,
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
)
def setup(ctx: typer.Context) -> None:
    """Set up a Lab workspace."""
    from ..lab_setup import run_lab_setup

    code = run_lab_setup(list(ctx.args), console=console)
    if code != 0:
        raise typer.Exit(code)


@app.command(
    add_help_option=False,
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
)
def sync(ctx: typer.Context) -> None:
    """Refresh Lab skills and local agent guidance."""
    from ..lab_setup import run_lab_sync

    code = run_lab_sync(list(ctx.args), console=console)
    if code != 0:
        raise typer.Exit(code)


@app.command(
    add_help_option=False,
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
)
def doctor(ctx: typer.Context) -> None:
    """Check a Lab workspace."""
    from ..lab_setup import run_lab_doctor

    code = run_lab_doctor(list(ctx.args), console=console)
    if code != 0:
        raise typer.Exit(code)


@app.command("hygiene")
def hygiene(
    fix: bool = typer.Option(
        False,
        "--fix",
        help="Apply safe local remediations such as dirs and gitignore entries.",
    ),
) -> None:
    """Check cheap Lab git hygiene."""

    from ..lab_hygiene import LabHygieneOptions, run_lab_hygiene_preflight

    result = run_lab_hygiene_preflight(
        LabHygieneOptions(fix=fix, fail_on_tracked=True),
        workspace=Path.cwd(),
        emit=lambda message: console.print(message, markup=False),
    )
    if result.exit_code != 0:
        raise typer.Exit(result.exit_code)


@app.command("register-github")
def register_github() -> None:
    """Write the GitHub workflow for Lab git hygiene."""

    from ..lab_hygiene import write_lab_github_workflow

    path = write_lab_github_workflow(Path.cwd())
    console.print(f"Wrote {path}", markup=False)
