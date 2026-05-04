"""Lab platform commands."""

from pathlib import Path

import typer
from rich.console import Console

from ..lab_setup import run_lab_doctor, run_lab_setup, run_lab_sync

app = typer.Typer(
    help="Lab platform commands",
    invoke_without_command=True,
    no_args_is_help=False,
)
console = Console()


@app.callback()
def lab(
    ctx: typer.Context,
    limit: int = typer.Option(1000, "--limit", "-n", help="Max rows to load per section"),
    env_dir: str = typer.Option(
        "./environments",
        "--env-dir",
        help="Local environments directory for discovering eval outputs",
    ),
    outputs_dir: str = typer.Option(
        "./outputs",
        "--outputs-dir",
        help="Local outputs directory for discovering eval outputs",
    ),
) -> None:
    """Launch the interactive Lab viewer."""
    if ctx.invoked_subcommand is not None:
        return
    _launch_view(limit=limit, env_dir=env_dir, outputs_dir=outputs_dir)


@app.command(
    add_help_option=False,
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
)
def setup(ctx: typer.Context) -> None:
    """Set up a Lab workspace."""
    args = list(ctx.args)
    if any(arg in {"-h", "--help"} for arg in args):
        print_lab_setup_help()
        return
    code = run_lab_setup(args, console=console)
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
    code = run_lab_doctor(list(ctx.args), console=console)
    if code != 0:
        raise typer.Exit(code)


@app.command("view")
def view(
    limit: int = typer.Option(1000, "--limit", "-n", help="Max rows to load per section"),
    env_dir: str = typer.Option(
        "./environments",
        "--env-dir",
        help="Local environments directory for discovering eval outputs",
    ),
    outputs_dir: str = typer.Option(
        "./outputs",
        "--outputs-dir",
        help="Local outputs directory for discovering eval outputs",
    ),
) -> None:
    """Launch the interactive Lab viewer."""
    _launch_view(limit=limit, env_dir=env_dir, outputs_dir=outputs_dir)


@app.command("mcp")
def mcp(
    workspace: Path = typer.Option(
        Path.cwd(),
        "--workspace",
        help="Workspace whose running Lab TUI should receive MCP tool calls.",
    ),
) -> None:
    """Run the Lab MCP server over stdio."""

    from ..lab_mcp import run_lab_mcp_server

    run_lab_mcp_server(workspace)


def _launch_view(*, limit: int, env_dir: str, outputs_dir: str) -> None:
    if limit < 1:
        console.print("[red]Error:[/red] --limit must be at least 1")
        raise typer.Exit(1)

    from prime_lab_app import run_lab_view

    run_lab_view(
        limit=limit,
        env_dir=env_dir,
        outputs_dir=outputs_dir,
        workspace=Path.cwd(),
    )


def print_lab_setup_help() -> None:
    """Print help for the Prime-owned Lab setup backend."""

    run_lab_setup(["--help"], console=console)
