"""Lab commands for verifiers development."""

import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from ..verifiers_plugin import load_verifiers_prime_plugin

app = typer.Typer(help="Lab commands for verifiers development", no_args_is_help=True)
console = Console()

SUPPORTED_AGENTS = ("codex", "claude", "cursor", "opencode")


def _parse_agents_csv(raw_agents: str) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()

    for raw in raw_agents.split(","):
        agent = raw.strip().lower()
        if not agent:
            continue
        if agent not in SUPPORTED_AGENTS:
            allowed = ", ".join(SUPPORTED_AGENTS)
            raise typer.BadParameter(f"Unsupported agent '{agent}'. Supported values: {allowed}")
        if agent in seen:
            continue
        seen.add(agent)
        selected.append(agent)

    return selected


def _prompt_agents() -> list[str]:
    console.print(
        "Supported coding agents: [cyan]codex[/cyan], [cyan]claude[/cyan], "
        "[cyan]cursor[/cyan], [cyan]opencode[/cyan]"
    )
    while True:
        primary = typer.prompt("Primary coding agent", default="codex").strip().lower()
        if primary in SUPPORTED_AGENTS:
            break
        console.print("[red]Invalid agent.[/red] Choose one of: " + ", ".join(SUPPORTED_AGENTS))

    selected = [primary]
    if typer.confirm("Using multiple coding agents?", default=False):
        additional = typer.prompt(
            "Additional agents (comma-separated)",
            default="",
        )
        for agent in _parse_agents_csv(additional):
            if agent not in selected:
                selected.append(agent)

    return selected


def _create_skill_dirs(agents: list[str]) -> None:
    for agent in agents:
        skills_dir = Path(f".{agent}") / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[dim]Prepared {skills_dir}[/dim]")


@app.command(
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    }
)
def setup(
    ctx: typer.Context,
    prime_rl: bool = typer.Option(
        False, "--prime-rl", help="Install prime-rl and download prime-rl configs"
    ),
    agents: Optional[str] = typer.Option(
        None,
        "--agents",
        help="Comma-separated coding agents to scaffold (codex,claude,cursor,opencode)",
    ),
    no_interactive: bool = typer.Option(
        False,
        "--no-interactive",
        help="Disable interactive agent prompts",
    ),
    skip_agents_md: bool = typer.Option(
        False,
        "--skip-agents-md",
        help="Skip downloading AGENTS.md, CLAUDE.md, and environments/AGENTS.md",
    ),
    skip_install: bool = typer.Option(
        False,
        "--skip-install",
        help="Skip uv project initialization and verifiers installation",
    ),
) -> None:
    """Set up a verifiers training workspace."""
    selected_agents: list[str] = []
    if agents:
        selected_agents = _parse_agents_csv(agents)
    elif not no_interactive and sys.stdin.isatty():
        selected_agents = _prompt_agents()

    plugin = load_verifiers_prime_plugin(console=console)
    setup_args: list[str] = list(ctx.args)
    if prime_rl:
        setup_args.append("--prime-rl")
    if skip_agents_md:
        setup_args.append("--skip-agents-md")
    if skip_install:
        setup_args.append("--skip-install")

    command = plugin.build_module_command(plugin.setup_module, setup_args)
    result = subprocess.run(command)
    if result.returncode != 0:
        raise typer.Exit(result.returncode)

    if selected_agents:
        _create_skill_dirs(selected_agents)
