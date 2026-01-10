import typer

app = typer.Typer(help="Lab commands for verifiers development")


@app.command()
def setup(
    prime_rl: bool = typer.Option(
        False, "--prime-rl", help="Install prime-rl and download prime-rl configs"
    ),
    vf_rl: bool = typer.Option(False, "--vf-rl", help="Download vf-rl configs"),
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
    """Setup verifiers training workspace (passthrough to vf-setup)."""
    from verifiers.scripts.setup import run_setup

    run_setup(
        prime_rl=prime_rl, vf_rl=vf_rl, skip_agents_md=skip_agents_md, skip_install=skip_install
    )
