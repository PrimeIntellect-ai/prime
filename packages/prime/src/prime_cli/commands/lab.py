"""Lab platform commands."""

from pathlib import Path

from rich.console import Console

from prime_cli.command_configs import (
    LabDoctorConfig,
    LabHygieneConfig,
    LabMcpConfig,
    LabRegisterGithubConfig,
    LabSetupConfig,
    LabSyncConfig,
    LabViewConfig,
)

console = Console()


def setup(config: LabSetupConfig) -> None:
    """Set up a Lab workspace."""
    from ..lab_setup import (
        LabSetupOptions,
        emit_to_console,
        resolve_setup_agents,
        run_lab_setup_service,
    )

    options = LabSetupOptions(
        skip_agents_md=config.skip_agents_md,
        skip_install=config.skip_install,
        agents=resolve_setup_agents(config.agents, no_interactive=config.no_interactive),
    )
    result = run_lab_setup_service(
        options,
        workspace=Path.cwd(),
        emit=lambda item: emit_to_console(console, item),
    )
    if result.exit_code != 0:
        raise SystemExit(result.exit_code)


def sync(config: LabSyncConfig) -> None:
    """Refresh Lab skills and local agent guidance."""
    from ..lab_setup import (
        LabSyncOptions,
        emit_to_console,
        resolve_explicit_agents,
        run_lab_sync_service,
    )

    agents = resolve_explicit_agents(config.agents) if config.agents is not None else ()
    result = run_lab_sync_service(
        LabSyncOptions(agents=agents, skip_docs=config.skip_docs, no_agent=config.no_agent),
        workspace=Path.cwd(),
        emit=lambda item: emit_to_console(console, item),
    )
    if result.exit_code != 0:
        raise SystemExit(result.exit_code)


def doctor(config: LabDoctorConfig) -> None:
    """Check a Lab workspace."""
    from ..lab_setup import (
        LabDoctorOptions,
        print_lab_doctor_result,
        run_lab_doctor_service,
    )

    result = run_lab_doctor_service(
        LabDoctorOptions(fix=config.fix),
        workspace=Path.cwd(),
    )
    print_lab_doctor_result(result, console)
    if result.exit_code != 0:
        raise SystemExit(result.exit_code)


def hygiene(config: LabHygieneConfig) -> None:
    """Check cheap Lab git hygiene."""
    fix = config.fix

    from ..lab_hygiene import LabHygieneOptions, run_lab_hygiene_preflight

    result = run_lab_hygiene_preflight(
        LabHygieneOptions(fix=fix, fail_on_tracked=True),
        workspace=Path.cwd(),
        emit=lambda message: console.print(message, markup=False),
    )
    if result.exit_code != 0:
        raise SystemExit(result.exit_code)


def register_github(config: LabRegisterGithubConfig) -> None:
    """Write the GitHub workflow for Lab git hygiene."""

    from ..lab_hygiene import write_lab_github_workflow

    path = write_lab_github_workflow(Path.cwd())
    console.print(f"Wrote {path}", markup=False)


def mcp(config: LabMcpConfig) -> None:
    """Run the Lab MCP server over stdio."""
    workspace = config.workspace

    from ..lab_mcp import run_lab_mcp_server

    run_lab_mcp_server(workspace or Path.cwd())


def _launch_view(config: LabViewConfig) -> None:
    limit = config.limit
    env_dir = config.env_dir
    outputs_dir = config.outputs_dir

    if limit < 1:
        console.print("[red]Error:[/red] --limit must be at least 1")
        raise SystemExit(1)

    from prime_lab_app import run_lab_view

    run_lab_view(
        limit=limit,
        env_dir=env_dir,
        outputs_dir=outputs_dir,
        workspace=Path.cwd(),
    )
