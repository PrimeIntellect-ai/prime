"""Lab platform commands."""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field, model_validator
from pydantic_config import BaseConfig
from rich.console import Console

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


# --- inlined config schemas (previously in lab_configs) ---
class LabDoctorConfig(BaseConfig):
    """Check a Lab workspace."""

    fix: bool = Field(False, description="Apply safe local remediations.")


class LabHygieneConfig(BaseConfig):
    """Check cheap Lab git hygiene."""

    fix: bool = Field(
        False, description="Apply safe local remediations such as dirs and gitignore entries."
    )


class LabMcpConfig(BaseConfig):
    """Run the Lab MCP server over stdio."""

    workspace: Path | None = Field(
        None, description="Workspace whose running Lab TUI should receive MCP tool calls."
    )


class LabRegisterGithubConfig(BaseConfig):
    """Write the GitHub workflow for Lab git hygiene."""

    pass


class LabSetupConfig(BaseConfig):
    """Set up a Lab workspace."""

    skip_agents_md: bool = Field(
        False,
        description="Skip workspace agent guidance files.",
    )
    skip_install: bool = Field(
        False,
        description="Skip uv project initialization and Verifiers installation.",
    )
    agents: str | None = Field(
        None,
        validation_alias=AliasChoices("agents", "agent"),
        description="Comma-separated coding agents to configure, or 'all'.",
    )
    no_interactive: bool = Field(
        False,
        description="Use setup defaults without prompts.",
    )


class LabSyncConfig(BaseConfig):
    """Refresh Lab skills and local agent guidance."""

    agents: str | None = Field(
        None,
        validation_alias=AliasChoices("agents", "agent"),
        description="Comma-separated coding agents to configure, or 'all'.",
    )
    skip_docs: bool = Field(False, description="Skip workspace guidance refresh.")
    no_agent: bool = Field(
        False,
        description="Refresh shared assets without configuring agent skill roots.",
    )

    @model_validator(mode="after")
    def validate_agent_selection(self) -> "LabSyncConfig":
        if self.agents is not None and self.no_agent:
            raise ValueError("--agent and --no-agent cannot be used together")
        return self


class LabViewConfig(BaseConfig):
    """Launch the interactive Lab viewer."""

    limit: int = Field(1000, validation_alias=AliasChoices("limit", "n"))
    env_dir: str = Field("./environments")
    outputs_dir: str = Field("./outputs")
