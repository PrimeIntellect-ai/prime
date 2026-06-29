"""Pydantic Config schemas for the ``prime lab`` surface.

These were previously defined as one-file-per-command leaves under
``prime_cli.leaves``; they now live next to their command callbacks and are
consumed by the router via ``Command.config_attr``.
"""

from __future__ import annotations

import pathlib

from pydantic import AliasChoices, Field, model_validator
from pydantic_config import BaseConfig


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

    workspace: pathlib.Path | None = Field(
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
