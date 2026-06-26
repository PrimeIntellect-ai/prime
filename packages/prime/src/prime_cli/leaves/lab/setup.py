"""Configuration for ``prime lab setup``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
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


POSITIONALS = ()


def run(config: Config) -> None:
    from prime_cli.commands.lab import setup as callback

    return callback(config)
