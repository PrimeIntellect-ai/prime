"""Configuration for ``prime lab sync``."""

from __future__ import annotations

from pydantic import AliasChoices, Field, model_validator
from pydantic_config import BaseConfig


class Config(BaseConfig):
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
    def validate_agent_selection(self) -> "Config":
        if self.agents is not None and self.no_agent:
            raise ValueError("--agent and --no-agent cannot be used together")
        return self


POSITIONALS = ()


def run(config: Config) -> None:
    from prime_cli.commands.lab import sync as callback

    return callback(config)
