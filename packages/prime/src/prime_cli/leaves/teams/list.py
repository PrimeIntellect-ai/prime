"""Pydantic leaf for ``prime teams list``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """List teams for the current user."""

    limit: int = Field(100, description="Maximum number of teams to list")
    offset: int = Field(0, description="Number of teams to skip")
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


POSITIONALS = ()


def run(config: Config):
    from prime_cli.commands.teams import list_teams as callback

    return callback(config)
