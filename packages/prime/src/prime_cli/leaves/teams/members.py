"""Pydantic leaf for ``prime teams members``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """List members of a team."""

    team_id: str | None = Field(None, description="Team ID (uses config team_id if omitted)")
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


POSITIONALS = ()


def run(config: Config):
    from prime_cli.commands.teams import list_members as callback

    return callback(config)
