"""Pydantic leaf for ``prime train list``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """List your runs."""

    team: str | None = Field(
        None, validation_alias=AliasChoices("team", "t"), description="Filter by team ID"
    )
    mine: bool = Field(
        False, description="Filter to only your own runs (useful for admin accounts)"
    )
    num: int = Field(20, validation_alias=AliasChoices("num", "n"), description="Items per page")
    page: int = Field(1, validation_alias=AliasChoices("page", "p"), description="Page number")
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


POSITIONALS = ()


def run(config: Config):
    from prime_cli.commands.rl import list_runs as callback

    return callback(config)
