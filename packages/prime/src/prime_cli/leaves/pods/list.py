"""Pydantic leaf for ``prime pods list``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """List your running pods"""

    limit: int = Field(100, description="Maximum number of pods to list")
    offset: int = Field(0, description="Number of pods to skip")
    watch: bool = Field(
        False,
        validation_alias=AliasChoices("watch", "w"),
        description="Watch pods list in real-time",
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


POSITIONALS = ()


def run(config: Config):
    from prime_cli.commands.pods import list as callback

    return callback(config)
