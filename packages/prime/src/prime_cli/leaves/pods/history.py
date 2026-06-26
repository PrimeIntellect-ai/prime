"""Pydantic leaf for ``prime pods history``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """List your pods history (terminated pods)"""

    limit: int = Field(100, description="Maximum number of history items to list")
    offset: int = Field(0, description="Number of history items to skip")
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


POSITIONALS = ()


def run(config: Config):
    from prime_cli.commands.pods import history as callback

    return callback(config)
