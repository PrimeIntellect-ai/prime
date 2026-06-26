"""Pydantic leaf for ``prime sandbox get``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Get detailed information about a specific sandbox"""

    sandbox_id: str = Field(...)
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


POSITIONALS = ("sandbox_id",)


def run(config: Config):
    from prime_cli.commands.sandbox import get as callback

    return callback(config)
