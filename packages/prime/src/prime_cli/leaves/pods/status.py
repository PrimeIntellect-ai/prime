"""Pydantic leaf for ``prime pods status``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Get detailed status of a specific pod"""

    pod_id: str = Field(...)
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


POSITIONALS = ("pod_id",)


def run(config: Config):
    from prime_cli.commands.pods import status as callback

    return callback(config)
