"""Pydantic leaf for ``prime disks list``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """List your persistent disks"""

    limit: int = Field(100, description="Maximum number of disks to list")
    offset: int = Field(0, description="Number of disks to skip")
    watch: bool = Field(
        False,
        validation_alias=AliasChoices("watch", "w"),
        description="Watch disks list in real-time",
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


POSITIONALS = ()


def run(config: Config):
    from prime_cli.commands.disks import list as callback

    return callback(config)
