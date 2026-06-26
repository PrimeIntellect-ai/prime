"""Pydantic leaf for ``prime disks get``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Get detailed information about a specific disk"""

    disk_id: str = Field(...)
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


POSITIONALS = ("disk_id",)


def run(config: Config):
    from prime_cli.commands.disks import get as callback

    return callback(config)
