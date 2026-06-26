"""Pydantic leaf for ``prime disks terminate``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Terminate a disk"""

    disk_id: str = Field(...)
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


POSITIONALS = ("disk_id",)


def run(config: Config):
    from prime_cli.commands.disks import terminate as callback

    return callback(config)
