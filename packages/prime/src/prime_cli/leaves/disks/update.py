"""Pydantic leaf for ``prime disks update``."""

from __future__ import annotations

from pydantic import Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Update a disk's name"""

    disk_id: str = Field(...)
    name: str = Field(..., description="New name for the disk")


POSITIONALS = ("disk_id",)


def run(config: Config):
    from prime_cli.commands.disks import update as callback

    return callback(config)
