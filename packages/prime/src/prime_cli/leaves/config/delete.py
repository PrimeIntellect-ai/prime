"""Pydantic leaf for ``prime config delete``."""

from __future__ import annotations

from pydantic import Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Delete a saved environment"""

    name: str = Field(..., description="Name of the saved environment")


POSITIONALS = ("name",)


def run(config: Config):
    from prime_cli.commands.config import delete_env as callback

    return callback(config)
