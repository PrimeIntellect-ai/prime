"""Pydantic leaf for ``prime config save``."""

from __future__ import annotations

from pydantic import Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Save current config as environment (including API key)"""

    name: str = Field(..., description="Name for the environment")


POSITIONALS = ("name",)


def run(config: Config):
    from prime_cli.commands.config import save_env as callback

    return callback(config)
