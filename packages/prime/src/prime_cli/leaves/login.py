"""Pydantic leaf for ``prime login``."""

from __future__ import annotations

from pydantic import Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Login to Prime Intellect"""

    headless: bool = Field(False, description="Don't attempt to open browser")


POSITIONALS = ()


def run(config: Config):
    from prime_cli.commands.login import login as callback

    return callback(config)
