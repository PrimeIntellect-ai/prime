"""Pydantic leaf for ``prime config view``."""

from __future__ import annotations

from pydantic_config import BaseConfig


class Config(BaseConfig):
    """View current configuration"""

    pass


POSITIONALS = ()


def run(config: Config):
    from prime_cli.commands.config import view as callback

    return callback(config)
