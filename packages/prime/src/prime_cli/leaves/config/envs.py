"""Pydantic leaf for ``prime config envs``."""

from __future__ import annotations

from pydantic_config import BaseConfig


class Config(BaseConfig):
    """List available environments"""

    pass


POSITIONALS = ()


def run(config: Config):
    from prime_cli.commands.config import list_envs as callback

    return callback(config)
