"""Pydantic leaf for ``prime whoami``."""

from __future__ import annotations

from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Show current authenticated user and update config"""

    pass


POSITIONALS = ()


def run(config: Config):
    from prime_cli.commands.whoami import whoami as callback

    return callback(config)
