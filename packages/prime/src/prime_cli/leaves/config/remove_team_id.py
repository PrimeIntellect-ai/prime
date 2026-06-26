"""Pydantic leaf for ``prime config remove-team-id``."""

from __future__ import annotations

from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Remove team ID to use personal account"""

    pass


POSITIONALS = ()


def run(config: Config):
    from prime_cli.commands.config import remove_team_id as callback

    return callback(config)
