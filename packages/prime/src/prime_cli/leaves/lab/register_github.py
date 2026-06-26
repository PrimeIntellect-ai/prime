"""Pydantic leaf for ``prime lab register-github``."""

from __future__ import annotations

from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Write the GitHub workflow for Lab git hygiene."""

    pass


POSITIONALS = ()


def run(config: Config):
    from prime_cli.commands.lab import register_github as callback

    return callback(config)
