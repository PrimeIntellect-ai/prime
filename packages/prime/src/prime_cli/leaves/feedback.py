"""Pydantic leaf for ``prime feedback``."""

from __future__ import annotations

from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Submit feedback about Prime Intellect."""

    pass


POSITIONALS = ()


def run(config: Config):
    from prime_cli.commands.feedback import feedback as callback

    return callback(config)
