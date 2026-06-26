"""Pydantic leaf for ``prime train request``."""

from __future__ import annotations

from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Request models for Hosted Training."""

    pass


POSITIONALS = ()


def run(config: Config):
    from prime_cli.commands.rl import request_models as callback

    return callback(config)
