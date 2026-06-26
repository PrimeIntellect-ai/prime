"""Pydantic leaf for ``prime lab hygiene``."""

from __future__ import annotations

from pydantic import Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Check cheap Lab git hygiene."""

    fix: bool = Field(
        False, description="Apply safe local remediations such as dirs and gitignore entries."
    )


POSITIONALS = ()


def run(config: Config):
    from prime_cli.commands.lab import hygiene as callback

    return callback(config)
