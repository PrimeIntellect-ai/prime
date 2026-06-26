"""Configuration for ``prime lab doctor``."""

from __future__ import annotations

from pydantic import Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Check a Lab workspace."""

    fix: bool = Field(False, description="Apply safe local remediations.")


POSITIONALS = ()


def run(config: Config) -> None:
    from prime_cli.commands.lab import doctor as callback

    return callback(config)
