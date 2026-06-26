"""Pydantic leaf for ``prime config reset``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Reset configuration to defaults"""

    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


POSITIONALS = ()


def run(config: Config):
    from prime_cli.commands.config import reset as callback

    return callback(config)
