"""Pydantic leaf for ``prime sandbox reset-cache``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Reset sandbox authentication cache"""

    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


POSITIONALS = ()


def run(config: Config):
    from prime_cli.commands.sandbox import reset_cache as callback

    return callback(config)
