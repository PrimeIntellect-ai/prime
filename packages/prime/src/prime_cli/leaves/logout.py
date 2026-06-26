"""Pydantic leaf for ``prime logout``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Log out of Prime Intellect"""

    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


POSITIONALS = ()


def run(config: Config):
    from prime_cli.commands.logout import logout as callback

    return callback(config)
