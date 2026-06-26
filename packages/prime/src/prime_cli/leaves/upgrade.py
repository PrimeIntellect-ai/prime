"""Pydantic leaf for ``prime upgrade``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Upgrade the Prime CLI to the latest version"""

    check: bool = Field(
        False,
        validation_alias=AliasChoices("check", "c"),
        description="Only check for updates, don't upgrade",
    )
    force: bool = Field(
        False,
        validation_alias=AliasChoices("force", "f"),
        description="Force upgrade even if already on latest version",
    )


POSITIONALS = ()


def run(config: Config):
    from prime_cli.commands.upgrade import upgrade as callback

    return callback(config)
