"""Pydantic leaf for ``prime pods terminate``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Terminate a pod"""

    pod_id: str = Field(...)
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


POSITIONALS = ("pod_id",)


def run(config: Config):
    from prime_cli.commands.pods import terminate as callback

    return callback(config)
