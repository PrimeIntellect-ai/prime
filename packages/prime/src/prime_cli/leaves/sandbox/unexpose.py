"""Pydantic leaf for ``prime sandbox unexpose``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Unexpose a port from a sandbox"""

    sandbox_id: str = Field(..., description="Sandbox ID")
    exposure_id: str = Field(..., description="Exposure ID to remove")
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


POSITIONALS = ("sandbox_id", "exposure_id")


def run(config: Config):
    from prime_cli.commands.sandbox import unexpose_port as callback

    return callback(config)
