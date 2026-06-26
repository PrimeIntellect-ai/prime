"""Pydantic leaf for ``prime train delete``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Delete a run."""

    run_id: str = Field(..., description="Run ID to delete")
    force: bool = Field(
        False, validation_alias=AliasChoices("force", "f"), description="Skip confirmation"
    )


POSITIONALS = ("run_id",)


def run(config: Config):
    from prime_cli.commands.rl import delete_run as callback

    return callback(config)
