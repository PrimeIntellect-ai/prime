"""Pydantic leaf for ``prime train stop``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Stop a run."""

    run_id: str = Field(..., description="Run ID to stop")
    force: bool = Field(
        False, validation_alias=AliasChoices("force", "f"), description="Skip confirmation"
    )


POSITIONALS = ("run_id",)


def run(config: Config):
    from prime_cli.commands.rl import stop_run as callback

    return callback(config)
