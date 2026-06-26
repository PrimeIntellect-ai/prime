"""Pydantic leaf for ``prime train metrics``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Get Hosted Training metrics for a run."""

    run_id: str = Field(..., description="Run ID to get metrics for")
    min_step: int | None = Field(None, description="Minimum step (inclusive)")
    max_step: int | None = Field(None, description="Maximum step (inclusive)")
    limit: int | None = Field(
        None, validation_alias=AliasChoices("limit", "n"), description="Maximum number of records"
    )


POSITIONALS = ("run_id",)


def run(config: Config):
    from prime_cli.commands.rl import get_metrics as callback

    return callback(config)
