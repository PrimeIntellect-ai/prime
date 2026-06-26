"""Pydantic leaf for ``prime train distributions``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Get reward/advantage distribution histogram for a run."""

    run_id: str = Field(..., description="Run ID to get distributions for")
    distribution_type: str | None = Field(
        None,
        validation_alias=AliasChoices("distribution_type", "type", "t"),
        description="Distribution type (defaults to all)",
    )
    step: int | None = Field(
        None,
        validation_alias=AliasChoices("step", "s"),
        description="Step number (defaults to latest)",
    )


POSITIONALS = ("run_id",)


def run(config: Config):
    from prime_cli.commands.rl import get_distributions as callback

    return callback(config)
