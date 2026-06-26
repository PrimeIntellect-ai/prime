"""Pydantic leaf for ``prime train rollouts``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Get rollout samples for a run."""

    run_id: str = Field(..., description="Run ID to get rollouts for")
    step: int = Field(
        ...,
        validation_alias=AliasChoices("step", "s"),
        description="Step number to get rollouts for",
    )
    page: int = Field(
        1, validation_alias=AliasChoices("page", "p"), description="Page number (1-indexed)"
    )
    num: int = Field(100, validation_alias=AliasChoices("num", "n"), description="Items per page")


POSITIONALS = ("run_id",)


def run(config: Config):
    from prime_cli.commands.rl import get_rollouts as callback

    return callback(config)
