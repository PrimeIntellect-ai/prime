"""Pydantic leaf for ``prime train checkpoints``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """List checkpoints for a run."""

    run_id: str = Field(..., description="Run ID to list checkpoints for")
    status_filter: str | None = Field(
        None,
        validation_alias=AliasChoices("status_filter", "status", "s"),
        description="Filter by status (READY, PENDING, UPLOADING, FAILED)",
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


POSITIONALS = ("run_id",)


def run(config: Config):
    from prime_cli.commands.rl import list_checkpoints as callback

    return callback(config)
