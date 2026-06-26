"""Pydantic leaf for ``prime train get``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Get details of a specific run."""

    run_id: str = Field(..., description="Run ID to get details for")
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


POSITIONALS = ("run_id",)


def run(config: Config):
    from prime_cli.commands.rl import get_run as callback

    return callback(config)
