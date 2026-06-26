"""Pydantic leaf for ``prime train usage``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Show token usage and price for a single training run."""

    run_id: str = Field(..., description="RFT run ID (e.g. rft_...")
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )
    watch: bool = Field(
        False,
        validation_alias=AliasChoices("watch", "w"),
        description="Poll continuously and update in place",
    )
    interval: int = Field(
        30,
        validation_alias=AliasChoices("interval", "n"),
        description="Seconds between polls when --watch is set",
    )


POSITIONALS = ("run_id",)


def run(config: Config):
    from prime_cli.commands.usage import run_usage_command as callback

    return callback(config)
