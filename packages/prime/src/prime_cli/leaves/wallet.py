"""Pydantic leaf for ``prime wallet``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Show wallet balance and most recent billing rows."""

    limit: int = Field(
        20,
        validation_alias=AliasChoices("limit", "n"),
        description="Number of recent billing rows to fetch (max 100)",
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


POSITIONALS = ()


def run(config: Config):
    from prime_cli.commands.wallet import wallet_command as callback

    return callback(config)
