"""Pydantic leaf for ``prime secret get``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Get details of a specific secret."""

    secret_id: str = Field(..., description="Secret ID to get")
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


POSITIONALS = ("secret_id",)


def run(config: Config):
    from prime_cli.commands.secrets import secret_get as callback

    return callback(config)
