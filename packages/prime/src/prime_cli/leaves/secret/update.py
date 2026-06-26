"""Pydantic leaf for ``prime secret update``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Update an existing global secret."""

    secret_id: str | None = Field(
        None, description="Secret ID to update (interactive selection if not provided)"
    )
    name: str | None = Field(
        None, validation_alias=AliasChoices("name", "n"), description="New secret name"
    )
    value: str | None = Field(
        None, validation_alias=AliasChoices("value", "v"), description="New secret value"
    )
    description: str | None = Field(
        None,
        validation_alias=AliasChoices("description", "d"),
        description="New secret description",
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


POSITIONALS = ("secret_id",)


def run(config: Config):
    from prime_cli.commands.secrets import secret_update as callback

    return callback(config)
