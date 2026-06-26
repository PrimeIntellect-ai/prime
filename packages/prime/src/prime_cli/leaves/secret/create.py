"""Pydantic leaf for ``prime secret create``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Create a new global secret."""

    name: str | None = Field(
        None,
        validation_alias=AliasChoices("name", "n"),
        description="Secret name (used as environment variable name)",
    )
    value: str | None = Field(
        None, validation_alias=AliasChoices("value", "v"), description="Secret value"
    )
    description: str | None = Field(
        None, validation_alias=AliasChoices("description", "d"), description="Secret description"
    )
    file: bool = Field(
        False,
        validation_alias=AliasChoices("file", "f"),
        description="Treat value as file content (base64 encoded)",
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


POSITIONALS = ()


def run(config: Config):
    from prime_cli.commands.secrets import secret_create as callback

    return callback(config)
