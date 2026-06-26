"""Pydantic leaf for ``prime secret delete``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Delete a global secret."""

    secret_id: str | None = Field(
        None, description="Secret ID to delete (interactive selection if not provided)"
    )
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


POSITIONALS = ("secret_id",)


def run(config: Config):
    from prime_cli.commands.secrets import secret_delete as callback

    return callback(config)
