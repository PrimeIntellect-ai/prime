"""Pydantic leaf for ``prime eval view``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Launch the interactive evaluation viewer."""

    limit: int = Field(
        50, validation_alias=AliasChoices("limit", "n"), description="Max evaluation rows to load"
    )
    env_dir: str | None = Field(
        None,
        validation_alias=AliasChoices("env_dir", "e"),
        description="Path to environments directory",
    )
    outputs_dir: str | None = Field(
        None,
        validation_alias=AliasChoices("outputs_dir", "o"),
        description="Path to outputs directory",
    )


POSITIONALS = ()


def run(config: Config):
    from prime_cli.commands.evals import view_cmd as callback

    return callback(config)
