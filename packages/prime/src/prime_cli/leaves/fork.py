"""Pydantic leaf for ``prime fork``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Fork a public environment into your Prime Intellect namespace."""

    environment: str = Field(..., description="Public environment to fork, in owner/name format")
    team: str | None = Field(
        None,
        validation_alias=AliasChoices("team", "t"),
        description="Team slug to fork into (uses configured team ID if omitted)",
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


POSITIONALS = ("environment",)


def run(config: Config):
    from prime_cli.commands.fork import fork as callback

    return callback(config)
