"""Pydantic leaf for ``prime sandbox list-ports``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """List exposed ports for a sandbox, or all sandboxes if no ID is provided"""

    sandbox_id: str | None = Field(
        None, description="Sandbox ID (omit to list all exposed ports across all sandboxes)"
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


POSITIONALS = ("sandbox_id",)


def run(config: Config):
    from prime_cli.commands.sandbox import list_ports as callback

    return callback(config)
