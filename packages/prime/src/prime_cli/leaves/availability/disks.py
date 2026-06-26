"""Pydantic leaf for ``prime availability disks``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """List available disks"""

    regions: list[str] | None = Field(None, description="Filter by regions (e.g., united_states)")
    data_center_id: str | None = Field(None, description="Filter by data center ID (e.g., US-1)")
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


POSITIONALS = ()


def run(config: Config):
    from prime_cli.commands.availability import disks as callback

    return callback(config)
