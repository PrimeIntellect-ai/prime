"""Pydantic leaf for ``prime availability gpu-types``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """List available GPU types"""

    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


POSITIONALS = ()


def run(config: Config):
    from prime_cli.commands.availability import gpu_types as callback

    return callback(config)
