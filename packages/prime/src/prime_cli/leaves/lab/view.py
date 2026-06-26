"""Pydantic leaf for ``prime lab view``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Launch the interactive Lab viewer."""

    limit: int = Field(1000, validation_alias=AliasChoices("limit", "n"))
    env_dir: str = Field("./environments")
    outputs_dir: str = Field("./outputs")


POSITIONALS = ()


def run(config: Config):
    from prime_cli.commands.lab import _launch_view as callback

    return callback(config)
