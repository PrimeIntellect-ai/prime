"""Pydantic leaf for ``prime train init``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Generate a template config file for a Hosted Training run."""

    output_path: str = Field("rl.toml", description="Output path for the config file")
    force: bool = Field(
        False, validation_alias=AliasChoices("force", "f"), description="Overwrite existing file"
    )


POSITIONALS = ("output_path",)


def run(config: Config):
    from prime_cli.commands.rl import init_config as callback

    return callback(config)
