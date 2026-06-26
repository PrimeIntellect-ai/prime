"""Pydantic leaf for ``prime config use``."""

from __future__ import annotations

from pydantic import Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Switch to a different environment"""

    env: str = Field(
        ..., description="Environment name: 'production' or a custom saved environment"
    )


POSITIONALS = ("env",)


def run(config: Config):
    from prime_cli.commands.config import use_environment as callback

    return callback(config)
