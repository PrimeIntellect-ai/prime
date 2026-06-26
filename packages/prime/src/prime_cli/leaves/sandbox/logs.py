"""Pydantic leaf for ``prime sandbox logs``."""

from __future__ import annotations

from pydantic import Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Get logs from a sandbox"""

    sandbox_id: str = Field(...)


POSITIONALS = ("sandbox_id",)


def run(config: Config):
    from prime_cli.commands.sandbox import logs as callback

    return callback(config)
