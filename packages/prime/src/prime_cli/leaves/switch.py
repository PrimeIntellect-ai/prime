"""Pydantic leaf for ``prime switch``."""

from __future__ import annotations

from pydantic import Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Switch between your personal account and team contexts"""

    target: str | None = Field(None, description="'personal', a team slug, or a team ID")


POSITIONALS = ("target",)


def run(config: Config):
    from prime_cli.commands.switch import switch as callback

    return callback(config)
