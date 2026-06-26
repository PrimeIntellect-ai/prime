"""Pydantic leaf for ``prime config set-team-id``."""

from __future__ import annotations

from pydantic import Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Set your team ID."""

    team_id: str = Field(..., description="Your Prime Intellect team ID.")


POSITIONALS = ("team_id",)


def run(config: Config):
    from prime_cli.commands.config import set_team_id as callback

    return callback(config)
