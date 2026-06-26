"""Pydantic leaf for ``prime config set-share-resources-with-team``."""

from __future__ import annotations

from pydantic import Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Set whether to automatically share new resources with all team members"""

    enabled: str = Field(..., description="Enable or disable auto-sharing with team: true or false")


POSITIONALS = ("enabled",)


def run(config: Config):
    from prime_cli.commands.config import set_share_resources_with_team as callback

    return callback(config)
