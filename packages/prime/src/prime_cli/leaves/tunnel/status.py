"""Pydantic leaf for ``prime tunnel status``."""

from __future__ import annotations

from pydantic import Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Get status of a specific tunnel."""

    tunnel_id: str = Field(..., description="Tunnel ID to check")


POSITIONALS = ("tunnel_id",)


def run(config: Config):
    from prime_cli.commands.tunnel import tunnel_status as callback

    return callback(config)
