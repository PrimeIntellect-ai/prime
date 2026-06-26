"""Pydantic leaf for ``prime pods ssh``."""

from __future__ import annotations

from pydantic import Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """SSH / connect to a pod using configured SSH key"""

    pod_id: str = Field(...)


POSITIONALS = ("pod_id",)


def run(config: Config):
    from prime_cli.commands.pods import connect as callback

    return callback(config)
