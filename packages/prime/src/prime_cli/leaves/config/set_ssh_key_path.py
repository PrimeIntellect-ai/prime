"""Pydantic leaf for ``prime config set-ssh-key-path``."""

from __future__ import annotations

from pydantic import Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Set the SSH private key path"""

    path: str = Field(..., description="Path to your SSH private key file")


POSITIONALS = ("path",)


def run(config: Config):
    from prime_cli.commands.config import set_ssh_key_path as callback

    return callback(config)
