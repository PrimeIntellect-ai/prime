"""Pydantic leaf for ``prime sandbox download``."""

from __future__ import annotations

from pydantic import Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Download a file from a sandbox"""

    sandbox_id: str = Field(..., description="Sandbox ID to download file from")
    remote_path: str = Field(..., description="Path to file in sandbox")
    local_file: str = Field(..., description="Path where file should be saved locally")


POSITIONALS = ("sandbox_id", "remote_path", "local_file")


def run(config: Config):
    from prime_cli.commands.sandbox import download_file as callback

    return callback(config)
