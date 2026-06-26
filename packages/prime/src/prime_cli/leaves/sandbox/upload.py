"""Pydantic leaf for ``prime sandbox upload``."""

from __future__ import annotations

from pydantic import Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Upload a file to a sandbox"""

    sandbox_id: str = Field(..., description="Sandbox ID to upload file to")
    local_file: str = Field(..., description="Path to local file to upload")
    remote_path: str = Field(..., description="Path where file should be stored in sandbox")


POSITIONALS = ("sandbox_id", "local_file", "remote_path")


def run(config: Config):
    from prime_cli.commands.sandbox import upload_file as callback

    return callback(config)
