"""Pydantic leaf for ``prime registry check-image``."""

from __future__ import annotations

from pydantic import Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Verify that an image is accessible (optionally using registry credentials)."""

    image: str = Field(..., description="Image reference, e.g. ghcr.io/org/repo:tag")
    registry_credentials_id: str | None = Field(
        None, description="Registry credentials ID for private images"
    )


POSITIONALS = ("image",)


def run(config: Config):
    from prime_cli.commands.registry import check_docker_image as callback

    return callback(config)
