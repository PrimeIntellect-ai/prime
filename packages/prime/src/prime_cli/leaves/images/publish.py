"""Pydantic leaf for ``prime images publish``."""

from __future__ import annotations

from pydantic import Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Make an image public so other Prime users can run it."""

    image_reference: str = Field(
        ...,
        description="Personal or team image reference to make public.",
    )


POSITIONALS = ("image_reference",)


def run(config: Config):
    from prime_cli.commands.images import publish_image as callback

    return callback(config)
