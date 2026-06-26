"""Pydantic leaf for ``prime images unpublish``."""

from __future__ import annotations

from pydantic import Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Make a public image private again."""

    image_reference: str = Field(
        ...,
        description="Personal or team image reference to make private.",
    )


POSITIONALS = ("image_reference",)


def run(config: Config):
    from prime_cli.commands.images import unpublish_image as callback

    return callback(config)
