"""Pydantic leaf for ``prime images delete``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Delete an image from your registry."""

    image_reference: str = Field(
        ...,
        description="Personal or team image reference to delete.",
    )
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


POSITIONALS = ("image_reference",)


def run(config: Config):
    from prime_cli.commands.images import delete_image as callback

    return callback(config)
