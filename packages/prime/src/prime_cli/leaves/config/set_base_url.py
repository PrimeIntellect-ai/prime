"""Pydantic leaf for ``prime config set-base-url``."""

from __future__ import annotations

from pydantic import Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Set the API base URL (prompts if not provided)"""

    url: str | None = Field(
        None,
        description="Base URL for the Prime Intellect API. If not provided, you'll be prompted.",
    )


POSITIONALS = ("url",)


def run(config: Config):
    from prime_cli.commands.config import set_base_url as callback

    return callback(config)
