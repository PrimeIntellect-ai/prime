"""Pydantic leaf for ``prime config set-api-key``."""

from __future__ import annotations

from pydantic import Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Set your API key (prompts securely if not provided)"""

    api_key: str | None = Field(
        None,
        description="Your Prime Intellect API key. If not provided, you'll be prompted securely.",
    )


POSITIONALS = ("api_key",)


def run(config: Config):
    from prime_cli.commands.config import set_api_key as callback

    return callback(config)
