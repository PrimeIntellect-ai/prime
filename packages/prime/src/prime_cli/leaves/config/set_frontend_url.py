"""Pydantic leaf for ``prime config set-frontend-url``."""

from __future__ import annotations

from pydantic import Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Set the frontend URL (prompts if not provided)"""

    url: str | None = Field(
        None,
        description="Prime Intellect web app URL. Prompts when omitted.",
    )


POSITIONALS = ("url",)


def run(config: Config):
    from prime_cli.commands.config import set_frontend_url as callback

    return callback(config)
