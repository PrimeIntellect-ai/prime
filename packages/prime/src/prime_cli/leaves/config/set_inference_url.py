"""Pydantic leaf for ``prime config set-inference-url``."""

from __future__ import annotations

from pydantic import Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Set the inference URL (prompts if not provided)"""

    url: str | None = Field(
        None,
        description="Inference URL for Prime Inference API. If not provided, you'll be prompted.",
    )


POSITIONALS = ("url",)


def run(config: Config):
    from prime_cli.commands.config import set_inference_url as callback

    return callback(config)
