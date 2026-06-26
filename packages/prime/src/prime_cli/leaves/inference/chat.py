"""Pydantic leaf for ``prime inference chat``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Send a one-shot chat message to a Prime Inference model."""

    model: str = Field(..., description="Model id (see `prime inference models`)")
    message: str | None = Field(None, description="User message. If omitted, reads from stdin.")
    system: str | None = Field(
        None, validation_alias=AliasChoices("system", "s"), description="System prompt"
    )
    stream: bool = Field(False, description="Stream tokens as they arrive")
    temperature: float | None = Field(
        None, validation_alias=AliasChoices("temperature", "t"), description="Sampling temperature"
    )
    max_tokens: int | None = Field(None, description="Maximum tokens to generate")
    output: str = Field(
        "text", validation_alias=AliasChoices("output", "o"), description="text|json"
    )


POSITIONALS = ("model", "message")


def run(config: Config):
    from prime_cli.commands.inference import chat as callback

    return callback(config)
