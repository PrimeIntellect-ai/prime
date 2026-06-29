"""Pydantic Config schemas for the ``prime inference`` surface.

These were previously defined as one-file-per-command leaves under
``prime_cli.leaves``; they now live next to their command callbacks and are
consumed by the router via ``Command.config_attr``.
"""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class InferenceChatConfig(BaseConfig):
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


class InferenceModelsConfig(BaseConfig):
    """List available models from Prime Inference (/v1/models)."""

    output: str = Field(
        "table", validation_alias=AliasChoices("output", "o"), description="table|json"
    )
    search: str | None = Field(
        None,
        validation_alias=AliasChoices("search", "q"),
        description="Case-insensitive substring match on model id",
    )
    sort: str = Field(
        "id", validation_alias=AliasChoices("sort", "s"), description="Sort by: id, input, output"
    )
    order: str = Field(
        "asc",
        validation_alias=AliasChoices("order", "d"),
        description="Sort order (direction): asc, desc",
    )
