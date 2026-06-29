"""Pydantic Config schemas for the ``prime wallet`` surface.

These were previously defined as one-file-per-command leaves under
``prime_cli.leaves``; they now live next to their command callbacks and are
consumed by the router via ``Command.config_attr``.
"""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class WalletConfig(BaseConfig):
    """Show wallet balance and most recent billing rows."""

    limit: int = Field(
        20,
        validation_alias=AliasChoices("limit", "n"),
        description="Number of recent billing rows to fetch (max 100)",
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )
