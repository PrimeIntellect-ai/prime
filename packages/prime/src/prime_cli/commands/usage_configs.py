"""Pydantic Config schemas for the ``prime usage`` surface.

These were previously defined as one-file-per-command leaves under
``prime_cli.leaves``; they now live next to their command callbacks and are
consumed by the router via ``Command.config_attr``.
"""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class TrainUsageConfig(BaseConfig):
    """Show token usage and price for a single training run."""

    run_id: str = Field(..., description="RFT run ID (e.g. rft_...")
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )
    watch: bool = Field(
        False,
        validation_alias=AliasChoices("watch", "w"),
        description="Poll continuously and update in place",
    )
    interval: int = Field(
        30,
        validation_alias=AliasChoices("interval", "n"),
        description="Seconds between polls when --watch is set",
    )
