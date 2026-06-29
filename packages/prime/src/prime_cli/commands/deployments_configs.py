"""Pydantic Config schemas for the ``prime deployments`` surface.

These were previously defined as one-file-per-command leaves under
``prime_cli.leaves``; they now live next to their command callbacks and are
consumed by the router via ``Command.config_attr``.
"""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class DeploymentsCreateConfig(BaseConfig):
    """Deploy a model for inference."""

    model_id: str = Field(description="Model ID to deploy")
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


class DeploymentsDeleteConfig(BaseConfig):
    """Unload a model from inference."""

    model_id: str = Field(description="Model ID to unload")


class DeploymentsListConfig(BaseConfig):
    """List adapters and their deployment status."""

    team: str | None = Field(
        None, validation_alias=AliasChoices("team", "t"), description="Filter by team ID"
    )
    num: int = Field(20, validation_alias=AliasChoices("num", "n"), description="Items per page")
    page: int = Field(1, validation_alias=AliasChoices("page", "p"), description="Page number")
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )
