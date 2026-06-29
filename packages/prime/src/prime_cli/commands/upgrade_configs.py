"""Pydantic Config schemas for the ``prime upgrade`` surface.

These were previously defined as one-file-per-command leaves under
``prime_cli.leaves``; they now live next to their command callbacks and are
consumed by the router via ``Command.config_attr``.
"""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class UpgradeConfig(BaseConfig):
    """Upgrade the Prime CLI to the latest version"""

    check: bool = Field(
        False,
        validation_alias=AliasChoices("check", "c"),
        description="Only check for updates, don't upgrade",
    )
    force: bool = Field(
        False,
        validation_alias=AliasChoices("force", "f"),
        description="Force upgrade even if already on latest version",
    )
