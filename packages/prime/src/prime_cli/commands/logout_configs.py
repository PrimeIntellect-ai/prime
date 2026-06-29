"""Pydantic Config schemas for the ``prime logout`` surface.

These were previously defined as one-file-per-command leaves under
``prime_cli.leaves``; they now live next to their command callbacks and are
consumed by the router via ``Command.config_attr``.
"""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class LogoutConfig(BaseConfig):
    """Log out of Prime Intellect"""

    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )
