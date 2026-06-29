"""Pydantic Config schemas for the ``prime switch`` surface.

These were previously defined as one-file-per-command leaves under
``prime_cli.leaves``; they now live next to their command callbacks and are
consumed by the router via ``Command.config_attr``.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_config import BaseConfig


class SwitchConfig(BaseConfig):
    """Switch between your personal account and team contexts"""

    target: str | None = Field(None, description="'personal', a team slug, or a team ID")
