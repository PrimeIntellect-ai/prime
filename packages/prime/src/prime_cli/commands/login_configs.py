"""Pydantic Config schemas for the ``prime login`` surface.

These were previously defined as one-file-per-command leaves under
``prime_cli.leaves``; they now live next to their command callbacks and are
consumed by the router via ``Command.config_attr``.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_config import BaseConfig


class LoginConfig(BaseConfig):
    """Login to Prime Intellect"""

    headless: bool = Field(False, description="Don't attempt to open browser")
