"""Pydantic Config schemas for the ``prime whoami`` surface.

These were previously defined as one-file-per-command leaves under
``prime_cli.leaves``; they now live next to their command callbacks and are
consumed by the router via ``Command.config_attr``.
"""

from __future__ import annotations

from pydantic_config import BaseConfig


class WhoamiConfig(BaseConfig):
    """Show current authenticated user and update config"""

    pass
