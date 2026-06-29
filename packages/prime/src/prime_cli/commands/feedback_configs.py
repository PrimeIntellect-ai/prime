"""Pydantic Config schemas for the ``prime feedback`` surface.

These were previously defined as one-file-per-command leaves under
``prime_cli.leaves``; they now live next to their command callbacks and are
consumed by the router via ``Command.config_attr``.
"""

from __future__ import annotations

from pydantic_config import BaseConfig


class FeedbackConfig(BaseConfig):
    """Submit feedback about Prime Intellect."""

    pass
