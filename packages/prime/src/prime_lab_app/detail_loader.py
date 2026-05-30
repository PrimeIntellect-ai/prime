"""Shared Lab item detail loader types."""

from __future__ import annotations

from collections.abc import Callable

from .models import LabItem

DetailLoader = Callable[[LabItem, bool, int, int, int | None], LabItem]
