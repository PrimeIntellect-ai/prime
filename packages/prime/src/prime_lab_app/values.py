"""Small typed value helpers shared by Lab TUI render modules."""

from __future__ import annotations

import math
from typing import Any

from .models import LabItem


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def int_value(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def finite_float(value: Any) -> float | None:
    if not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def first_number(*values: Any) -> float | None:
    for value in values:
        number = finite_float(value)
        if number is not None:
            return number
    return None


def format_number(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.4g}"


def metadata_value(item: LabItem, key: str) -> str | None:
    for metadata_key, value in item.metadata:
        if metadata_key == key:
            return value
    return None
