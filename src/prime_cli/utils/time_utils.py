"""Time and date utilities for CLI commands."""

from datetime import datetime, timezone
from typing import Any, List, Union

ISO_FMT: str = "%Y-%m-%d %H:%M:%S UTC"


def now_utc() -> datetime:
    """Get current UTC time."""
    return datetime.now(timezone.utc)


def to_utc(dt: datetime) -> datetime:
    """Convert datetime to UTC, adding timezone if naive."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def human_age(created: datetime) -> str:
    """Format time difference as human-readable age (like kubectl)."""
    diff = now_utc() - to_utc(created)
    total_seconds = int(diff.total_seconds())

    if total_seconds < 60:
        return f"{total_seconds}s"
    elif total_seconds < 3600:
        minutes = total_seconds // 60
        return f"{minutes}m"
    elif total_seconds < 86400:
        hours = total_seconds // 3600
        return f"{hours}h"
    else:
        days = total_seconds // 86400
        return f"{days}d"


def iso_timestamp(dt: Union[datetime, str]) -> str:
    """Convert datetime or ISO string to standardized timestamp format."""
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
    return to_utc(dt).strftime(ISO_FMT)


def sort_by_created(items: List[Any], attr: str = "created_at", reverse: bool = False) -> List[Any]:
    """Sort items by creation time (oldest first by default).

    Handles datetime objects, ISO strings, and other types with robust fallback.
    """

    def get_sort_key(item: Any) -> datetime:
        value = getattr(item, attr, None)

        if value is None:
            return datetime.min.replace(tzinfo=timezone.utc)

        if isinstance(value, datetime):
            return to_utc(value)

        if isinstance(value, str):
            try:
                # Handle ISO strings with or without 'Z' suffix
                iso_string = value.replace("Z", "+00:00")
                return to_utc(datetime.fromisoformat(iso_string))
            except (ValueError, TypeError):
                # If parsing fails, fall back to minimum datetime
                return datetime.min.replace(tzinfo=timezone.utc)

        # For any other type, use minimum datetime as fallback
        return datetime.min.replace(tzinfo=timezone.utc)

    return sorted(items, key=get_sort_key, reverse=reverse)
