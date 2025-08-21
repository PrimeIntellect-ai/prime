"""Time and date utilities for CLI commands."""

from datetime import datetime, timezone
from typing import Any, List, Union

ISO_FMT = "%Y-%m-%d %H:%M:%S UTC"


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


def sort_by_created(
    items: List[Any], attr: str = "created_at", reverse: bool = False, parse_iso: bool = False
) -> List[Any]:
    """Sort items by creation time (oldest first by default)."""

    def key_fn(x: Any) -> Union[datetime, Any]:
        if parse_iso:
            return datetime.fromisoformat(getattr(x, attr).replace("Z", "+00:00"))
        else:
            return getattr(x, attr)

    return sorted(items, key=key_fn, reverse=reverse)
