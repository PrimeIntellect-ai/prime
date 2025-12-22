import json
import os
import time
from pathlib import Path
from typing import Tuple

import httpx
from packaging import version

from prime_cli import __version__

PYPI_URL = "https://pypi.org/pypi/prime/json"
CACHE_DURATION = 86400
NOTIFICATION_INTERVAL = 21600
REQUEST_TIMEOUT = 2.0


def _get_cache_file() -> Path:
    """Get the path to the version check cache file."""
    cache_dir = Path.home() / ".prime"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / "version_check.json"


def _read_cache() -> dict | None:
    """Read the cached version check data."""
    cache_file = _get_cache_file()
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text())
        return data
    except (json.JSONDecodeError, OSError):
        return None


def _write_cache(
    latest_version: str, last_notified: float | None = None, preserve_notified: bool = False
) -> None:
    """Write version check data to cache."""
    cache_file = _get_cache_file()
    try:
        existing_notified = None
        if preserve_notified:
            existing_cache = _read_cache()
            if existing_cache:
                existing_notified = existing_cache.get("last_notified")

        cache_data = {
            "last_check": time.time(),
            "latest_version": latest_version,
            "last_notified": last_notified if last_notified is not None else existing_notified,
        }
        cache_file.write_text(json.dumps(cache_data))
    except OSError:
        pass


def _mark_notified(latest_version: str) -> None:
    """Mark that we've shown the update notification."""
    cache_file = _get_cache_file()
    try:
        cache_data = _read_cache() or {}
        cache_data["last_notified"] = time.time()
        cache_data["latest_version"] = latest_version
        if "last_check" not in cache_data:
            cache_data["last_check"] = time.time()
        cache_file.write_text(json.dumps(cache_data))
    except OSError:
        pass


def _should_show_notification(cache_data: dict | None) -> bool:
    """Check if enough time has passed since the last notification."""
    if cache_data is None:
        return True
    last_notified = cache_data.get("last_notified")
    if last_notified is None:
        return True
    return (time.time() - last_notified) >= NOTIFICATION_INTERVAL


def _is_cache_valid(cache_data: dict) -> bool:
    """Check if the cache is still valid (within cache duration)."""
    last_check = cache_data.get("last_check", 0)
    return (time.time() - last_check) < CACHE_DURATION


def get_latest_pypi_version() -> str | None:
    """Fetch the latest version from PyPI."""
    try:
        response = httpx.get(PYPI_URL, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        return data["info"]["version"]
    except (
        httpx.RequestError,
        httpx.HTTPStatusError,
        KeyError,
        TypeError,
        AttributeError,
        json.JSONDecodeError,
    ):
        return None


def check_for_update() -> Tuple[bool, str | None]:
    """Check if a newer version of prime is available on PyPI."""
    if os.environ.get("PRIME_DISABLE_VERSION_CHECK", "").lower() in ("1", "true", "yes"):
        return (False, None)

    try:
        cache_data = _read_cache()
        if cache_data and _is_cache_valid(cache_data):
            latest_version = cache_data.get("latest_version")
        else:
            latest_version = get_latest_pypi_version()
            if latest_version:
                _write_cache(latest_version, preserve_notified=True)
                cache_data = _read_cache()

        if not latest_version:
            return (False, None)

        installed = version.parse(__version__)
        latest = version.parse(latest_version)

        if installed < latest:
            if _should_show_notification(cache_data):
                _mark_notified(latest_version)
                return (True, latest_version)
            return (False, latest_version)

        return (False, latest_version)

    except Exception:
        return (False, None)
