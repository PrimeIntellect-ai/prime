"""Version check utilities for update notifications."""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import httpx

CACHE_PATH = Path.home() / ".prime" / "version_cache.json"
CACHE_TTL = 3600  # 1 hour
PYPI_URL = "https://pypi.org/pypi/prime/json"


def _is_prerelease(v: str) -> bool:
    return any(tag in v.lower() for tag in ("a", "b", "rc", "dev", "alpha", "beta"))


def _parse_version(v: str) -> Tuple[int, ...]:
    # Strip prerelease suffixes to get base version
    base = re.split(r"(a|b|rc|\.dev|\.post)", v.lower())[0]
    parts = tuple(int(p) for p in base.split(".") if p.isdigit()) or (0,)
    # Suffix: 0=prerelease, 1=stable, 2+=post-release
    if _is_prerelease(v):
        suffix = (0,)
    elif ".post" in v.lower():
        post_match = re.search(r"\.post(\d+)", v.lower())
        suffix = (2 + int(post_match.group(1)) if post_match else 2,)
    else:
        suffix = (1,)
    return parts + suffix


def _get_latest_stable(releases: dict) -> Optional[str]:
    stable = [v for v in releases.keys() if not _is_prerelease(v)]
    if not stable:
        return None
    return max(stable, key=_parse_version)


def check_for_update(current_version: str) -> Optional[str]:
    if os.environ.get("PRIME_NO_UPDATE_NOTIFIER") or not sys.stderr.isatty():
        return None

    # Check cache
    latest = None
    should_fetch = True
    if CACHE_PATH.exists():
        try:
            cache = json.loads(CACHE_PATH.read_text())
            if time.time() - cache.get("checked_at", 0) < CACHE_TTL:
                latest = cache.get("latest_version")
                should_fetch = False
        except (json.JSONDecodeError, OSError, TypeError, AttributeError):
            pass

    # Fetch from PyPI if needed
    if should_fetch:
        try:
            resp = httpx.get(PYPI_URL, timeout=3.0, follow_redirects=True)
            resp.raise_for_status()
            latest = _get_latest_stable(resp.json().get("releases", {}))
        except (httpx.HTTPError, json.JSONDecodeError, KeyError):
            return None
        if latest:
            try:
                CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                CACHE_PATH.write_text(json.dumps({
                    "checked_at": time.time(),
                    "latest_version": latest,
                }))
            except OSError:
                pass

    # Compare versions
    if latest and _parse_version(latest) > _parse_version(current_version):
        return latest
    return None
