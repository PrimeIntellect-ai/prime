"""Tests for version check functionality."""

import json
import time
from pathlib import Path
from unittest.mock import patch

from prime_cli.utils.version_check import _get_latest_stable, check_for_update

MOCK_TTY = patch("sys.stderr.isatty", return_value=True)


class TestGetLatestStable:
    def test_finds_latest_stable_ignoring_prereleases(self) -> None:
        releases = {"0.5.0": [], "0.5.5": [], "0.6.0a1": [], "0.5.6": []}
        assert _get_latest_stable(releases) == "0.5.6"


class TestCheckForUpdate:
    def test_returns_none_when_disabled(self) -> None:
        with patch.dict("os.environ", {"PRIME_NO_UPDATE_NOTIFIER": "1"}):
            assert check_for_update("0.5.5") is None

    def test_uses_cache_when_fresh(self, tmp_path: Path) -> None:
        cache_file = tmp_path / "version_cache.json"
        cache_file.write_text(json.dumps({
            "checked_at": time.time(),
            "latest_version": "0.6.0",
        }))
        with (
            MOCK_TTY,
            patch("prime_cli.utils.version_check.CACHE_PATH", cache_file),
            patch("httpx.get") as mock_get,
        ):
            assert check_for_update("0.5.5") == "0.6.0"
            mock_get.assert_not_called()

    def test_fetches_when_cache_expired(self, tmp_path: Path) -> None:
        cache_file = tmp_path / "version_cache.json"
        cache_file.write_text(json.dumps({
            "checked_at": time.time() - 7200,
            "latest_version": "0.6.0",
        }))
        mock_resp = type("Response", (), {
            "raise_for_status": lambda self: None,
            "json": lambda self: {"releases": {"0.5.5": [], "0.7.0": []}}
        })()
        with (
            MOCK_TTY,
            patch("prime_cli.utils.version_check.CACHE_PATH", cache_file),
            patch("httpx.get", return_value=mock_resp),
        ):
            assert check_for_update("0.5.5") == "0.7.0"
