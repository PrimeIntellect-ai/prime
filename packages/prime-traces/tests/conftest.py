from pathlib import Path
from typing import Callable

import httpx
import pytest

from prime_traces import TracesAPIClient, TracesClient

_PRIME_ENV_VARS = (
    "PRIME_API_KEY",
    "PRIME_TEAM_ID",
    "PRIME_TRACES_URL",
    "PRIME_API_BASE_URL",
    "PRIME_BASE_URL",
)


@pytest.fixture(autouse=True)
def isolated_prime_config(monkeypatch, tmp_path):
    """Keep tests hermetic: never read the developer's real ~/.prime or env.

    `TracesAPIClient` constructs a `Config` even when every parameter is
    explicit, so without this a malformed local config.json (or a PRIME_* var
    in the developer's shell) leaks into — or crashes — unrelated tests.
    Mirrors the home-isolation fixture in prime-sandboxes' conftest.
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for name in _PRIME_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return tmp_path


@pytest.fixture
def make_client() -> Callable[..., TracesClient]:
    def _make(handler: Callable[[httpx.Request], httpx.Response]) -> TracesClient:
        api_client = TracesAPIClient(
            api_key="test-key",
            base_url="http://testserver",
            # "" forces no team header rather than resolving from config.
            team_id="",
            transport=httpx.MockTransport(handler),
        )
        return TracesClient(api_client=api_client)

    return _make


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Record retry sleeps instead of actually sleeping."""
    sleeps: list = []
    monkeypatch.setattr("prime_traces.traces.time.sleep", sleeps.append)
    return sleeps
