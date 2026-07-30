from typing import Callable

import httpx
import pytest

from prime_traces import TracesAPIClient, TracesClient


@pytest.fixture
def make_client() -> Callable[..., TracesClient]:
    def _make(handler: Callable[[httpx.Request], httpx.Response]) -> TracesClient:
        api_client = TracesAPIClient(
            api_key="test-key",
            base_url="http://testserver",
            # "" forces no team header; None would fall back to the
            # developer's real ~/.prime/config.json.
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
