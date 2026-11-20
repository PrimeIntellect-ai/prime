"""AsyncAPIClient bounds in-flight HTTP attempts without holding a slot across retry backoff."""

import asyncio

import httpx
import pytest

from prime_sandboxes.core import client as client_module
from prime_sandboxes.core.client import AsyncAPIClient


@pytest.mark.asyncio
async def test_in_flight_requests_never_exceed_cap(monkeypatch):
    monkeypatch.setattr(client_module, "MAX_CONCURRENT_API_REQUESTS", 3)
    in_flight = peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return httpx.Response(200, request=request, json={"ok": True})

    client = AsyncAPIClient(api_key="test-key")
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    results = await asyncio.gather(*(client.request("GET", "test") for _ in range(20)))

    assert results == [{"ok": True}] * 20
    assert peak == 3


@pytest.mark.asyncio
async def test_rate_limit_backoff_does_not_hold_a_slot(monkeypatch):
    """With one slot, a request sleeping on a 429 must not block another request."""
    monkeypatch.setattr(client_module, "MAX_CONCURRENT_API_REQUESTS", 1)
    other_done = asyncio.Event()
    calls = {"limited": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/limited"):
            calls["limited"] += 1
            if calls["limited"] == 1:
                return httpx.Response(429, request=request, text="rate limited")
        return httpx.Response(200, request=request, json={"ok": True})

    async def backoff(_delay: float) -> None:
        # The retry only proceeds once the other request got the slot and finished.
        await asyncio.wait_for(other_done.wait(), timeout=2)

    monkeypatch.setattr(AsyncAPIClient._idempotent_request_with_retry.retry, "sleep", backoff)
    client = AsyncAPIClient(api_key="test-key")
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def other():
        await asyncio.sleep(0)  # let the limited request take the slot first
        result = await client.request("GET", "other")
        other_done.set()
        return result

    limited, other_result = await asyncio.gather(client.request("GET", "limited"), other())

    assert limited == {"ok": True} and other_result == {"ok": True}
    assert calls["limited"] == 2
