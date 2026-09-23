"""Regression tests: omitted per-request timeouts keep the client default.

The HTTP clients are constructed with a 30-second bound, but ``request()``
used to forward ``timeout=None`` explicitly, which httpx reads as "no
timeout" and disables the constructor default.
"""

import asyncio
from typing import Any

from prime_evals.core import APIClient, AsyncAPIClient


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"ok": True}


class RecordingHttpClient:
    def __init__(self, **kwargs: Any):
        self.request_calls = []

    def request(self, method, url, **kwargs):
        self.request_calls.append((method, url, kwargs))
        return FakeResponse()


def test_request_keeps_configured_timeout_when_no_override_given(monkeypatch):
    api_client = APIClient(api_key="test-key")
    http = RecordingHttpClient()
    monkeypatch.setattr(api_client, "client", http)

    api_client.request("GET", "/evaluations/eval-123")

    method, url, kwargs = http.request_calls[0]
    assert (method, url) == ("GET", f"{api_client.base_url}/api/v1/evaluations/eval-123")
    assert "timeout" not in kwargs, "timeout=None must not disable the client default"


def test_request_forwards_explicit_timeout(monkeypatch):
    api_client = APIClient(api_key="test-key")
    http = RecordingHttpClient()
    monkeypatch.setattr(api_client, "client", http)

    api_client.request("GET", "/evaluations/eval-123", timeout=5)

    _, _, kwargs = http.request_calls[0]
    assert kwargs["timeout"] == 5


def test_get_delegation_keeps_configured_timeout(monkeypatch):
    api_client = APIClient(api_key="test-key")
    http = RecordingHttpClient()
    monkeypatch.setattr(api_client, "client", http)

    api_client.get("/evaluations/eval-123")

    _, _, kwargs = http.request_calls[0]
    assert "timeout" not in kwargs


def test_async_request_keeps_configured_timeout_when_no_override_given():
    api_client = AsyncAPIClient(api_key="test-key")

    class RecordingAsyncHttpClient(RecordingHttpClient):
        async def request(self, method, url, **kwargs):
            return super().request(method, url, **kwargs)

    http = RecordingAsyncHttpClient()
    api_client.client = http

    result = asyncio.run(api_client.request("GET", "/evaluations/eval-123"))

    assert result == {"ok": True}
    _, _, kwargs = http.request_calls[0]
    assert "timeout" not in kwargs, "timeout=None must not disable the client default"


def test_async_request_forwards_explicit_timeout():
    api_client = AsyncAPIClient(api_key="test-key")

    class RecordingAsyncHttpClient(RecordingHttpClient):
        async def request(self, method, url, **kwargs):
            return super().request(method, url, **kwargs)

    http = RecordingAsyncHttpClient()
    api_client.client = http

    asyncio.run(api_client.request("GET", "/evaluations/eval-123", timeout=5))

    _, _, kwargs = http.request_calls[0]
    assert kwargs["timeout"] == 5
