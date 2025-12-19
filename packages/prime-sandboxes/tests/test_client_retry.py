"""Tests for retry logic on transient connection errors."""

import httpx
import pytest

from prime_sandboxes.core.client import APIClient, APIError


class FailThenSucceedTransport(httpx.BaseTransport):
    """Transport that fails N times then succeeds."""

    def __init__(self, fail_times: int = 2):
        self.fail_times = fail_times
        self.call_count = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.call_count += 1
        if self.call_count <= self.fail_times:
            raise httpx.RemoteProtocolError("Server disconnected")
        return httpx.Response(200, json={"success": True})


class AlwaysFailTransport(httpx.BaseTransport):
    """Transport that always fails."""

    def __init__(self):
        self.call_count = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.call_count += 1
        raise httpx.RemoteProtocolError("Server disconnected")


class TestClientRetry:
    def test_retries_and_succeeds(self):
        """Retries on RemoteProtocolError, succeeds on 3rd attempt."""
        transport = FailThenSucceedTransport(fail_times=2)
        client = APIClient(api_key="test-key")
        client.client = httpx.Client(transport=transport)

        result = client.request("GET", "test")

        assert result == {"success": True}
        assert transport.call_count == 3

    def test_gives_up_after_max_retries(self):
        """Raises after 3 failed attempts."""
        transport = AlwaysFailTransport()
        client = APIClient(api_key="test-key")
        client.client = httpx.Client(transport=transport)

        with pytest.raises(APIError, match="RemoteProtocolError"):
            client.request("GET", "test")

        assert transport.call_count == 3
