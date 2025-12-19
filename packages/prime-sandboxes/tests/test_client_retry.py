"""Tests for retry logic on transient connection errors."""

import httpx
import pytest

from prime_sandboxes.core.client import APIClient, APIError, AsyncAPIClient


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


class AsyncFailThenSucceedTransport(httpx.AsyncBaseTransport):
    """Async transport that fails N times then succeeds."""

    def __init__(self, fail_times: int = 2):
        self.fail_times = fail_times
        self.call_count = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.call_count += 1
        if self.call_count <= self.fail_times:
            raise httpx.RemoteProtocolError("Server disconnected")
        return httpx.Response(200, json={"success": True})


class AsyncAlwaysFailTransport(httpx.AsyncBaseTransport):
    """Async transport that always fails."""

    def __init__(self):
        self.call_count = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.call_count += 1
        raise httpx.RemoteProtocolError("Server disconnected")


class TestSyncAPIClientRetry:
    """Tests for sync APIClient retry behavior."""

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


class TestAsyncAPIClientRetry:
    """Tests for async AsyncAPIClient retry behavior."""

    @pytest.mark.asyncio
    async def test_retries_and_succeeds(self):
        """Retries on RemoteProtocolError, succeeds on 3rd attempt."""
        transport = AsyncFailThenSucceedTransport(fail_times=2)
        client = AsyncAPIClient(api_key="test-key")
        client.client = httpx.AsyncClient(transport=transport)

        result = await client.request("GET", "test")

        assert result == {"success": True}
        assert transport.call_count == 3

    @pytest.mark.asyncio
    async def test_gives_up_after_max_retries(self):
        """Raises after 3 failed attempts."""
        transport = AsyncAlwaysFailTransport()
        client = AsyncAPIClient(api_key="test-key")
        client.client = httpx.AsyncClient(transport=transport)

        with pytest.raises(APIError, match="RemoteProtocolError"):
            await client.request("GET", "test")

        assert transport.call_count == 3


class TestSyncGatewayRetry:
    """Tests for sync SandboxClient gateway retry behavior."""

    def test_gateway_retry_decorator_works(self):
        """Test the _gateway_retry decorator directly."""
        from prime_sandboxes.sandbox import _gateway_retry

        call_count = 0

        @_gateway_retry
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.RemoteProtocolError("Server disconnected")
            return "success"

        result = flaky_function()
        assert result == "success"
        assert call_count == 3

    def test_gateway_retry_gives_up(self):
        """Test the _gateway_retry decorator gives up after max retries."""
        from prime_sandboxes.sandbox import _gateway_retry

        call_count = 0

        @_gateway_retry
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise httpx.RemoteProtocolError("Server disconnected")

        with pytest.raises(httpx.RemoteProtocolError):
            always_fail()

        assert call_count == 3

    def test_gateway_retry_on_pool_timeout(self):
        """Test retry on PoolTimeout."""
        from prime_sandboxes.sandbox import _gateway_retry

        call_count = 0

        @_gateway_retry
        def pool_timeout_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.PoolTimeout("No connections available")
            return "success"

        result = pool_timeout_then_succeed()
        assert result == "success"
        assert call_count == 2

    def test_gateway_retry_on_read_timeout(self):
        """Test retry on ReadTimeout."""
        from prime_sandboxes.sandbox import _gateway_retry

        call_count = 0

        @_gateway_retry
        def read_timeout_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.ReadTimeout("Read timed out")
            return "success"

        result = read_timeout_then_succeed()
        assert result == "success"
        assert call_count == 2
