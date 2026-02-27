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

        assert call_count == 4

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

    def test_gateway_retry_on_connect_error(self):
        """Test retry on ConnectError."""
        from prime_sandboxes.sandbox import _gateway_retry

        call_count = 0

        @_gateway_retry
        def connect_error_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.ConnectError("Connection refused")
            return "success"

        result = connect_error_then_succeed()
        assert result == "success"
        assert call_count == 2

    def test_gateway_retry_on_5xx(self):
        """Test retry on 5xx HTTPStatusError (e.g. Cloudflare 524 timeout)."""
        from prime_sandboxes.sandbox import _gateway_retry

        call_count = 0

        @_gateway_retry
        def server_error_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                response = httpx.Response(524, request=httpx.Request("POST", "http://test"))
                raise httpx.HTTPStatusError(
                    "524 timeout", request=response.request, response=response
                )
            return "success"

        result = server_error_then_succeed()
        assert result == "success"
        assert call_count == 3

    def test_gateway_retry_on_502(self):
        """Test retry on 502 Bad Gateway."""
        from prime_sandboxes.sandbox import _gateway_retry

        call_count = 0

        @_gateway_retry
        def bad_gateway_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                response = httpx.Response(502, request=httpx.Request("POST", "http://test"))
                raise httpx.HTTPStatusError("502", request=response.request, response=response)
            return "success"

        result = bad_gateway_then_succeed()
        assert result == "success"
        assert call_count == 2

    def test_gateway_no_retry_on_502_sandbox_not_found(self):
        """Test that 502 with sandbox_not_found body is NOT retried (permanent error)."""
        import json

        from prime_sandboxes.sandbox import _gateway_retry

        call_count = 0

        @_gateway_retry
        def sandbox_gone():
            nonlocal call_count
            call_count += 1
            body = json.dumps({"error": "sandbox_not_found"}).encode()
            response = httpx.Response(
                502,
                request=httpx.Request("GET", "http://test"),
                content=body,
                headers={"content-type": "application/json"},
            )
            raise httpx.HTTPStatusError("502", request=response.request, response=response)

        with pytest.raises(httpx.HTTPStatusError):
            sandbox_gone()

        assert call_count == 1  # no retry — permanent error

    def test_gateway_no_retry_on_404(self):
        """Test that non-retryable status codes are NOT retried."""
        from prime_sandboxes.sandbox import _gateway_retry

        call_count = 0

        @_gateway_retry
        def not_found():
            nonlocal call_count
            call_count += 1
            response = httpx.Response(404, request=httpx.Request("GET", "http://test"))
            raise httpx.HTTPStatusError("404", request=response.request, response=response)

        with pytest.raises(httpx.HTTPStatusError):
            not_found()

        assert call_count == 1  # no retry


class TestSyncGatewayPostRetry:
    """Tests for _gateway_post_retry — retries connection errors but NOT 5xx."""

    def test_post_retry_on_connect_error(self):
        """Connection errors are retried (request was never sent)."""
        from prime_sandboxes.sandbox import _gateway_post_retry

        call_count = 0

        @_gateway_post_retry
        def connect_error_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.ConnectError("Connection refused")
            return "success"

        result = connect_error_then_succeed()
        assert result == "success"
        assert call_count == 2

    def test_post_retry_on_remote_protocol_error(self):
        """RemoteProtocolError is retried."""
        from prime_sandboxes.sandbox import _gateway_post_retry

        call_count = 0

        @_gateway_post_retry
        def protocol_error_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.RemoteProtocolError("Server disconnected")
            return "success"

        result = protocol_error_then_succeed()
        assert result == "success"
        assert call_count == 3

    def test_post_retry_gives_up(self):
        """Gives up after max attempts on connection errors."""
        from prime_sandboxes.sandbox import _gateway_post_retry

        call_count = 0

        @_gateway_post_retry
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise httpx.PoolTimeout("No connections available")

        with pytest.raises(httpx.PoolTimeout):
            always_fail()

        assert call_count == 4

    def test_post_no_retry_on_5xx(self):
        """5xx errors are NOT retried for POST (server received the request)."""
        from prime_sandboxes.sandbox import _gateway_post_retry

        call_count = 0

        @_gateway_post_retry
        def server_error():
            nonlocal call_count
            call_count += 1
            response = httpx.Response(502, request=httpx.Request("POST", "http://test"))
            raise httpx.HTTPStatusError("502", request=response.request, response=response)

        with pytest.raises(httpx.HTTPStatusError):
            server_error()

        assert call_count == 1  # no retry

    def test_post_no_retry_on_524(self):
        """Cloudflare 524 timeout is NOT retried for POST."""
        from prime_sandboxes.sandbox import _gateway_post_retry

        call_count = 0

        @_gateway_post_retry
        def cf_timeout():
            nonlocal call_count
            call_count += 1
            response = httpx.Response(524, request=httpx.Request("POST", "http://test"))
            raise httpx.HTTPStatusError("524", request=response.request, response=response)

        with pytest.raises(httpx.HTTPStatusError):
            cf_timeout()

        assert call_count == 1  # no retry
