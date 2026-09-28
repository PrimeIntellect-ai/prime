"""Tests for retry logic on transient connection errors."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from prime_sandboxes.core.client import APIClient, APIError, AsyncAPIClient
from prime_sandboxes.models import CreateSandboxRequest
from prime_sandboxes.sandbox import (
    AsyncSandboxAuthCache,
    AsyncSandboxClient,
    SandboxAuthCache,
    SandboxClient,
)


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


class ReadErrorThenSucceedTransport(httpx.BaseTransport):
    """Transport that raises ReadError once then succeeds."""

    def __init__(self):
        self.call_count = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.call_count += 1
        if self.call_count == 1:
            raise httpx.ReadError("Connection broken", request=request)
        return httpx.Response(200, json={"success": True})


class StatusThenSucceedTransport(httpx.BaseTransport):
    """Transport that returns an HTTP status once then succeeds."""

    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self.payload = payload or {"success": True}
        self.call_count = 0
        self.requests: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.call_count += 1
        self.requests.append(request)
        if self.call_count == 1:
            return httpx.Response(self.status_code, request=request, text="Bad Gateway")
        return httpx.Response(200, request=request, json=self.payload)


class AsyncReadErrorThenSucceedTransport(httpx.AsyncBaseTransport):
    """Async transport that raises ReadError once then succeeds."""

    def __init__(self):
        self.call_count = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.call_count += 1
        if self.call_count == 1:
            raise httpx.ReadError("Connection broken", request=request)
        return httpx.Response(200, json={"success": True})


class AsyncStatusThenSucceedTransport(httpx.AsyncBaseTransport):
    """Async transport that returns an HTTP status once then succeeds."""

    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self.payload = payload or {"success": True}
        self.call_count = 0
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.call_count += 1
        self.requests.append(request)
        if self.call_count == 1:
            return httpx.Response(self.status_code, request=request, text="Bad Gateway")
        return httpx.Response(200, request=request, json=self.payload)


def _sandbox_response(sandbox_id: str = "sandbox-1"):
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": sandbox_id,
        "name": sandbox_id,
        "docker_image": "python:3.11-slim",
        "cpu_cores": 1.0,
        "memory_gb": 2.0,
        "disk_size_gb": 5.0,
        "disk_mount_path": "/workspace",
        "gpu_count": 0,
        "status": "RUNNING",
        "timeout_minutes": 60,
        "created_at": now,
        "updated_at": now,
    }


def _auth_response():
    return {
        "gateway_url": "https://gateway.example",
        "token": "test-token",
        "user_ns": "test-ns",
        "job_id": "job-123",
        "expires_at": "2099-01-01T00:00:00Z",
    }


class RecordingCreateAPIClient:
    def __init__(self):
        self.config = SimpleNamespace(team_id="team-1")
        self.calls = []

    def request(self, method: str, path: str, **kwargs):
        self.calls.append((method, path, kwargs))
        return _sandbox_response(f"sandbox-{len(self.calls)}")


class AsyncRecordingCreateAPIClient:
    def __init__(self):
        self.config = SimpleNamespace(team_id="team-1")
        self.calls = []

    async def request(self, method: str, path: str, **kwargs):
        self.calls.append((method, path, kwargs))
        return _sandbox_response(f"sandbox-{len(self.calls)}")


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

    def test_rate_limit_retries_use_long_waits(self, monkeypatch):
        """Platform rate limits wait 10s then 30s across three attempts."""
        call_count = 0
        delays = []

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return httpx.Response(429, request=request, text="rate limited")
            return httpx.Response(200, request=request, json={"success": True})

        monkeypatch.setattr(APIClient._idempotent_request_with_retry.retry, "sleep", delays.append)
        client = APIClient(api_key="test-key")
        client.client = httpx.Client(transport=httpx.MockTransport(handler))

        result = client.request("GET", "test")

        assert result == {"success": True}
        assert call_count == 3
        assert delays == [10.0, 30.0]

    def test_connection_error_retry_keeps_tight_wait(self, monkeypatch):
        """Connection errors keep the existing fast first retry."""
        call_count = 0
        delays = []

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ConnectError("Connection refused", request=request)
            return httpx.Response(200, request=request, json={"success": True})

        monkeypatch.setattr(APIClient._idempotent_request_with_retry.retry, "sleep", delays.append)
        client = APIClient(api_key="test-key")
        client.client = httpx.Client(transport=httpx.MockTransport(handler))

        result = client.request("GET", "test")

        assert result == {"success": True}
        assert call_count == 2
        assert len(delays) == 1
        assert 0 <= delays[0] <= 0.1

    def test_gives_up_after_max_retries(self):
        """Raises after 3 failed attempts."""
        transport = AlwaysFailTransport()
        client = APIClient(api_key="test-key")
        client.client = httpx.Client(transport=transport)

        with pytest.raises(APIError, match="RemoteProtocolError"):
            client.request("GET", "test")

        assert transport.call_count == 3

    def test_get_retries_read_error(self):
        """GET retries ReadError because it is idempotent."""
        transport = ReadErrorThenSucceedTransport()
        client = APIClient(api_key="test-key")
        client.client = httpx.Client(transport=transport)

        result = client.request("GET", "test")

        assert result == {"success": True}
        assert transport.call_count == 2

    @pytest.mark.parametrize("status_code", [502, 503, 504])
    def test_get_retries_transient_gateway_statuses(self, status_code):
        """GET retries transient gateway failures from the public API/LB path."""
        transport = StatusThenSucceedTransport(status_code)
        client = APIClient(api_key="test-key")
        client.client = httpx.Client(transport=transport)

        result = client.request("GET", "sandbox/sandbox-1")

        assert result == {"success": True}
        assert transport.call_count == 2

    def test_patch_does_not_retry_read_error(self):
        """PATCH does not retry ReadError because the server may have committed."""
        transport = ReadErrorThenSucceedTransport()
        client = APIClient(api_key="test-key")
        client.client = httpx.Client(transport=transport)

        with pytest.raises(APIError, match="ReadError"):
            client.request("PATCH", "test", json={"name": "sandbox"})

        assert transport.call_count == 1

    @pytest.mark.parametrize("status_code", [502, 503, 504])
    def test_patch_does_not_retry_transient_gateway_statuses(self, status_code):
        """PATCH does not retry transient gateway statuses without explicit idempotency."""
        transport = StatusThenSucceedTransport(status_code)
        client = APIClient(api_key="test-key")
        client.client = httpx.Client(transport=transport)

        with pytest.raises(APIError, match=f"HTTP {status_code}"):
            client.request("PATCH", "test", json={"name": "sandbox"})

        assert transport.call_count == 1

    def test_post_does_not_retry_read_error(self):
        """POST does not retry ReadError because the server may have committed."""
        transport = ReadErrorThenSucceedTransport()
        client = APIClient(api_key="test-key")
        client.client = httpx.Client(transport=transport)

        with pytest.raises(APIError, match="ReadError"):
            client.request("POST", "test", json={"name": "sandbox"})

        assert transport.call_count == 1

    def test_idempotent_post_retries_read_error(self):
        """Idempotency-keyed POST retries ReadError safely."""
        transport = ReadErrorThenSucceedTransport()
        client = APIClient(api_key="test-key")
        client.client = httpx.Client(transport=transport)

        result = client.request(
            "POST",
            "test",
            json={"name": "sandbox", "idempotency_key": "key-1"},
            idempotent_post=True,
        )

        assert result == {"success": True}
        assert transport.call_count == 2

    @pytest.mark.parametrize("status_code", [502, 503, 504])
    def test_idempotent_post_retries_transient_gateway_statuses(self, status_code):
        """Idempotency-keyed POST retries transient gateway failures."""
        transport = StatusThenSucceedTransport(status_code)
        client = APIClient(api_key="test-key")
        client.client = httpx.Client(transport=transport)

        result = client.request(
            "POST",
            "test",
            json={"name": "sandbox", "idempotency_key": "key-1"},
            idempotent_post=True,
        )

        assert result == {"success": True}
        assert transport.call_count == 2

    def test_post_does_not_retry_502(self):
        """Generic POST still does not retry 502."""
        transport = StatusThenSucceedTransport(502)
        client = APIClient(api_key="test-key")
        client.client = httpx.Client(transport=transport)

        with pytest.raises(APIError, match="HTTP 502"):
            client.request("POST", "test", json={"name": "sandbox"})

        assert transport.call_count == 1

    def test_get_with_idempotent_post_flag_still_raises_status_error(self):
        """idempotent_post=True must not bypass status checks for non-POST requests."""
        transport = StatusThenSucceedTransport(404)
        client = APIClient(api_key="test-key")
        client.client = httpx.Client(transport=transport)

        with pytest.raises(APIError, match="HTTP 404"):
            client.request("GET", "test", idempotent_post=True)

        assert transport.call_count == 1


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

    @pytest.mark.parametrize("status_code", [502, 503, 504])
    @pytest.mark.asyncio
    async def test_get_retries_transient_gateway_statuses(self, status_code):
        """Async GET retries transient gateway failures from the public API/LB path."""
        transport = AsyncStatusThenSucceedTransport(status_code)
        client = AsyncAPIClient(api_key="test-key")
        client.client = httpx.AsyncClient(transport=transport)

        result = await client.request("GET", "sandbox/sandbox-1")

        assert result == {"success": True}
        assert transport.call_count == 2

    @pytest.mark.asyncio
    async def test_patch_does_not_retry_read_error(self):
        """Async PATCH does not retry ReadError because the server may have committed."""
        transport = AsyncReadErrorThenSucceedTransport()
        client = AsyncAPIClient(api_key="test-key")
        client.client = httpx.AsyncClient(transport=transport)

        with pytest.raises(APIError, match="ReadError"):
            await client.request("PATCH", "test", json={"name": "sandbox"})

        assert transport.call_count == 1

    @pytest.mark.parametrize("status_code", [502, 503, 504])
    @pytest.mark.asyncio
    async def test_patch_does_not_retry_transient_gateway_statuses(self, status_code):
        """Async PATCH does not retry transient gateway statuses without explicit idempotency."""
        transport = AsyncStatusThenSucceedTransport(status_code)
        client = AsyncAPIClient(api_key="test-key")
        client.client = httpx.AsyncClient(transport=transport)

        with pytest.raises(APIError, match=f"HTTP {status_code}"):
            await client.request("PATCH", "test", json={"name": "sandbox"})

        assert transport.call_count == 1

    @pytest.mark.asyncio
    async def test_idempotent_post_retries_read_error(self):
        """Async idempotency-keyed POST retries ReadError safely."""
        transport = AsyncReadErrorThenSucceedTransport()
        client = AsyncAPIClient(api_key="test-key")
        client.client = httpx.AsyncClient(transport=transport)

        result = await client.request(
            "POST",
            "test",
            json={"name": "sandbox", "idempotency_key": "key-1"},
            idempotent_post=True,
        )

        assert result == {"success": True}
        assert transport.call_count == 2

    @pytest.mark.asyncio
    async def test_idempotent_post_retries_502(self):
        """Async idempotency-keyed POST retries transient gateway failures."""
        transport = AsyncStatusThenSucceedTransport(502)
        client = AsyncAPIClient(api_key="test-key")
        client.client = httpx.AsyncClient(transport=transport)

        result = await client.request(
            "POST",
            "test",
            json={"name": "sandbox", "idempotency_key": "key-1"},
            idempotent_post=True,
        )

        assert result == {"success": True}
        assert transport.call_count == 2

    @pytest.mark.asyncio
    async def test_get_with_idempotent_post_flag_still_raises_status_error(self):
        """Async idempotent_post=True must not bypass status checks for non-POST requests."""
        transport = AsyncStatusThenSucceedTransport(404)
        client = AsyncAPIClient(api_key="test-key")
        client.client = httpx.AsyncClient(transport=transport)

        with pytest.raises(APIError, match="HTTP 404"):
            await client.request("GET", "test", idempotent_post=True)

        assert transport.call_count == 1


class RecordingAPIClient(APIClient):
    """Real-config APIClient that records requests instead of hitting the wire."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    def request(self, method: str, path: str, **kwargs):
        self.calls.append((method, path, kwargs))
        return _sandbox_response(f"sandbox-{len(self.calls)}")


class TestCreateSandboxTeamContextFromConfig:
    def test_sync_create_omits_team_id_when_prime_team_id_env_is_empty(self, monkeypatch, tmp_path):
        # PRIME_TEAM_ID="" means personal scope: the create payload must not
        # carry team_id: "", which the backend rejects as an unknown wallet.
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        for variable in ("PRIME_CONTEXT", "PRIME_TEAM_ID", "PRIME_USER_ID"):
            monkeypatch.delenv(variable, raising=False)
        monkeypatch.setenv("PRIME_TEAM_ID", "")

        recording = RecordingAPIClient(api_key="test-key")
        assert recording.config.team_id is None

        SandboxClient(recording).create(
            CreateSandboxRequest(name="sandbox", docker_image="python:3.11-slim")
        )

        payload = recording.calls[0][2]["json"]
        assert "team_id" not in payload

    def test_sync_create_keeps_team_id_from_env_when_set(self, monkeypatch, tmp_path):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        for variable in ("PRIME_CONTEXT", "PRIME_TEAM_ID", "PRIME_USER_ID"):
            monkeypatch.delenv(variable, raising=False)
        monkeypatch.setenv("PRIME_TEAM_ID", "team-77")

        recording = RecordingAPIClient(api_key="test-key")

        SandboxClient(recording).create(
            CreateSandboxRequest(name="sandbox", docker_image="python:3.11-slim")
        )

        payload = recording.calls[0][2]["json"]
        assert payload["team_id"] == "team-77"


class TestCreateSandboxIdempotencyPayload:
    def test_sync_create_does_not_reuse_generated_idempotency_key_on_request_reuse(self):
        client = SandboxClient(APIClient(api_key="test-key"))
        recording = RecordingCreateAPIClient()
        client.client = recording
        request = CreateSandboxRequest(name="sandbox", docker_image="python:3.11-slim")

        client.create(request)
        client.create(request)

        first_payload = recording.calls[0][2]["json"]
        second_payload = recording.calls[1][2]["json"]
        assert first_payload["idempotency_key"] != second_payload["idempotency_key"]
        assert first_payload["team_id"] == "team-1"
        assert second_payload["team_id"] == "team-1"
        assert request.idempotency_key is None
        assert request.team_id is None

    @pytest.mark.asyncio
    async def test_async_create_does_not_reuse_generated_idempotency_key_on_request_reuse(self):
        client = AsyncSandboxClient(api_key="test-key")
        recording = AsyncRecordingCreateAPIClient()
        client.client = recording
        request = CreateSandboxRequest(name="sandbox", docker_image="python:3.11-slim")

        await client.create(request)
        await client.create(request)

        first_payload = recording.calls[0][2]["json"]
        second_payload = recording.calls[1][2]["json"]
        assert first_payload["idempotency_key"] != second_payload["idempotency_key"]
        assert first_payload["team_id"] == "team-1"
        assert second_payload["team_id"] == "team-1"
        assert request.idempotency_key is None
        assert request.team_id is None


class TestSandboxAuthRetry:
    def test_sync_auth_refresh_retries_transient_gateway_status(self, tmp_path):
        """Auth POST is idempotent and retries transient public API 502s."""
        transport = StatusThenSucceedTransport(502, payload=_auth_response())
        client = APIClient(api_key="test-key")
        client.client = httpx.Client(transport=transport)
        cache = SandboxAuthCache(tmp_path / "auth_cache.json", client)

        result = cache.get_or_refresh("sandbox-1")

        assert result["token"] == "test-token"
        assert transport.call_count == 2
        assert [request.method for request in transport.requests] == ["POST", "POST"]
        assert str(transport.requests[0].url).endswith("/api/v1/sandbox/sandbox-1/auth")

    @pytest.mark.asyncio
    async def test_async_auth_refresh_retries_transient_gateway_status(self, tmp_path):
        """Async auth POST is idempotent and retries transient public API 502s."""
        transport = AsyncStatusThenSucceedTransport(502, payload=_auth_response())
        client = AsyncAPIClient(api_key="test-key")
        client.client = httpx.AsyncClient(transport=transport)
        cache = AsyncSandboxAuthCache(tmp_path / "auth_cache.json", client)

        result = await cache.get_or_refresh("sandbox-1")

        assert result["token"] == "test-token"
        assert transport.call_count == 2
        assert [request.method for request in transport.requests] == ["POST", "POST"]
        assert str(transport.requests[0].url).endswith("/api/v1/sandbox/sandbox-1/auth")
        await client.client.aclose()


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

    def test_gateway_retry_on_connect_error(self, monkeypatch):
        """Connection errors retain the tight retry curve."""
        from prime_sandboxes.sandbox import _gateway_retry

        call_count = 0
        delays = []

        @_gateway_retry
        def connect_error_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.ConnectError("Connection refused")
            return "success"

        monkeypatch.setattr(connect_error_then_succeed.retry, "sleep", delays.append)

        result = connect_error_then_succeed()
        assert result == "success"
        assert call_count == 2
        assert delays == [1.0]

    def test_gateway_retry_on_429_uses_long_waits(self, monkeypatch):
        """Gateway rate limits use the longer inter-attempt delays."""
        from prime_sandboxes.sandbox import _gateway_retry

        call_count = 0
        delays = []

        @_gateway_retry
        def rate_limit_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                response = httpx.Response(429, request=httpx.Request("GET", "http://test"))
                raise httpx.HTTPStatusError(
                    "429 rate limited", request=response.request, response=response
                )
            return "success"

        monkeypatch.setattr(rate_limit_then_succeed.retry, "sleep", delays.append)

        result = rate_limit_then_succeed()
        assert result == "success"
        assert call_count == 4
        assert delays == [10.0, 30.0, 40.0]

    def test_gateway_retry_on_read_error(self):
        """Test retry on ReadError (TCP drop mid-response) — idempotent path."""
        from prime_sandboxes.sandbox import _gateway_retry

        call_count = 0

        @_gateway_retry
        def read_error_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.ReadError(
                    "Connection broken", request=httpx.Request("GET", "http://test")
                )
            return "success"

        result = read_error_then_succeed()
        assert result == "success"
        assert call_count == 2

    def test_is_retryable_gateway_error_accepts_read_error(self):
        """Predicate reports True for ReadError so idempotent GETs retry."""
        from prime_sandboxes.sandbox import _is_retryable_gateway_error

        exc = httpx.ReadError("Connection broken", request=httpx.Request("GET", "http://test"))
        assert _is_retryable_gateway_error(exc) is True

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

    def test_post_no_retry_on_read_error(self):
        """ReadError is NOT retried for POST — server may have processed the request,
        and a retry would duplicate side effects (same hazard as ReadTimeout / 5xx)."""
        from prime_sandboxes.sandbox import _gateway_post_retry

        call_count = 0

        @_gateway_post_retry
        def read_error_on_post():
            nonlocal call_count
            call_count += 1
            raise httpx.ReadError(
                "Connection broken mid-response",
                request=httpx.Request("POST", "http://test"),
            )

        with pytest.raises(httpx.ReadError):
            read_error_on_post()

        assert call_count == 1  # no retry


class DummySandboxAuthCache:
    """Minimal auth cache for read_file unit tests."""

    def get_or_refresh(self, sandbox_id: str):
        return {
            "gateway_url": "https://gateway.example",
            "user_ns": "test-ns",
            "job_id": sandbox_id,
            "token": "test-token",
        }


class TestReadFileRetry:
    """Tests for read_file-specific retry behavior."""

    @pytest.mark.parametrize(
        "exc_cls",
        [httpx.ReadTimeout, httpx.ConnectTimeout, httpx.PoolTimeout],
    )
    def test_read_file_retry_on_timeout_types(self, exc_cls):
        """read_file retries transient timeout classes before succeeding."""
        from prime_sandboxes.sandbox import _read_file_retry

        call_count = 0
        request = httpx.Request("GET", "https://gateway.example/read-file")

        @_read_file_retry
        def timeout_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise exc_cls("temporary timeout", request=request)
            return "success"

        result = timeout_then_succeed()

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_read_file_retry_works_for_async_functions(self):
        """The shared decorator retries async read_file helpers too."""
        from prime_sandboxes.sandbox import _read_file_retry

        call_count = 0
        request = httpx.Request("GET", "https://gateway.example/read-file")

        @_read_file_retry
        async def timeout_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.ReadTimeout("temporary timeout", request=request)
            return "success"

        result = await timeout_then_succeed()

        assert result == "success"
        assert call_count == 2

    @pytest.mark.parametrize(
        "exc_cls, expected_name",
        [
            (httpx.ReadTimeout, "ReadTimeout"),
            (httpx.ConnectTimeout, "ConnectTimeout"),
            (httpx.PoolTimeout, "PoolTimeout"),
        ],
    )
    def test_read_file_timeout_error_names_timeout_type(self, monkeypatch, exc_cls, expected_name):
        """Final read_file timeout errors include the concrete timeout class."""
        from prime_sandboxes.sandbox import SandboxClient

        request = httpx.Request("GET", "https://gateway.example/read-file")

        def raise_timeout(url, headers, params, timeout):
            raise exc_cls("temporary timeout", request=request)

        monkeypatch.setattr(
            SandboxClient,
            "_gateway_read_file_get",
            staticmethod(raise_timeout),
        )

        client = SandboxClient.__new__(SandboxClient)
        client._auth_cache = DummySandboxAuthCache()

        with pytest.raises(APIError) as exc_info:
            client.read_file("sandbox-123", "/tmp/job.exit", timeout=12)

        message = str(exc_info.value)
        assert "Read file timed out after 12s" in message
        assert f"({expected_name})" in message
        assert "/tmp/job.exit" in message

    def test_read_file_connection_error_surfaces_nested_os_errno(self, monkeypatch):
        """Transport wrappers retain the actionable nested socket errno."""
        import errno

        from prime_sandboxes.sandbox import SandboxClient

        request = httpx.Request("GET", "https://gateway.example/read-file")

        def raise_connect_error(url, headers, params, timeout):
            try:
                raise OSError(errno.EMFILE, "Too many open files")
            except OSError as os_error:
                raise httpx.ConnectError(
                    "All connection attempts failed",
                    request=request,
                ) from os_error

        monkeypatch.setattr(
            SandboxClient,
            "_gateway_read_file_get",
            staticmethod(raise_connect_error),
        )

        client = SandboxClient.__new__(SandboxClient)
        client._auth_cache = DummySandboxAuthCache()

        with pytest.raises(APIError) as exc_info:
            client.read_file("sandbox-123", "/tmp/job.stdout.log")

        message = str(exc_info.value)
        assert "ConnectError" in message
        assert "errno=24" in message
        assert "Too many open files" in message


class AlwaysStatusTransport(httpx.BaseTransport):
    """Transport that always returns one HTTP status."""

    def __init__(self, status_code: int):
        self.status_code = status_code
        self.call_count = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.call_count += 1
        return httpx.Response(self.status_code, request=request, text="rate limited")


class AsyncBatchRateLimitTransport(httpx.AsyncBaseTransport):
    """Rate-limit one batch request, then return all requested statuses."""

    def __init__(self):
        self.call_count = 0
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.call_count += 1
        self.requests.append(request)
        if self.call_count == 1:
            return httpx.Response(429, request=request, text="rate limited")
        sandbox_ids = json.loads(request.content)["sandbox_ids"]
        return httpx.Response(
            200,
            request=request,
            json={
                "statuses": [
                    {
                        "sandbox_id": sandbox_id,
                        "status": "RUNNING",
                        "error_type": None,
                        "error_message": None,
                        "pending_image_build_id": None,
                    }
                    for sandbox_id in sandbox_ids
                ],
                "errors": [],
            },
        )


class AsyncGatewayStatusSequenceTransport(httpx.AsyncBaseTransport):
    """Return a fixed status sequence for gateway upload tests."""

    def __init__(self, statuses: list[int]):
        self.statuses = statuses
        self.call_count = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        status = self.statuses[self.call_count]
        self.call_count += 1
        if status != 200:
            return httpx.Response(status, request=request, text=f"HTTP {status}")
        return httpx.Response(
            200,
            request=request,
            json={
                "success": True,
                "path": "/tmp/file",
                "size": 4,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )


class AsyncGatewayAuthCache:
    async def get_or_refresh(self, _sandbox_id: str) -> dict[str, str]:
        return {
            "gateway_url": "https://gateway.example",
            "user_ns": "test-ns",
            "job_id": "job-123",
            "token": "test-token",
        }


@pytest.fixture
def no_sync_retry_sleep(monkeypatch):
    monkeypatch.setattr(
        APIClient._idempotent_post_request_with_retry.retry,
        "sleep",
        lambda _delay: None,
    )
    monkeypatch.setattr(
        APIClient._non_idempotent_request_with_retry.retry,
        "sleep",
        lambda _delay: None,
    )


@pytest.fixture
def no_async_retry_sleep(monkeypatch):
    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(AsyncAPIClient._idempotent_post_request_with_retry.retry, "sleep", no_sleep)
    monkeypatch.setattr(AsyncSandboxClient._gateway_post.retry, "sleep", no_sleep)


class TestRateLimitRetry:
    def test_create_retries_429_once(self, no_sync_retry_sleep):
        transport = StatusThenSucceedTransport(429, payload=_sandbox_response())
        api_client = APIClient(api_key="test-key")
        api_client.client = httpx.Client(transport=transport)

        sandbox = SandboxClient(api_client).create(
            CreateSandboxRequest(name="sandbox", docker_image="python:3.11-slim")
        )

        assert sandbox.id == "sandbox-1"
        assert transport.call_count == 2
        assert [request.method for request in transport.requests] == ["POST", "POST"]

    @pytest.mark.asyncio
    async def test_batch_status_429_retry_preserves_all_waiters(self, no_async_retry_sleep):
        transport = AsyncBatchRateLimitTransport()
        client = AsyncSandboxClient(api_key="test-key")
        await client.client.client.aclose()
        client.client.client = httpx.AsyncClient(transport=transport)

        async def reachable(_sandbox_id: str) -> bool:
            return True

        client._is_sandbox_reachable = reachable  # type: ignore[method-assign]
        try:
            await asyncio.gather(
                client.wait_for_creation("sandbox-a"),
                client.wait_for_creation("sandbox-b"),
            )
        finally:
            await client.aclose()

        assert transport.call_count == 2
        assert json.loads(transport.requests[1].content)["sandbox_ids"] == [
            "sandbox-a",
            "sandbox-b",
        ]

    def test_429_budget_exhaustion_raises_api_error(self, no_sync_retry_sleep):
        transport = AlwaysStatusTransport(429)
        client = APIClient(api_key="test-key")
        client.client = httpx.Client(transport=transport)

        with pytest.raises(APIError, match="HTTP 429"):
            client.request("POST", "sandbox", json={"name": "sandbox"})

        assert transport.call_count == 3

    @pytest.mark.asyncio
    async def test_429_during_gateway_retry_does_not_enter_409_status_path(
        self, no_async_retry_sleep
    ):
        transport = AsyncGatewayStatusSequenceTransport([409, 429, 200])
        client = AsyncSandboxClient(api_key="test-key")
        client._auth_cache = AsyncGatewayAuthCache()  # type: ignore[assignment]
        client._gateway_client = httpx.AsyncClient(transport=transport)
        status_checks = 0

        async def retry_409(
            _sandbox_id: str,
            _error: httpx.HTTPStatusError,
            _attempt: int,
            command: str | None = None,
        ) -> bool:
            nonlocal status_checks
            status_checks += 1
            return True

        client._should_retry_409 = retry_409  # type: ignore[method-assign]
        try:
            response = await client.upload_bytes(
                "sandbox-1",
                "/tmp/file",
                b"data",
                "file",
            )
        finally:
            await client.aclose()

        assert response.success
        assert transport.call_count == 3
        assert status_checks == 1


def _terminated_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        410,
        request=request,
        json={
            "error": "sandbox_terminated",
            "sandboxId": "sandbox-123",
            "message": "The sandbox has been terminated",
            "code": "not_found",
        },
    )


class TestGatewaySandboxTerminated:
    """Gateway HTTP 410 sandbox_terminated is terminal: typed error, no retry."""

    def test_read_file_raises_not_running_without_retry(self, monkeypatch):
        from prime_sandboxes.exceptions import SandboxNotRunningError

        calls = 0

        def terminated(url, headers, params, timeout):
            nonlocal calls
            calls += 1
            return _terminated_response(httpx.Request("GET", url))

        monkeypatch.setattr(SandboxClient, "_gateway_read_file_get", staticmethod(terminated))
        client = SandboxClient.__new__(SandboxClient)
        client._auth_cache = DummySandboxAuthCache()
        client._get_sandbox_error_context = lambda _sandbox_id: {  # type: ignore[method-assign]
            "status": None,
            "error_type": None,
            "error_message": None,
        }

        with pytest.raises(SandboxNotRunningError) as exc_info:
            client.read_file("sandbox-123", "/tmp/file")

        assert exc_info.value.status == "TERMINATED"
        assert isinstance(exc_info.value.__cause__, httpx.HTTPStatusError)
        assert calls == 1

    @pytest.mark.asyncio
    async def test_async_upload_raises_not_running_without_retry(self):
        from prime_sandboxes.exceptions import SandboxNotRunningError

        calls = 0

        async def terminated(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return _terminated_response(request)

        client = AsyncSandboxClient(api_key="test-key")
        client._auth_cache = AsyncGatewayAuthCache()  # type: ignore[assignment]
        client._gateway_client = httpx.AsyncClient(transport=httpx.MockTransport(terminated))

        async def error_context(_sandbox_id: str) -> dict:
            return {"status": None, "error_type": None, "error_message": None}

        client._get_sandbox_error_context = error_context  # type: ignore[method-assign]
        try:
            with pytest.raises(SandboxNotRunningError) as exc_info:
                await client.upload_bytes("sandbox-123", "/tmp/file", b"data", "file")
        finally:
            await client.aclose()

        assert exc_info.value.status == "TERMINATED"
        assert calls == 1

    def test_reachability_does_not_retry_terminated(self):
        from prime_sandboxes.exceptions import SandboxNotRunningError
        from prime_sandboxes.sandbox import _is_retryable_reachability_error

        request = httpx.Request("POST", "https://gateway.example/ns/job/exec")
        cause = httpx.HTTPStatusError(
            "gone", request=request, response=_terminated_response(request)
        )
        error = SandboxNotRunningError("sandbox-123", status="TERMINATED")
        error.__cause__ = cause

        assert _is_retryable_reachability_error(cause) is False
        assert _is_retryable_reachability_error(error) is False
