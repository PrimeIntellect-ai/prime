"""A request that names no timeout keeps the client's own, rather than waiting forever."""

import asyncio

import httpx
import pytest

from prime_sandboxes.core.client import APIClient, AsyncAPIClient

DEFAULT = httpx.Timeout(30.0, connect=10.0)


class RecordingTransport(httpx.BaseTransport):
    def __init__(self):
        self.timeouts = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.timeouts.append(request.extensions["timeout"])
        return httpx.Response(200, json={})


class AsyncRecordingTransport(httpx.AsyncBaseTransport):
    def __init__(self):
        self.timeouts = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.timeouts.append(request.extensions["timeout"])
        return httpx.Response(200, json={})


@pytest.mark.parametrize("method", ["GET", "POST", "DELETE"])
def test_sync_request_keeps_the_client_timeout_unless_given_one(method):
    transport = RecordingTransport()
    client = APIClient(api_key="test-key")
    client.client = httpx.Client(transport=transport, timeout=DEFAULT)

    client.request(method, "test")
    client.request(method, "test", timeout=5.0)

    assert transport.timeouts == [DEFAULT.as_dict(), httpx.Timeout(5.0).as_dict()]


@pytest.mark.parametrize("method", ["GET", "POST", "DELETE"])
def test_async_request_keeps_the_client_timeout_unless_given_one(method):
    transport = AsyncRecordingTransport()
    client = AsyncAPIClient(api_key="test-key")
    client.client = httpx.AsyncClient(transport=transport, timeout=DEFAULT)

    async def run():
        await client.request(method, "test")
        await client.request(method, "test", timeout=5.0)

    asyncio.run(run())

    assert transport.timeouts == [DEFAULT.as_dict(), httpx.Timeout(5.0).as_dict()]


def test_async_request_to_a_server_that_never_answers_times_out():
    """The API accepts the connection and never replies: the request ends, it does not hang."""

    async def run():
        async def silent(reader, writer):
            await asyncio.sleep(3600)

        server = await asyncio.start_server(silent, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        client = AsyncAPIClient(api_key="test-key")
        client.base_url = f"http://127.0.0.1:{port}"
        client.client = httpx.AsyncClient(timeout=httpx.Timeout(0.2))
        try:
            with pytest.raises(Exception) as raised:
                await asyncio.wait_for(client.request("GET", "test"), timeout=10)
            assert not isinstance(raised.value, asyncio.TimeoutError), "the request hung"
        finally:
            server.close()

    asyncio.run(run())
