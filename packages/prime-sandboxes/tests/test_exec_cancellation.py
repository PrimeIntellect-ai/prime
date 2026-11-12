"""Cancelling a task inside an exec must raise CancelledError, not APIError.

connectrpc wraps a local cancellation as ConnectError(CANCELED); callers that
retry on errors would otherwise swallow the cancellation and never stop.
"""

import asyncio
from typing import Any, cast

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError

from prime_sandboxes._connectrpc import raise_if_cancelled
from prime_sandboxes.sandbox import AsyncSandboxClient


@pytest.mark.asyncio
async def test_cancelled_execute_command_raises_cancelled_error():
    async def never_respond(_reader, writer):
        await asyncio.sleep(60)
        writer.close()

    server = await asyncio.start_server(never_respond, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]

    class _HangingGatewayCache:
        async def get_or_refresh(self, _sandbox_id: str):
            return {
                "gateway_url": f"http://127.0.0.1:{port}",
                "user_ns": "ns",
                "job_id": "job",
                "token": "tok",
            }

    client = AsyncSandboxClient(api_key="test-key")
    cast(Any, client)._auth_cache = _HangingGatewayCache()
    task = asyncio.create_task(client.execute_command("sbx-vm", "sleep 3"))
    await asyncio.sleep(0.1)
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()
    finally:
        server.close()
        await client.aclose()


def test_server_sent_canceled_stays_an_error():
    raise_if_cancelled(ConnectError(Code.CANCELED, "cancelled by server"))
