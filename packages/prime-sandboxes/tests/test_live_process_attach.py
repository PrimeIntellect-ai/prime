"""A live process outlives its handle: another client attaches by session uuid."""

import asyncio
from typing import Any, cast

import pytest
from conftest import _AsyncFakeCache, _end_event, _start_event, _stdout_event
from connectrpc.code import Code
from connectrpc.errors import ConnectError

from prime_sandboxes._proto.command_session import command_session_pb2 as pb
from prime_sandboxes.core import APIError
from prime_sandboxes.sandbox import AsyncSandboxClient


class _FakeConnectClient:
    """Start streams a pid and output then idles; Connect replays the session's end."""

    calls: list[tuple[str, Any]] = []

    def __init__(self, _address: str, **_kwargs):
        pass

    def execute_server_stream(self, **kwargs):
        method = kwargs["method"].name
        request = kwargs["request"]
        type(self).calls.append((method, request))

        async def events():
            if method == "Start":
                yield _start_event(42)
                yield _stdout_event(b"started\n")
                await asyncio.Event().wait()
            yield _start_event(42, pb.ConnectResponse)
            yield _stdout_event(b"attached\n", pb.ConnectResponse)
            yield _end_event(3, pb.ConnectResponse)

        return events()

    async def execute_unary(self, **kwargs):
        type(self).calls.append((kwargs["method"].name, kwargs["request"]))
        return kwargs["method"].output()

    async def close(self):
        pass


def _client(monkeypatch) -> AsyncSandboxClient:
    _FakeConnectClient.calls = []
    monkeypatch.setattr("prime_sandboxes.sandbox.ConnectClient", _FakeConnectClient)
    client = AsyncSandboxClient(api_key="test-key")
    cast(Any, client)._auth_cache = _AsyncFakeCache()
    return client


async def _read(process) -> bytes:
    data = b""
    async for chunk in process.stdout:
        data += chunk
    return data


@pytest.mark.asyncio
async def test_open_process_exposes_its_session_uuid(monkeypatch):
    client = _client(monkeypatch)
    try:
        process = await client.open_process("sbx-vm", "cat")
        (method, request), *_ = _FakeConnectClient.calls
        assert method == "Start"
        assert process.session_uuid == request.session_uuid
        await process.detach()
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_attach_process_connects_to_the_session(monkeypatch):
    client = _client(monkeypatch)
    try:
        process = await client.attach_process("sbx-vm", "session-1")
        assert process.session_uuid == "session-1"
        assert process.pid == 42
        assert await process.wait() == 3
        assert await _read(process) == b"attached\n"
        method, request = _FakeConnectClient.calls[0]
        assert method == "Connect"
        assert request.session.session_uuid == "session-1"
        assert all(method == "Connect" for method, _ in _FakeConnectClient.calls)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_attach_process_reconnects_with_connect(monkeypatch):
    """A drop before the attached stream re-announces the pid still re-attaches with
    Connect: there is no Start request to retry."""
    monkeypatch.setattr("prime_sandboxes.process._STREAM_RECONNECT_BACKOFF_SECONDS", 0)
    methods = []

    class _DroppingClient(_FakeConnectClient):
        def execute_server_stream(self, **kwargs):
            methods.append(kwargs["method"].name)
            if len(methods) == 1:

                async def dropped():
                    raise ConnectError(Code.UNAVAILABLE, "stream dropped")
                    yield

                return dropped()
            return super().execute_server_stream(**kwargs)

    monkeypatch.setattr("prime_sandboxes.sandbox.ConnectClient", _DroppingClient)
    client = AsyncSandboxClient(api_key="test-key")
    cast(Any, client)._auth_cache = _AsyncFakeCache()
    try:
        process = await client.attach_process("sbx-vm", "session-1")
        assert await process.wait() == 3
        assert methods == ["Connect", "Connect"]
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_detach_leaves_the_process_running(monkeypatch):
    client = _client(monkeypatch)
    try:
        process = await client.open_process("sbx-vm", "cat")
        await process.detach()
        assert [method for method, _ in _FakeConnectClient.calls] == ["Start"]
        with pytest.raises(APIError, match="detached"):
            await process.wait()
        await process.aclose()  # a detached handle is closed: no signal follows
        assert [method for method, _ in _FakeConnectClient.calls] == ["Start"]
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_attach_process_rejects_a_non_positive_timeout(monkeypatch):
    client = _client(monkeypatch)
    try:
        with pytest.raises(ValueError, match="positive"):
            await client.attach_process("sbx-vm", "session-1", timeout=0)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_detach_does_not_reattach_its_cancelled_stream(monkeypatch):
    """The stream reports its own cancellation as a dropped stream; detaching must not
    take that for a fault to re-attach after."""
    monkeypatch.setattr("prime_sandboxes.process._STREAM_RECONNECT_BACKOFF_SECONDS", 0)

    class _CancelAsDropClient(_FakeConnectClient):
        def execute_server_stream(self, **kwargs):
            type(self).calls.append((kwargs["method"].name, kwargs["request"]))

            async def events():
                yield _start_event(42)
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    raise ConnectError(Code.CANCELED, "Request was cancelled") from None

            return events()

    client = _client(monkeypatch)
    monkeypatch.setattr("prime_sandboxes.sandbox.ConnectClient", _CancelAsDropClient)
    try:
        process = await client.open_process("sbx-vm", "cat")
        await asyncio.wait_for(process.detach(), timeout=5)
        assert [method for method, _ in _FakeConnectClient.calls] == ["Start"]
    finally:
        await client.aclose()
