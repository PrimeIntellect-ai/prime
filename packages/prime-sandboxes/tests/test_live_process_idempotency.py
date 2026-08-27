"""Live-process RPCs retry safely via sandboxd's idempotency keys.

Start carries a session_uuid (create-or-attach; see also the Start-retry test in
test_command_transport_selection.py), stdin writes carry an input_uuid
(duplicate applies are acknowledged, not repeated), and signals address the
session by id, so every transient link fault is retried instead of surfacing.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, cast

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError

from prime_sandboxes._proto.command_session import command_session_pb2 as pb
from prime_sandboxes.core import APIError
from prime_sandboxes.sandbox import AsyncSandboxClient

_EV = pb.CommandSessionEvent


def _start_event(pid, response_type=pb.StartResponse):
    return response_type(event=_EV(start=_EV.StartEvent(pid=pid)))


def _stdout_event(data, response_type=pb.StartResponse):
    return response_type(event=_EV(data=_EV.DataEvent(stdout=data)))


def _end_event(code, response_type=pb.StartResponse):
    return response_type(event=_EV(end=_EV.EndEvent(exit_code=code)))


class _AsyncFakeCache:
    async def get_or_refresh(self, _sandbox_id: str):
        return {
            "gateway_url": "https://gateway.example.com",
            "user_ns": "ns",
            "job_id": "job",
            "token": "tok",
            "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat(),
        }

    async def is_vm(self, _sandbox_id: str) -> bool:
        return True


async def _open_process(monkeypatch, fake_client_factory, command="cat"):
    monkeypatch.setattr("prime_sandboxes.process._STREAM_RECONNECT_BACKOFF_SECONDS", 0)
    monkeypatch.setattr("prime_sandboxes.sandbox._PROCESS_CONTROL_RETRY_INITIAL_DELAY", 0)
    monkeypatch.setattr("prime_sandboxes.sandbox.ConnectClient", fake_client_factory)
    client = AsyncSandboxClient(api_key="test-key")
    cast(Any, client)._auth_cache = _AsyncFakeCache()
    return client, await client.open_process("sbx-vm", command)


@pytest.mark.asyncio
async def test_reconnect_selects_session_uuid_and_replays_exit(monkeypatch):
    """A post-PID drop re-attaches by session selector; a missed exit is replayed."""
    connect_selectors = []
    session_uuids = []

    class _FakeConnectClient:
        def __init__(self, _address: str, **_kwargs):
            pass

        def execute_server_stream(self, **kwargs):
            method = kwargs["method"].name

            async def events():
                if method == "Start":
                    session_uuids.append(kwargs["request"].session_uuid)
                    yield _start_event(42)
                    yield _stdout_event(b"partial\n")
                    raise ConnectError(Code.UNAVAILABLE, "stream dropped")
                selector = kwargs["request"].session
                connect_selectors.append((selector.WhichOneof("selector"), selector.session_uuid))
                # The process exited while detached; sandboxd replays the retained end.
                yield _start_event(42, pb.ConnectResponse)
                yield _end_event(5, pb.ConnectResponse)

            return events()

        async def close(self):
            pass

    client, process = await _open_process(monkeypatch, _FakeConnectClient)
    try:
        assert await process.wait() == 5
        assert connect_selectors == [("session_uuid", session_uuids[0])]
        await process.aclose()
    finally:
        await client.aclose()


class _ControlFakeConnectClient:
    """Start succeeds; each queued fault is raised by one unary control RPC."""

    def __init__(self):
        self.input_requests = []
        self.signal_requests = []
        self.unary_faults = []
        self.exit_requested = asyncio.Event()

    def execute_server_stream(self, **kwargs):
        assert kwargs["method"].name == "Start"

        async def events():
            yield _start_event(42)
            await self.exit_requested.wait()
            yield _end_event(0)

        return events()

    async def execute_unary(self, **kwargs):
        request = kwargs["request"]
        if kwargs["method"].name == "SendInput":
            self.input_requests.append(request)
            if self.unary_faults:
                raise self.unary_faults.pop(0)  # response lost after the write applied
            return pb.SendInputResponse()
        assert kwargs["method"].name == "SendSignal"
        self.signal_requests.append(request)
        if self.unary_faults:
            raise self.unary_faults.pop(0)
        self.exit_requested.set()
        return pb.SendSignalResponse()

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_stdin_retry_reuses_input_uuid_and_signal_retry_succeeds(monkeypatch):
    fake = _ControlFakeConnectClient()
    # INTERNAL "Error reading content" exercises the message-marker arm of the
    # transient classifier; DEADLINE_EXCEEDED below exercises the code arm.
    fake.unary_faults = [ConnectError(Code.INTERNAL, "Error reading content")]

    client, process = await _open_process(monkeypatch, lambda *_args, **_kwargs: fake)
    try:
        await process.write_stdin(b"hello\n")

        # The write's fault was ambiguous; the retry reused the input_uuid, so the
        # server could acknowledge a duplicate apply without writing again.
        assert len(fake.input_requests) == 2
        assert fake.input_requests[0].input_uuid
        assert fake.input_requests[0].input_uuid == fake.input_requests[1].input_uuid

        await process.write_stdin(b"world\n")
        input_uuids = {request.input_uuid for request in fake.input_requests}
        assert len(input_uuids) == 2  # a distinct logical write gets a distinct id

        fake.unary_faults = [ConnectError(Code.DEADLINE_EXCEEDED, "deadline")]
        await process.terminate()
        assert len(fake.signal_requests) == 2  # transient fault, retried

        assert await process.wait() == 0
        await process.aclose()
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_failed_precondition_control_rpc_is_not_retried(monkeypatch):
    fake = _ControlFakeConnectClient()
    fake.unary_faults = [ConnectError(Code.FAILED_PRECONDITION, "session spec conflict")]

    client, process = await _open_process(monkeypatch, lambda *_args, **_kwargs: fake)
    try:
        with pytest.raises(APIError, match="failed_precondition"):
            await process.write_stdin(b"hello\n")
        assert len(fake.input_requests) == 1  # surfaced immediately, no retry

        await process.terminate()
        assert await process.wait() == 0
        await process.aclose()
    finally:
        await client.aclose()
