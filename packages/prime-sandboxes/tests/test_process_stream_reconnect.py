"""A live-process output stream re-attaches after a transient mid-stream drop (mode #3)."""

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError

from prime_sandboxes._proto.command_session import command_session_pb2 as pb
from prime_sandboxes.core import APIError
from prime_sandboxes.process import AsyncSandboxProcess

_EV = pb.CommandSessionEvent


def _start(pid):
    return pb.StartResponse(event=_EV(start=_EV.StartEvent(pid=pid)))


def _stdout(data):
    return pb.StartResponse(event=_EV(data=_EV.DataEvent(stdout=data)))


def _end(code):
    return pb.StartResponse(event=_EV(end=_EV.EndEvent(exit_code=code)))


class _FakeStreamClient:
    async def close(self):
        pass


async def _noop_stdin(pid, data):
    pass


async def _noop_signal(pid, sig):
    pass


async def _drain(stream):
    out = b""
    async for chunk in stream:
        out += chunk
    return out


# Both production stream-break variants must trigger a reconnect: UNAVAILABLE "... timed out"
# and INTERNAL "Error reading content".
_STREAM_FAULTS = [
    ConnectError(Code.UNAVAILABLE, "error reading a body from connection: timed out"),
    ConnectError(Code.INTERNAL, "Error reading content"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", _STREAM_FAULTS, ids=["unavailable_timeout", "internal_reading_content"])
async def test_stream_reconnects_and_resumes_after_transient_drop(fault):
    async def faulty():
        yield _start(42)
        yield _stdout(b"before\n")
        raise fault

    async def resumed():
        yield _start(42)  # Connect re-announces the pid; already known, ignored
        yield _stdout(b"after\n")
        yield _end(0)

    reconnect_calls = []

    async def reconnect(pid):
        reconnect_calls.append(pid)
        return resumed()

    proc = await AsyncSandboxProcess._create(
        _FakeStreamClient(), faulty(), _noop_stdin, _noop_signal, reconnect=reconnect
    )
    stdout = await _drain(proc.stdout)
    rc = await proc.wait()

    assert reconnect_calls == [42]  # re-attached to the same pid
    assert rc == 0  # exit observed on the resumed stream
    assert stdout == b"before\nafter\n"  # output from both segments
    await proc.aclose()


@pytest.mark.asyncio
async def test_stream_without_reconnect_still_fails():
    # No reconnect callable -> a transient drop is fatal (baseline behaviour preserved).
    async def faulty():
        yield _start(7)
        raise ConnectError(Code.UNAVAILABLE, "error reading a body from connection: timed out")

    proc = await AsyncSandboxProcess._create(
        _FakeStreamClient(), faulty(), _noop_stdin, _noop_signal, reconnect=None
    )
    with pytest.raises(APIError, match="process stream RPC failed"):
        await proc.wait()
    await proc.aclose()


@pytest.mark.asyncio
async def test_permanent_fault_is_not_reconnected():
    async def faulty():
        yield _start(9)
        raise ConnectError(Code.NOT_FOUND, "session gone")

    calls = []

    async def reconnect(pid):
        calls.append(pid)
        raise AssertionError("should not reconnect on a permanent fault")

    proc = await AsyncSandboxProcess._create(
        _FakeStreamClient(), faulty(), _noop_stdin, _noop_signal, reconnect=reconnect
    )
    with pytest.raises(APIError, match="process stream RPC failed"):
        await proc.wait()
    assert calls == []
    await proc.aclose()
