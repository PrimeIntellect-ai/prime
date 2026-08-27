"""Live-process output streams re-attach to the running process after a drop."""

import asyncio

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


async def _noop_stdin(data):
    pass


async def _noop_signal(sig):
    pass


async def _drain(stream):
    out = b""
    async for chunk in stream:
        out += chunk
    return out


_STREAM_FAULTS = [
    ConnectError(Code.UNAVAILABLE, "error reading a body from connection: timed out"),
    ConnectError(Code.INTERNAL, "Error reading content"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", _STREAM_FAULTS, ids=["unavailable_timeout", "internal_reading_content"]
)
async def test_stream_reconnects_and_resumes_after_drop(fault, monkeypatch):
    monkeypatch.setattr("prime_sandboxes.process._STREAM_RECONNECT_BACKOFF_SECONDS", 0)

    async def faulty():
        yield _start(42)
        yield _stdout(b"before\n")
        raise fault

    async def resumed():
        yield _start(42)  # Connect re-announces the pid; already known, ignored
        yield _stdout(b"after\n")
        yield _end(0)

    reconnect_calls = []

    def reconnect(started):
        reconnect_calls.append(started)
        return resumed()

    proc = await AsyncSandboxProcess._create(
        _FakeStreamClient(), faulty(), _noop_stdin, _noop_signal, reconnect=reconnect
    )
    stdout = await _drain(proc.stdout)
    rc = await proc.wait()

    assert reconnect_calls == [True]
    assert rc == 0  # exit observed on the resumed stream
    assert stdout == b"before\nafter\n"  # output from both segments
    await proc.aclose()


@pytest.mark.asyncio
async def test_stream_reconnects_after_clean_eof(monkeypatch):
    monkeypatch.setattr("prime_sandboxes.process._STREAM_RECONNECT_BACKOFF_SECONDS", 0)

    async def ended_without_exit():
        yield _start(42)
        yield _stdout(b"before\n")

    async def resumed():
        yield _start(42)
        yield _stdout(b"after\n")
        yield _end(0)

    reconnect_calls = []

    def reconnect(started):
        reconnect_calls.append(started)
        return resumed()

    proc = await AsyncSandboxProcess._create(
        _FakeStreamClient(),
        ended_without_exit(),
        _noop_stdin,
        _noop_signal,
        reconnect=reconnect,
    )

    assert await _drain(proc.stdout) == b"before\nafter\n"
    assert await proc.wait() == 0
    assert reconnect_calls == [True]
    await proc.aclose()


@pytest.mark.asyncio
async def test_stream_recovers_before_pid_is_received(monkeypatch):
    monkeypatch.setattr("prime_sandboxes.process._STREAM_RECONNECT_BACKOFF_SECONDS", 0)

    async def dropped_before_start():
        raise ConnectError(Code.UNAVAILABLE, "stream dropped")
        yield _start(42)

    reconnect_calls = []

    def reconnect(started):
        # started=False tells the caller to retry Start (create-or-attach).
        reconnect_calls.append(started)

        async def stream():
            yield _start(42)
            yield _end(0)

        return stream()

    proc = await AsyncSandboxProcess._create(
        _FakeStreamClient(),
        dropped_before_start(),
        _noop_stdin,
        _noop_signal,
        reconnect=reconnect,
    )

    assert proc.pid == 42
    assert await proc.wait() == 0
    assert reconnect_calls == [False]
    await proc.aclose()


@pytest.mark.asyncio
async def test_reconnect_after_exit_replays_end_event(monkeypatch):
    monkeypatch.setattr("prime_sandboxes.process._STREAM_RECONNECT_BACKOFF_SECONDS", 0)

    async def dropped_mid_stream():
        yield _start(42)
        yield _stdout(b"before\n")
        raise ConnectError(Code.UNAVAILABLE, "stream dropped")

    async def replayed_after_exit():
        # sandboxd retains exited sessions briefly; Connect replays the exit.
        yield _start(42)
        yield _end(3)

    proc = await AsyncSandboxProcess._create(
        _FakeStreamClient(),
        dropped_mid_stream(),
        _noop_stdin,
        _noop_signal,
        reconnect=lambda started: replayed_after_exit(),
    )

    assert await _drain(proc.stdout) == b"before\n"
    assert await proc.wait() == 3
    await proc.aclose()


@pytest.mark.asyncio
async def test_end_before_pid_fails_instead_of_hanging():
    async def ended_before_start():
        yield _end(0)

    with pytest.raises(APIError, match="ended before reporting its PID"):
        await asyncio.wait_for(
            AsyncSandboxProcess._create(
                _FakeStreamClient(),
                ended_before_start(),
                _noop_stdin,
                _noop_signal,
            ),
            timeout=1,
        )


@pytest.mark.asyncio
async def test_stream_without_reconnect_still_fails():
    # No reconnect callable preserves the previous fatal behavior.
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
@pytest.mark.parametrize(
    "fault",
    [
        ConnectError(Code.NOT_FOUND, "session gone"),
        ConnectError(Code.FAILED_PRECONDITION, "session spec conflict"),
    ],
    ids=["not_found", "failed_precondition"],
)
async def test_permanent_fault_is_not_reconnected(fault):
    async def faulty():
        yield _start(9)
        raise fault

    calls = []

    def reconnect(started):
        calls.append(started)
        raise AssertionError("should not reconnect on a permanent fault")

    proc = await AsyncSandboxProcess._create(
        _FakeStreamClient(), faulty(), _noop_stdin, _noop_signal, reconnect=reconnect
    )
    with pytest.raises(APIError, match="process stream RPC failed"):
        await proc.wait()
    assert calls == []
    await proc.aclose()
