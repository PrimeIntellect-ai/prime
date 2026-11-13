"""Live-process output streams re-attach to the running process after a drop."""

import asyncio

import pytest
from conftest import _end_event, _start_event, _stdout_event
from connectrpc.code import Code
from connectrpc.errors import ConnectError

from prime_sandboxes.core import APIError
from prime_sandboxes.process import AsyncSandboxProcess


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
        yield _start_event(42, output_replay=True)
        yield _stdout_event(b"before\n", sequence=1)
        raise fault

    async def resumed():
        yield _start_event(
            42, output_replay=True
        )  # Connect re-announces the pid; already known, ignored
        yield _stdout_event(b"after\n", sequence=2)
        yield _end_event(0, output_sequence=2)

    reconnect_calls = []

    def reconnect(started, after_sequence):
        reconnect_calls.append((started, after_sequence))
        return resumed()

    proc = await AsyncSandboxProcess._create(
        _FakeStreamClient(), faulty(), _noop_stdin, _noop_signal, reconnect=reconnect
    )
    stdout = await _drain(proc.stdout)
    rc = await proc.wait()

    assert reconnect_calls == [(True, 1)]
    assert rc == 0  # exit observed on the resumed stream
    assert stdout == b"before\nafter\n"  # output from both segments
    await proc.aclose()


@pytest.mark.asyncio
async def test_stream_reconnects_after_clean_eof(monkeypatch):
    monkeypatch.setattr("prime_sandboxes.process._STREAM_RECONNECT_BACKOFF_SECONDS", 0)

    async def ended_without_exit():
        yield _start_event(42, output_replay=True)
        yield _stdout_event(b"before\n", sequence=1)

    async def resumed():
        yield _start_event(42, output_replay=True)
        yield _stdout_event(b"after\n", sequence=2)
        yield _end_event(0, output_sequence=2)

    reconnect_calls = []

    def reconnect(started, after_sequence):
        reconnect_calls.append((started, after_sequence))
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
    assert reconnect_calls == [(True, 1)]
    await proc.aclose()


@pytest.mark.asyncio
async def test_stream_recovers_before_pid_is_received(monkeypatch):
    monkeypatch.setattr("prime_sandboxes.process._STREAM_RECONNECT_BACKOFF_SECONDS", 0)

    async def dropped_before_start():
        raise ConnectError(Code.UNAVAILABLE, "stream dropped")
        yield _start_event(42, output_replay=True)

    reconnect_calls = []

    def reconnect(started, after_sequence):
        # started=False tells the caller to retry Start (create-or-attach).
        reconnect_calls.append((started, after_sequence))

        async def stream():
            yield _start_event(42, output_replay=True)
            yield _end_event(0)

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
    assert reconnect_calls == [(False, 0)]
    await proc.aclose()


@pytest.mark.asyncio
async def test_reconnect_after_exit_replays_end_event(monkeypatch):
    monkeypatch.setattr("prime_sandboxes.process._STREAM_RECONNECT_BACKOFF_SECONDS", 0)

    async def dropped_mid_stream():
        yield _start_event(42, output_replay=True)
        yield _stdout_event(b"before\n", sequence=1)
        raise ConnectError(Code.UNAVAILABLE, "stream dropped")

    async def replayed_after_exit():
        # sandboxd retains exited sessions briefly; Connect replays the exit.
        yield _start_event(42, output_replay=True)
        yield _end_event(3, output_sequence=1)

    proc = await AsyncSandboxProcess._create(
        _FakeStreamClient(),
        dropped_mid_stream(),
        _noop_stdin,
        _noop_signal,
        reconnect=lambda started, after_sequence: replayed_after_exit(),
    )

    assert await _drain(proc.stdout) == b"before\n"
    assert await proc.wait() == 3
    await proc.aclose()


@pytest.mark.asyncio
async def test_end_before_pid_fails_instead_of_hanging():
    async def ended_before_start():
        yield _end_event(0)

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
        yield _start_event(7)
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
        yield _start_event(9)
        raise fault

    calls = []

    def reconnect(started, after_sequence):
        calls.append(started)
        raise AssertionError("should not reconnect on a permanent fault")

    proc = await AsyncSandboxProcess._create(
        _FakeStreamClient(), faulty(), _noop_stdin, _noop_signal, reconnect=reconnect
    )
    with pytest.raises(APIError, match="process stream RPC failed"):
        await proc.wait()
    assert calls == []
    await proc.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("spacing, survives", [(0, False), (200, True)], ids=["burst", "spread"])
async def test_reconnect_budget_resets_after_a_healthy_stretch(spacing, survives, monkeypatch):
    # A day-long stream may drop many times; only drops within one outage share a budget.
    monkeypatch.setattr("prime_sandboxes.process._STREAM_RECONNECT_BACKOFF_SECONDS", 0)
    clock = [1000.0]
    monkeypatch.setattr("prime_sandboxes.process.time.monotonic", lambda: clock[0])
    drops = 12  # more than _STREAM_MAX_RECONNECTS

    async def segment(n):
        clock[0] += spacing
        yield _start_event(42, output_replay=True)
        if n == drops:
            yield _end_event(0)
        raise ConnectError(Code.UNAVAILABLE, "Error reading content")

    calls = [0]

    def reconnect(started, after_sequence):
        calls[0] += 1
        return segment(calls[0])

    proc = await AsyncSandboxProcess._create(
        _FakeStreamClient(), segment(0), _noop_stdin, _noop_signal, reconnect=reconnect
    )
    if survives:
        assert await proc.wait() == 0
        assert calls[0] == drops
    else:
        with pytest.raises(APIError, match="Error reading content"):
            await proc.wait()
        assert calls[0] == 8
    await proc.aclose()


@pytest.mark.asyncio
async def test_replay_deduplicates_and_recovers_both_streams(monkeypatch):
    from prime_sandboxes._proto.command_session import command_session_pb2 as pb

    monkeypatch.setattr("prime_sandboxes.process._STREAM_RECONNECT_BACKOFF_SECONDS", 0)

    async def first():
        yield _start_event(42, output_replay=True)
        yield _stdout_event(b"before", sequence=1)
        raise ConnectError(Code.UNAVAILABLE, "disconnect")

    calls = []

    async def replay():
        yield _start_event(42, output_replay=True)
        yield _stdout_event(b"before", sequence=1)  # Already queued, never duplicate.
        yield pb.ConnectResponse(
            event=pb.CommandSessionEvent(
                data=pb.CommandSessionEvent.DataEvent(stderr=b"during", sequence=2)
            )
        )
        yield _stdout_event(b"after", sequence=3)
        yield _end_event(7, output_sequence=3)

    def reconnect(started, cursor):
        calls.append((started, cursor))
        return replay()

    proc = await AsyncSandboxProcess._create(
        _FakeStreamClient(), first(), _noop_stdin, _noop_signal, reconnect=reconnect
    )
    assert await _drain(proc.stdout) == b"beforeafter"
    assert await _drain(proc.stderr) == b"during"
    assert await proc.wait() == 7
    assert calls == [(True, 1)]
    await proc.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    ["jump", "terminal_gap", "eviction", "truncated", "old_server", "downgrade", "pid_change"],
)
async def test_incomplete_replay_fails_closed(failure, monkeypatch):
    from prime_sandboxes import ProcessOutputLostError

    monkeypatch.setattr("prime_sandboxes.process._STREAM_RECONNECT_BACKOFF_SECONDS", 0)

    async def initial():
        yield _start_event(42, output_replay=failure != "old_server")
        yield _stdout_event(b"before", sequence=1)
        if failure == "jump":
            yield _stdout_event(b"missing-middle", sequence=3)
        elif failure == "terminal_gap":
            yield _end_event(0, output_sequence=2)
        elif failure in ("eviction", "truncated"):
            raise ConnectError(
                Code.OUT_OF_RANGE if failure == "eviction" else Code.DATA_LOSS, "output unavailable"
            )
        else:
            raise ConnectError(Code.UNAVAILABLE, "disconnected")

    calls = []

    async def replay():
        yield _start_event(
            43 if failure == "pid_change" else 42, output_replay=failure != "downgrade"
        )
        yield _end_event(0, output_sequence=1)

    def reconnect(started, cursor):
        calls.append(cursor)
        return replay()

    proc = await AsyncSandboxProcess._create(
        _FakeStreamClient(), initial(), _noop_stdin, _noop_signal, reconnect=reconnect
    )
    with pytest.raises(ProcessOutputLostError):
        await proc.wait()
    with pytest.raises(ProcessOutputLostError):
        await _drain(proc.stdout)
    assert proc.returncode is None
    assert calls == ([1] if failure in ("downgrade", "pid_change") else [])
    await proc.aclose()


@pytest.mark.asyncio
async def test_old_server_initial_response_loss_still_cleans_up(monkeypatch):
    from prime_sandboxes import ProcessOutputLostError

    monkeypatch.setattr("prime_sandboxes.process._STREAM_RECONNECT_BACKOFF_SECONDS", 0)

    async def missing_start():
        raise ConnectError(Code.UNAVAILABLE, "response lost")
        yield

    async def old_server():
        yield _start_event(42)

    signals = []

    async def signal(kind):
        signals.append(kind)

    with pytest.raises(ProcessOutputLostError):
        await AsyncSandboxProcess._create(
            _FakeStreamClient(),
            missing_start(),
            _noop_stdin,
            signal,
            reconnect=lambda started, cursor: old_server(),
        )
    assert signals == ["terminate", "kill"]


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [Code.OUT_OF_RANGE, Code.DATA_LOSS])
async def test_replay_failure_before_pid_stops_session_by_uuid(code):
    """A lost initial response plus unrecoverable output must not leak a process."""
    from prime_sandboxes import ProcessOutputLostError

    async def failed_replay():
        raise ConnectError(code, "output no longer available")
        yield

    signals = []

    async def signal(kind):
        signals.append(kind)

    with pytest.raises(ProcessOutputLostError):
        await asyncio.wait_for(
            AsyncSandboxProcess._create(_FakeStreamClient(), failed_replay(), _noop_stdin, signal),
            timeout=1,
        )
    assert signals == ["terminate", "kill"]
