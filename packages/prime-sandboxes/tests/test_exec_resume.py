"""execute_command re-attaches after link faults without re-running or losing output."""

from typing import Any, cast

import pytest
from conftest import _AsyncFakeCache, _auth_payload, _end_event, _stdout_event
from connectrpc.code import Code
from connectrpc.errors import ConnectError

from prime_sandboxes._proto.command_session import command_session_pb2 as pb
from prime_sandboxes.core.client import APIClient, APIError
from prime_sandboxes.sandbox import AsyncSandboxClient, SandboxClient

_EV = pb.CommandSessionEvent
_DROP = ConnectError(Code.UNAVAILABLE, "Failed to route request to sandbox")


def _start(echo=None):
    start = _EV.StartEvent(pid=7)
    if echo is not None:
        start.data_offset = echo
    return pb.StartResponse(event=_EV(start=start))


class _FakeCache:
    def get_or_refresh(self, _sandbox_id):
        return _auth_payload()


def _scripted(calls, script):
    """A ConnectClient double whose Nth stream replays script[N] (an event or an exception)."""

    class _Client:
        def __init__(self, *_a, **_k):
            pass

        def _stream(self, method, request):
            calls.append((method.name, request))
            for item in script[len(calls) - 1]:
                if isinstance(item, BaseException):
                    raise item
                yield item

        def execute_server_stream(self, *, request, method, **_k):
            return self._stream(method, request)

        def close(self):
            pass

    class _AsyncClient(_Client):
        async def execute_server_stream(self, *, request, method, **_k):
            for item in self._stream(method, request):
                yield item

        async def close(self):
            pass

    return _Client, _AsyncClient


def _run_sync(monkeypatch, script):
    calls = []
    sync_client, _ = _scripted(calls, script)
    monkeypatch.setattr("prime_sandboxes.sandbox.ConnectClientSync", sync_client)
    monkeypatch.setattr("prime_sandboxes.sandbox.time.sleep", lambda _: None)
    client = SandboxClient(APIClient(api_key="test-key"))
    cast(Any, client)._auth_cache = _FakeCache()
    return client, calls


def test_drop_after_output_resumes_with_connect_at_offset(monkeypatch):
    client, calls = _run_sync(
        monkeypatch,
        [
            [_start(0), _stdout_event(b"a"), _DROP],
            [_start(1), _stdout_event(b"b"), _end_event(0)],
        ],
    )
    result = client._execute_command_connect_rpc("sb", "echo ab")
    assert result.stdout == "ab"
    (start_name, start_req), (connect_name, connect_req) = calls
    assert (start_name, start_req.data_offset) == ("Start", 0)
    assert (connect_name, connect_req.data_offset) == ("Connect", 1)
    assert connect_req.session.session_uuid == start_req.session_uuid != ""


def test_drop_before_start_event_retries_the_same_start(monkeypatch):
    client, calls = _run_sync(
        monkeypatch,
        [[_DROP], [_start(0), _stdout_event(b"x"), _end_event(3)]],
    )
    result = client._execute_command_connect_rpc("sb", "x")
    assert (result.stdout, result.exit_code) == ("x", 3)
    assert [name for name, _ in calls] == ["Start", "Start"]
    assert calls[0][1] is calls[1][1]  # byte-identical create-or-attach


def test_server_without_replay_keeps_faults_final(monkeypatch):
    client, calls = _run_sync(monkeypatch, [[_start(), _stdout_event(b"a"), _DROP]])
    with pytest.raises(APIError, match="unavailable"):
        client._execute_command_connect_rpc("sb", "x")
    assert len(calls) == 1


def test_retried_start_without_echo_raises_the_original_fault(monkeypatch):
    # The retry may have attached mid-run on an old sandboxd, so output could be missing.
    client, calls = _run_sync(monkeypatch, [[_DROP], [_start(), _end_event(0)]])
    with pytest.raises(APIError, match="Failed to route request to sandbox"):
        client._execute_command_connect_rpc("sb", "x")
    assert len(calls) == 2


def test_evicted_output_is_final(monkeypatch):
    client, calls = _run_sync(
        monkeypatch,
        [
            [_start(0), _stdout_event(b"a"), _DROP],
            [ConnectError(Code.OUT_OF_RANGE, "output before data event 1 was evicted")],
        ],
    )
    with pytest.raises(APIError, match="out_of_range"):
        client._execute_command_connect_rpc("sb", "x")
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_async_drop_after_output_resumes_with_connect_at_offset(monkeypatch):
    calls = []
    _, async_client = _scripted(
        calls,
        [
            [_start(0), _stdout_event(b"a"), _DROP],
            [_start(1), _stdout_event(b"b"), _end_event(0)],
        ],
    )
    monkeypatch.setattr("prime_sandboxes.sandbox.ConnectClient", async_client)

    async def _no_sleep(_):
        return None

    monkeypatch.setattr("prime_sandboxes.sandbox.asyncio.sleep", _no_sleep)
    client = AsyncSandboxClient(APIClient(api_key="test-key"))
    cast(Any, client)._auth_cache = _AsyncFakeCache()
    try:
        result = await client._execute_command_connect_rpc("sb", "echo ab")
    finally:
        await client.aclose()
    assert result.stdout == "ab"
    assert [(name, req.data_offset) for name, req in calls] == [("Start", 0), ("Connect", 1)]
