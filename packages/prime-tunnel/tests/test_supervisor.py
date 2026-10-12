import asyncio
import http.server
import threading
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from test_tunnel import _client_for, _fake_frpc_binary, _frpc_line, _make_started_tunnel

from prime_tunnel import Tunnel, TunnelState
from prime_tunnel.exceptions import TunnelConnectionError, TunnelGoneError
from prime_tunnel.frpc import FrpcEvent
from prime_tunnel.supervisor import Supervisor, _Response, _Verdict


@pytest.fixture(autouse=True)
def fast_supervisor(monkeypatch):
    """Shrink the supervisor's timings so tests take milliseconds."""
    for name, value in (
        ("probe_interval", 0.02),
        ("probe_timeout", 1.0),
        ("grace", 0.02),
        ("settle", 0.01),
        ("tick", 0.01),
        ("backoff_base", 0.01),
        ("backoff_max", 0.04),
    ):
        monkeypatch.setattr(Supervisor, name, value)


def _supervised_tunnel(states: list | None = None) -> Tunnel:
    """A started tunnel with a live fake frpc, ready for a Supervisor."""
    tunnel = _make_started_tunnel()
    tunnel._process.poll.return_value = None
    tunnel._loop = asyncio.get_running_loop()
    tunnel._reported_state = TunnelState.CONNECTED
    tunnel._ended.clear()
    tunnel._client = AsyncMock()
    tunnel.restart = AsyncMock(return_value=tunnel.url)
    if states is not None:
        tunnel.on_state_change = states.append
    return tunnel


def _supervise(tunnel: Tunnel, verdicts) -> Supervisor:
    """Start a Supervisor whose probe returns `verdicts` in turn, then the last one."""
    verdicts = list(verdicts)

    async def check(public, local):
        verdict = verdicts.pop(0) if len(verdicts) > 1 else verdicts[0]
        return verdict, "probe failed"

    supervisor = Supervisor(tunnel)
    supervisor._check = check
    tunnel._supervisor = supervisor
    supervisor.start()
    return supervisor


async def _until(condition, timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        assert asyncio.get_running_loop().time() < deadline, "condition never became true"
        await asyncio.sleep(0.005)


@pytest.mark.asyncio
async def test_supervisor_leaves_a_serving_tunnel_alone():
    states = []
    tunnel = _supervised_tunnel(states)
    supervisor = _supervise(tunnel, [_Verdict.SERVING])

    await asyncio.sleep(0.2)
    await supervisor.stop()

    tunnel.restart.assert_not_awaited()
    assert tunnel.state is TunnelState.CONNECTED
    assert states == []


@pytest.mark.asyncio
async def test_supervisor_restarts_frpc_that_exited():
    states = []
    tunnel = _supervised_tunnel(states)
    tunnel._process.poll.return_value = 1

    async def restart():
        tunnel._process.poll.return_value = None
        return tunnel.url

    tunnel.restart.side_effect = restart
    supervisor = _supervise(tunnel, [_Verdict.SERVING])

    await _until(lambda: states == [TunnelState.RECOVERING, TunnelState.CONNECTED])
    await supervisor.stop()

    tunnel.restart.assert_awaited_once()


@pytest.mark.asyncio
async def test_supervisor_restarts_only_after_a_second_failed_probe():
    tunnel = _supervised_tunnel()
    # One failed probe that the next one does not confirm, twice over.
    supervisor = _supervise(
        tunnel,
        [_Verdict.NOT_SERVING, _Verdict.SERVING, _Verdict.NOT_SERVING, _Verdict.SERVING],
    )

    await asyncio.sleep(0.3)
    await supervisor.stop()

    tunnel.restart.assert_not_awaited()
    assert tunnel.state is TunnelState.CONNECTED


@pytest.mark.asyncio
async def test_supervisor_restarts_a_tunnel_that_stays_down_and_reports_recovery():
    states = []
    tunnel = _supervised_tunnel(states)
    supervisor = _supervise(tunnel, [_Verdict.NOT_SERVING, _Verdict.NOT_SERVING, _Verdict.SERVING])

    await _until(lambda: states == [TunnelState.RECOVERING, TunnelState.CONNECTED])
    await supervisor.stop()

    tunnel.restart.assert_awaited_once()


@pytest.mark.asyncio
async def test_supervisor_does_not_blame_the_tunnel_for_a_local_service_that_is_down():
    tunnel = _supervised_tunnel()
    supervisor = _supervise(tunnel, [_Verdict.UNKNOWN])

    await asyncio.sleep(0.2)
    await supervisor.stop()

    tunnel.restart.assert_not_awaited()
    assert tunnel.state is TunnelState.CONNECTED


@pytest.mark.asyncio
async def test_supervisor_keeps_restarting_with_backoff_while_restarts_fail():
    tunnel = _supervised_tunnel()
    tunnel._process.poll.return_value = 1
    tunnel._client.get_tunnel.return_value = tunnel._tunnel_info.model_copy(
        update={"status": "disconnected"}
    )
    tunnel.restart.side_effect = TunnelConnectionError("frps unreachable")
    delays = []
    backoff = Supervisor._backoff

    def record(self, attempt):
        delays.append(attempt)
        return backoff(self, attempt)

    with patch.object(Supervisor, "_backoff", record):
        supervisor = _supervise(tunnel, [_Verdict.SERVING])
        await _until(lambda: tunnel.restart.await_count >= 4)
        await supervisor.stop()

    assert delays[:4] == [0, 1, 2, 3]
    assert tunnel.state is TunnelState.RECOVERING
    assert not tunnel.is_gone


def test_backoff_doubles_with_jitter_up_to_the_maximum(monkeypatch):
    monkeypatch.setattr(Supervisor, "backoff_base", 1.0)
    monkeypatch.setattr(Supervisor, "backoff_max", 60.0)
    supervisor = object.__new__(Supervisor)

    assert supervisor._backoff(0) == 0.0
    for attempt, full in ((1, 1.0), (2, 2.0), (5, 16.0), (7, 60.0), (30, 60.0)):
        for _ in range(20):
            assert full / 2 <= supervisor._backoff(attempt) <= full


@pytest.mark.asyncio
async def test_supervisor_marks_the_tunnel_dead_when_a_restart_finds_the_registration_gone():
    states = []
    tunnel = _supervised_tunnel(states)
    tunnel._process.poll.return_value = 1
    tunnel.restart.side_effect = TunnelGoneError(message="Tunnel is inactive")
    supervisor = _supervise(tunnel, [_Verdict.SERVING])

    assert await asyncio.wait_for(tunnel.wait_dead(), timeout=3.0) is True
    await asyncio.wait_for(supervisor._task, timeout=3.0)

    assert states == [TunnelState.RECOVERING, TunnelState.DEAD]
    assert tunnel.state is TunnelState.DEAD
    tunnel.restart.assert_awaited_once()


@pytest.mark.asyncio
async def test_supervisor_asks_the_service_when_a_restart_fails_and_gives_up_if_it_is_gone():
    tunnel = _supervised_tunnel()
    tunnel._process.poll.return_value = 1
    tunnel.restart.side_effect = TunnelConnectionError("login failed")
    tunnel._client.get_tunnel.return_value = tunnel._tunnel_info.model_copy(
        update={"status": "terminated"}
    )
    supervisor = _supervise(tunnel, [_Verdict.SERVING])

    assert await asyncio.wait_for(tunnel.wait_dead(), timeout=3.0) is True
    await asyncio.wait_for(supervisor._task, timeout=3.0)

    tunnel.restart.assert_awaited_once()


@pytest.mark.asyncio
async def test_supervisor_keeps_trying_when_the_service_cannot_be_asked():
    tunnel = _supervised_tunnel()
    tunnel._process.poll.return_value = 1
    tunnel.restart.side_effect = TunnelConnectionError("network down")
    tunnel._client.get_tunnel.side_effect = Exception("API down")
    supervisor = _supervise(tunnel, [_Verdict.SERVING])

    await _until(lambda: tunnel.restart.await_count >= 3)
    await supervisor.stop()

    assert not tunnel.is_gone


@pytest.mark.asyncio
async def test_supervisor_probes_at_once_when_frpc_reports_a_refused_proxy(monkeypatch):
    monkeypatch.setattr(Supervisor, "probe_interval", 60.0)
    monkeypatch.setattr(Supervisor, "settle", 60.0)
    tunnel = _supervised_tunnel()
    supervisor = _supervise(tunnel, [_Verdict.NOT_SERVING])

    await asyncio.sleep(0.05)
    tunnel.restart.assert_not_awaited()
    supervisor.on_frpc_event(FrpcEvent.PROXY_REFUSED)
    await _until(lambda: tunnel.restart.await_count >= 1)
    await supervisor.stop()


@pytest.mark.asyncio
async def test_supervisor_stays_out_of_a_restart_someone_else_is_running():
    tunnel = _supervised_tunnel()
    tunnel._process.poll.return_value = 1

    async with tunnel._process_lock:
        supervisor = _supervise(tunnel, [_Verdict.SERVING])
        await asyncio.sleep(0.1)
        tunnel.restart.assert_not_awaited()
        tunnel._process.poll.return_value = None
    await asyncio.sleep(0.1)
    await supervisor.stop()

    tunnel.restart.assert_not_awaited()


@pytest.mark.asyncio
async def test_supervisor_stops_watching_a_stopped_tunnel():
    tunnel = _supervised_tunnel()
    supervisor = _supervise(tunnel, [_Verdict.SERVING])

    tunnel._stopping = True
    await asyncio.wait_for(supervisor._task, timeout=3.0)

    tunnel.restart.assert_not_awaited()


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(self.server.status)
        if self.server.stamp:
            self.send_header("X-Service", "1")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):
        pass


def _serve(status: int = 200, stamp: bool = False) -> http.server.ThreadingHTTPServer:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.status, server.stamp = status, stamp
    threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True).start()
    return server


async def _check(edge, local, probe_header=None):
    """Probe with `edge` standing in for the public URL and `local` for the service."""
    tunnel = _make_started_tunnel()
    closed = _serve()
    closed_port = closed.server_address[1]
    closed.shutdown()
    closed.server_close()
    edge_port = edge.server_address[1] if edge else closed_port
    tunnel._tunnel_info = tunnel._tunnel_info.model_copy(
        update={"url": f"http://127.0.0.1:{edge_port}"}
    )
    tunnel.local_port = local.server_address[1] if local else closed_port
    supervisor = Supervisor(tunnel, probe_path="health", probe_header=probe_header)
    import httpx

    try:
        async with httpx.AsyncClient(trust_env=False) as client:
            verdict, _ = await supervisor._check(client, client)
    finally:
        for server in (edge, local):
            if server:
                server.shutdown()
                server.server_close()
    return verdict


@pytest.mark.asyncio
async def test_check_takes_a_good_response_through_the_url_as_serving():
    assert await _check(_serve(200), None) is _Verdict.SERVING


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [404, 502, 503])
async def test_check_blames_the_tunnel_when_only_the_url_fails(status):
    assert await _check(_serve(status), _serve(200)) is _Verdict.NOT_SERVING


@pytest.mark.asyncio
async def test_check_blames_the_tunnel_when_the_url_does_not_answer():
    assert await _check(None, _serve(200)) is _Verdict.NOT_SERVING


@pytest.mark.asyncio
async def test_check_accepts_an_error_the_local_service_returns_itself():
    assert await _check(_serve(404), _serve(404)) is _Verdict.SERVING


@pytest.mark.asyncio
async def test_check_cannot_judge_when_the_local_service_is_down():
    assert await _check(_serve(404), None) is _Verdict.UNKNOWN
    assert await _check(None, None) is _Verdict.UNKNOWN


@pytest.mark.asyncio
async def test_check_with_a_header_goes_by_the_header_not_the_status():
    header = "X-Service"
    assert await _check(_serve(500, stamp=True), None, header) is _Verdict.SERVING
    assert await _check(_serve(200), _serve(200, stamp=True), header) is _Verdict.NOT_SERVING


@pytest.mark.asyncio
async def test_check_with_a_header_cannot_judge_a_service_that_does_not_set_it(caplog):
    with caplog.at_level("WARNING", logger="prime_tunnel.supervisor"):
        assert await _check(_serve(200), _serve(200), "X-Service") is _Verdict.UNKNOWN
    assert "does not set the X-Service header" in caplog.text


@pytest.mark.asyncio
async def test_get_gives_up_on_a_response_that_takes_too_long(monkeypatch):
    monkeypatch.setattr(Supervisor, "probe_timeout", 0.05)
    supervisor = Supervisor(_make_started_tunnel())

    class Slow:
        def stream(self, *args, **kwargs):
            class Context:
                async def __aenter__(self):
                    await asyncio.sleep(5)

                async def __aexit__(self, *exc):
                    return False

            return Context()

    assert await supervisor._get(Slow(), "http://example.invalid/") is None


def test_response_keeps_status_and_headers():
    import httpx

    response = _Response(204, httpx.Headers({"X-Service": "1"}))
    assert response.status_code == 204
    assert "x-service" in response.headers


@pytest.mark.asyncio
async def test_state_and_wait_dead_without_a_supervisor():
    tunnel = Tunnel(local_port=8080)
    assert tunnel.state is TunnelState.STOPPED
    assert await asyncio.wait_for(tunnel.wait_dead(), timeout=1.0) is False

    states = []
    tunnel = _supervised_tunnel(states)
    assert tunnel.state is TunnelState.CONNECTED
    waiter = asyncio.create_task(tunnel.wait_dead())
    await asyncio.sleep(0.02)
    assert not waiter.done()

    # As a reader thread does when frpc's reconnect is rejected.
    await asyncio.to_thread(tunnel._mark_gone)

    assert await asyncio.wait_for(waiter, timeout=1.0) is True
    assert tunnel.state is TunnelState.DEAD
    assert states == [TunnelState.DEAD]


@pytest.mark.asyncio
async def test_wait_dead_returns_false_when_the_tunnel_is_stopped():
    states = []
    tunnel = _supervised_tunnel(states)
    waiter = asyncio.create_task(tunnel.wait_dead())
    await asyncio.sleep(0.02)

    await tunnel.stop()

    assert await asyncio.wait_for(waiter, timeout=1.0) is False
    assert states == [TunnelState.STOPPED]


@pytest.mark.asyncio
async def test_a_failing_state_callback_does_not_break_the_tunnel():
    tunnel = _supervised_tunnel()

    def fail(state):
        raise RuntimeError("boom")

    tunnel.on_state_change = fail
    tunnel._mark_gone()
    assert await asyncio.wait_for(tunnel.wait_dead(), timeout=1.0) is True


@pytest.mark.asyncio
async def test_status_reports_a_gone_registration_to_waiters():
    tunnel = _supervised_tunnel()
    tunnel._client.get_tunnel.return_value = None

    assert (await tunnel.status()).gone is True
    assert await asyncio.wait_for(tunnel.wait_dead(), timeout=1.0) is True


@pytest.mark.asyncio
async def test_sync_stop_cancels_the_supervisor():
    tunnel = _supervised_tunnel()
    supervisor = _supervise(tunnel, [_Verdict.SERVING])

    with patch("prime_tunnel.tunnel.httpx.delete"):
        tunnel.sync_stop()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(supervisor._task, timeout=1.0)
    assert tunnel.state is TunnelState.STOPPED
    assert await asyncio.wait_for(tunnel.wait_dead(), timeout=1.0) is False


@pytest.mark.asyncio
async def test_supervised_tunnel_restarts_a_real_frpc_that_dies(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    connected = _frpc_line("I", "[t-test123] start proxy success")
    # Dies shortly after connecting the first time, stays up the second.
    frpc = _fake_frpc_binary(
        tmp_path,
        f"echo '{connected}'\n"
        f"if [ -e '{tmp_path}/ran' ]; then exec sleep 30; fi\n"
        f"touch '{tmp_path}/ran'\nsleep 0.2\nexit 1\n",
    )
    states = []
    tunnel = Tunnel(
        local_port=8080, connection_timeout=10.0, supervise=True, on_state_change=states.append
    )
    _client_for(tunnel)

    async def check(self, public, local):
        return _Verdict.SERVING, ""

    monkeypatch.setattr(Supervisor, "_check", check)
    with patch("prime_tunnel.tunnel.get_frpc_path", return_value=frpc):
        await tunnel.start()
        first = tunnel._process
        try:
            expected = [TunnelState.CONNECTED, TunnelState.RECOVERING, TunnelState.CONNECTED]
            await _until(lambda: states == expected, timeout=10.0)
            assert tunnel._process is not first
            assert tunnel.is_running
        finally:
            await tunnel.stop()

    assert states[-1] is TunnelState.STOPPED
    assert tunnel._supervisor is None


@pytest.mark.asyncio
async def test_supervised_tunnel_dies_and_ends_frpc_when_a_reconnect_is_rejected(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    connected = _frpc_line("I", "[t-test123] start proxy success")
    rejected = _frpc_line("W", "connect to server error: Tunnel is inactive")
    frpc = _fake_frpc_binary(
        tmp_path, f"echo '{connected}'\nsleep 0.2\necho '{rejected}'\nexec sleep 30\n"
    )
    states = []
    tunnel = Tunnel(
        local_port=8080, connection_timeout=10.0, supervise=True, on_state_change=states.append
    )
    _client_for(tunnel)

    async def check(self, public, local):
        return _Verdict.SERVING, ""

    monkeypatch.setattr(Supervisor, "_check", check)
    with patch("prime_tunnel.tunnel.get_frpc_path", return_value=frpc):
        await tunnel.start()
        try:
            assert await asyncio.wait_for(tunnel.wait_dead(), timeout=10.0) is True
            await _until(lambda: not tunnel.is_running, timeout=10.0)
            assert states == [TunnelState.CONNECTED, TunnelState.DEAD]
        finally:
            await tunnel.stop()
