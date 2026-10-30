import asyncio
import http.server
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from test_tunnel import _client_for, _fake_frpc_binary, _frpc_line, _make_started_tunnel

from prime_tunnel import Tunnel
from prime_tunnel import tunnel as tunnel_module
from prime_tunnel.exceptions import TunnelConnectionError, TunnelError, TunnelGoneError


@pytest.fixture(autouse=True)
def fast_timings(monkeypatch):
    """Shrink keep_serving's timings so tests take milliseconds."""
    for name, value in (
        ("_PROBE_INTERVAL", 0.02),
        ("_PROBE_TIMEOUT", 0.2),
        ("_PROBE_GRACE", 0.02),
        ("_RESTART_SETTLE", 0.01),
        ("_RESTART_BACKOFF_MAX", 0.04),
    ):
        monkeypatch.setattr(tunnel_module, name, value)
    monkeypatch.setattr(tunnel_module.random, "uniform", lambda low, high: 0.01)


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        # Anything else on the machine may probe a port it sees open.
        if self.path != "/health":
            self.send_response(404)
            self.end_headers()
            return
        self.server.requests.append((self.path, dict(self.headers)))
        answer = self.server.answers[0]
        if len(self.server.answers) > 1:
            self.server.answers.pop(0)
        if answer == "hang":
            time.sleep(0.5)
            return
        self.send_response(answer)
        self.end_headers()
        self.wfile.write(b"page")

    def log_message(self, *args):
        pass


@pytest.fixture
def serve():
    """Start an HTTP server giving `answers` in turn, then the last one for good."""
    servers = []

    def start(*answers) -> http.server.ThreadingHTTPServer:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        server.daemon_threads = True
        server.answers, server.requests = list(answers), []
        threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True).start()
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def _tunnel(edge, local) -> Tunnel:
    """A started tunnel with a live frpc, `edge` as its public URL and `local` as its service."""
    tunnel = _make_started_tunnel()
    tunnel._process.poll.return_value = None
    tunnel._tunnel_info = tunnel._tunnel_info.model_copy(
        update={"url": f"http://127.0.0.1:{edge.server_address[1]}"}
    )
    tunnel.local_port = local.server_address[1]
    tunnel.restart = AsyncMock(return_value=tunnel.url)
    return tunnel


async def _until(condition, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition never became true"
        await asyncio.sleep(0.005)


@pytest_asyncio.fixture
async def watching():
    """Run keep_serving as a task for the test and cancel it afterwards."""
    tasks = []

    def start(tunnel: Tunnel, path: str = "/health") -> asyncio.Task:
        task = asyncio.create_task(tunnel.keep_serving(path))
        tasks.append(task)
        return task

    yield start
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_a_serving_tunnel_is_left_alone(serve, watching):
    edge, local = serve(200), serve(200)
    tunnel = _tunnel(edge, local)
    watching(tunnel)

    await _until(lambda: len(edge.requests) >= 3)

    tunnel.restart.assert_not_awaited()
    assert local.requests == []


@pytest.mark.asyncio
async def test_404_page_through_the_url_restarts_frpc_after_a_second_probe(serve, watching):
    edge, local = serve(404), serve(200)
    tunnel = _tunnel(edge, local)
    tunnel.restart.side_effect = lambda: edge.answers.__setitem__(0, 200)
    watching(tunnel)

    await _until(lambda: tunnel.restart.await_count == 1)

    assert len(edge.requests) == 2
    assert len(local.requests) == 1


@pytest.mark.asyncio
async def test_connected_tunnel_that_hangs_requests_restarts_frpc(serve, watching):
    """frps holds requests while frpc stays connected, so nothing but the URL shows it."""
    edge, local = serve("hang"), serve(200)
    tunnel = _tunnel(edge, local)
    watching(tunnel)

    await _until(lambda: tunnel.restart.await_count == 1)

    assert len(local.requests) == 1


@pytest.mark.asyncio
async def test_a_tunnel_that_recovers_by_itself_is_not_restarted(serve, watching):
    edge, local = serve(404, 200), serve(200)
    tunnel = _tunnel(edge, local)
    watching(tunnel)

    await _until(lambda: len(edge.requests) >= 4)

    tunnel.restart.assert_not_awaited()


@pytest.mark.parametrize("answer", [404, 500])
@pytest.mark.asyncio
async def test_a_failing_local_service_is_not_blamed_on_the_tunnel(serve, watching, answer):
    edge, local = serve(answer), serve(answer)
    tunnel = _tunnel(edge, local)
    watching(tunnel)

    await _until(lambda: len(local.requests) >= 3)

    tunnel.restart.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_local_service_that_is_down_is_not_blamed_on_the_tunnel(serve, watching):
    edge, closed = serve(404), serve(200)
    tunnel = _tunnel(edge, closed)
    closed.shutdown()
    closed.server_close()
    watching(tunnel)

    await _until(lambda: len(edge.requests) >= 3)

    tunnel.restart.assert_not_awaited()


@pytest.mark.asyncio
async def test_frpc_that_exited_is_restarted_without_a_probe(serve, watching):
    edge, local = serve(200), serve(200)
    tunnel = _tunnel(edge, local)
    tunnel._process.poll.return_value = 1

    def restarted():
        tunnel._process.poll.return_value = None

    tunnel.restart.side_effect = restarted
    watching(tunnel)

    await _until(lambda: tunnel.restart.await_count == 1)

    assert edge.requests == []


@pytest.mark.asyncio
async def test_registration_gone_ends_the_task_with_the_error(serve, watching):
    edge, local = serve(404), serve(200)
    tunnel = _tunnel(edge, local)
    tunnel.restart.side_effect = TunnelGoneError(tunnel_id="t-test123", message="gone")

    with pytest.raises(TunnelGoneError):
        await asyncio.wait_for(watching(tunnel), timeout=3.0)


@pytest.mark.asyncio
async def test_failed_restarts_are_tried_again_until_one_works(serve, watching):
    edge, local = serve(200), serve(200)
    tunnel = _tunnel(edge, local)
    tunnel._process.poll.return_value = 1
    results = [TunnelConnectionError(message="no"), OSError("no")]

    def restart():
        if results:
            raise results.pop(0)
        tunnel._process.poll.return_value = None

    tunnel.restart.side_effect = restart
    watching(tunnel)

    await _until(lambda: len(edge.requests) >= 1)

    assert tunnel.restart.await_count == 3


@pytest.mark.asyncio
async def test_a_probe_that_passes_resets_the_backoff(serve, watching, monkeypatch):
    edge, local = serve(404, 404, 200, 404, 404, 200), serve(200)
    tunnel = _tunnel(edge, local)
    delays = []
    real = tunnel_module._restart_delay
    monkeypatch.setattr(
        tunnel_module, "_restart_delay", lambda attempt: delays.append(attempt) or real(attempt)
    )
    watching(tunnel)

    await _until(lambda: len(delays) >= 2)

    assert delays == [0, 0]


def test_restart_delay_doubles_up_to_the_maximum(monkeypatch):
    monkeypatch.setattr(tunnel_module, "_RESTART_BACKOFF_MAX", 60.0)
    monkeypatch.setattr(tunnel_module.random, "uniform", lambda low, high: high)

    delays = [tunnel_module._restart_delay(attempt) for attempt in range(9)]

    assert delays == [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]
    assert tunnel_module._restart_delay(5000) == 60.0


@pytest.mark.asyncio
async def test_task_ends_when_the_tunnel_is_stopped(serve, watching):
    edge, local = serve(200), serve(200)
    tunnel = _tunnel(edge, local)
    task = watching(tunnel)
    await _until(lambda: len(edge.requests) >= 1)

    tunnel._started = False

    await asyncio.wait_for(task, timeout=3.0)
    tunnel.restart.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelling_the_task_stops_watching(serve, watching):
    edge, local = serve(200), serve(200)
    tunnel = _tunnel(edge, local)
    task = watching(tunnel)
    await _until(lambda: len(edge.requests) >= 1)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_a_tunnel_that_is_not_started_cannot_be_watched():
    with pytest.raises(TunnelError):
        await Tunnel(local_port=8080).keep_serving("/health")


@pytest.mark.asyncio
async def test_probes_ask_for_the_health_path_and_get_past_the_edge(serve, watching):
    edge, local = serve(404), serve(200)
    tunnel = _tunnel(edge, local)
    watching(tunnel, "health")

    await _until(lambda: len(local.requests) >= 1)

    path, headers = edge.requests[0]
    assert path == "/health"
    assert "python" not in headers["User-Agent"].lower()
    path, headers = local.requests[0]
    assert path == "/health"
    assert headers["Host"] == tunnel.hostname


@pytest.mark.asyncio
async def test_only_the_url_gets_the_tunnels_basic_auth(serve, watching):
    edge, local = serve(404), serve(200)
    tunnel = _tunnel(edge, local)
    tunnel.http_user = "user"
    tunnel._tunnel_info = tunnel._tunnel_info.model_copy(update={"http_password": "secret"})
    watching(tunnel)

    await _until(lambda: len(local.requests) >= 1)

    assert "Authorization" in edge.requests[0][1]
    assert "Authorization" not in local.requests[0][1]


@pytest.mark.asyncio
async def test_a_real_frpc_that_dies_is_replaced(serve, watching, monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    connected = _frpc_line("I", "[t-test123] start proxy success")
    # Dies shortly after connecting the first time, stays up the second.
    frpc = _fake_frpc_binary(
        tmp_path,
        f"echo '{connected}'\n"
        f"if [ -e '{tmp_path}/ran' ]; then exec sleep 30; fi\n"
        f"touch '{tmp_path}/ran'\nsleep 0.2\nexit 1\n",
    )
    edge, local = serve(200), serve(200)
    tunnel = Tunnel(local_port=local.server_address[1], connection_timeout=10.0)
    client = _client_for(tunnel)
    client.create_tunnel.return_value = client.create_tunnel.return_value.model_copy(
        update={"url": f"http://127.0.0.1:{edge.server_address[1]}"}
    )

    with patch("prime_tunnel.tunnel.get_frpc_path", return_value=frpc):
        await tunnel.start()
        first = tunnel._process
        try:
            watching(tunnel)
            await _until(lambda: tunnel._process is not first and tunnel.is_running, timeout=10.0)
        finally:
            await tunnel.stop()


@pytest.mark.asyncio
async def test_a_real_frpc_that_is_rejected_ends_the_task_and_frpc(
    serve, watching, monkeypatch, tmp_path
):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    connected = _frpc_line("I", "[t-test123] start proxy success")
    rejected = _frpc_line("W", "connect to server error: Tunnel is inactive")
    # The first frpc connects and stops serving; the service rejects the next one.
    frpc = _fake_frpc_binary(
        tmp_path,
        f"if [ -e '{tmp_path}/ran' ]; then echo '{rejected}'; exec sleep 30; fi\n"
        f"touch '{tmp_path}/ran'\necho '{connected}'\nexec sleep 30\n",
    )
    edge, local = serve(404), serve(200)
    tunnel = Tunnel(local_port=local.server_address[1], connection_timeout=10.0)
    client = _client_for(tunnel)
    client.create_tunnel.return_value = client.create_tunnel.return_value.model_copy(
        update={"url": f"http://127.0.0.1:{edge.server_address[1]}"}
    )

    with patch("prime_tunnel.tunnel.get_frpc_path", return_value=frpc):
        await tunnel.start()
        try:
            with pytest.raises(TunnelGoneError):
                await asyncio.wait_for(watching(tunnel), timeout=10.0)
            assert not tunnel.is_running
            assert tunnel.is_gone
        finally:
            await tunnel.stop()
