"""Tests for the TunnelState state machine and on_state_change callback API."""

from unittest.mock import MagicMock

import pytest

from prime_tunnel import Tunnel, TunnelState


def _make_tunnel() -> Tunnel:
    return Tunnel(local_port=8080)


def test_initial_state_is_stopped():
    tunnel = _make_tunnel()
    assert tunnel.state is TunnelState.STOPPED


def test_set_state_emits_callback_with_old_and_new():
    tunnel = _make_tunnel()
    events: list[tuple[TunnelState, TunnelState]] = []
    tunnel.on_state_change(lambda old, new: events.append((old, new)))

    tunnel._set_state(TunnelState.CONNECTING)
    tunnel._set_state(TunnelState.CONNECTED)

    assert events == [
        (TunnelState.STOPPED, TunnelState.CONNECTING),
        (TunnelState.CONNECTING, TunnelState.CONNECTED),
    ]
    assert tunnel.state is TunnelState.CONNECTED


def test_set_state_is_idempotent():
    tunnel = _make_tunnel()
    events: list[tuple[TunnelState, TunnelState]] = []
    tunnel.on_state_change(lambda old, new: events.append((old, new)))

    tunnel._set_state(TunnelState.CONNECTED)
    tunnel._set_state(TunnelState.CONNECTED)

    assert len(events) == 1


def test_off_state_change_removes_callback():
    tunnel = _make_tunnel()
    calls: list[tuple[TunnelState, TunnelState]] = []

    def cb(old, new):
        calls.append((old, new))

    tunnel.on_state_change(cb)
    tunnel.off_state_change(cb)
    tunnel._set_state(TunnelState.CONNECTED)

    assert calls == []


def test_off_state_change_unknown_callback_is_silent():
    tunnel = _make_tunnel()
    tunnel.off_state_change(lambda *_: None)  # should not raise


def test_callback_exception_does_not_break_dispatch():
    tunnel = _make_tunnel()
    other_called = False

    def raising(old, new):
        raise RuntimeError("boom")

    def other(old, new):
        nonlocal other_called
        other_called = True

    tunnel.on_state_change(raising)
    tunnel.on_state_change(other)
    tunnel._set_state(TunnelState.CONNECTED)

    assert other_called


def test_on_state_change_returns_callback_for_decorator_use():
    tunnel = _make_tunnel()

    @tunnel.on_state_change
    def handler(old, new):
        pass

    # handler should be the same object returned and registered
    assert callable(handler)
    assert handler in tunnel._state_callbacks


# -- _handle_log_line transitions --------------------------------------------


def test_handle_log_start_proxy_success_marks_connected():
    tunnel = _make_tunnel()
    tunnel._set_state(TunnelState.CONNECTING)

    tunnel._handle_log_line("[I] [control.go:176] [abc] start proxy success")

    assert tunnel.state is TunnelState.CONNECTED


def test_handle_log_heartbeat_timeout_while_connected_marks_disconnected():
    tunnel = _make_tunnel()
    tunnel._set_state(TunnelState.CONNECTED)

    tunnel._handle_log_line("[W] [control.go:274] heartbeat timeout")

    assert tunnel.state is TunnelState.DISCONNECTED


def test_handle_log_pong_error_while_connected_marks_disconnected():
    tunnel = _make_tunnel()
    tunnel._set_state(TunnelState.CONNECTED)

    tunnel._handle_log_line(
        "[E] [control.go:196] pong message contains error: session expired"
    )

    assert tunnel.state is TunnelState.DISCONNECTED


def test_handle_log_try_to_connect_while_connected_marks_disconnected():
    """Reconnect attempt after a drop — the key prod signal."""
    tunnel = _make_tunnel()
    tunnel._set_state(TunnelState.CONNECTED)

    tunnel._handle_log_line("[I] [service.go:378] try to connect to server...")

    assert tunnel.state is TunnelState.DISCONNECTED


def test_handle_log_try_to_connect_while_connecting_is_ignored():
    """Initial connect also logs 'try to connect to server' — must not flip."""
    tunnel = _make_tunnel()
    tunnel._set_state(TunnelState.CONNECTING)

    tunnel._handle_log_line("[I] [service.go:378] try to connect to server...")

    assert tunnel.state is TunnelState.CONNECTING


def test_handle_log_strips_ansi_before_matching():
    tunnel = _make_tunnel()
    tunnel._set_state(TunnelState.CONNECTED)

    tunnel._handle_log_line(
        "\033[1;33m2026-04-20 19:32:34 [W] [control.go:274] heartbeat timeout\033[0m"
    )

    assert tunnel.state is TunnelState.DISCONNECTED


def test_handle_log_reconnect_cycle_emits_both_transitions():
    """Full prod pattern: CONNECTED -> DISCONNECTED -> CONNECTED on reconnect."""
    tunnel = _make_tunnel()
    tunnel._set_state(TunnelState.CONNECTED)

    events: list[tuple[TunnelState, TunnelState]] = []
    tunnel.on_state_change(lambda old, new: events.append((old, new)))

    tunnel._handle_log_line("[W] [control.go:274] heartbeat timeout")
    tunnel._handle_log_line("[I] [service.go:378] try to connect to server...")
    tunnel._handle_log_line("[I] [service.go:370] login to server success, get run id [x]")
    tunnel._handle_log_line("[I] [control.go:176] [t-abc] start proxy success")

    assert events == [
        (TunnelState.CONNECTED, TunnelState.DISCONNECTED),
        (TunnelState.DISCONNECTED, TunnelState.CONNECTED),
    ]


def test_handle_log_unrelated_line_does_not_change_state():
    tunnel = _make_tunnel()
    tunnel._set_state(TunnelState.CONNECTED)

    tunnel._handle_log_line("[D] [control.go:244] send heartbeat to server")

    assert tunnel.state is TunnelState.CONNECTED


# -- sync_stop / _cleanup integration ----------------------------------------


def test_sync_stop_transitions_to_stopped():
    from datetime import datetime, timezone

    from prime_tunnel.models import TunnelInfo

    tunnel = _make_tunnel()
    tunnel._set_state(TunnelState.CONNECTED)
    tunnel._started = True
    tunnel._process = MagicMock()
    tunnel._tunnel_info = TunnelInfo(
        tunnel_id="t-test123",
        hostname="t-test123.tunnel.example.com",
        url="https://t-test123.tunnel.example.com",
        frp_token="tok",
        server_host="frp.example.com",
        server_port=7000,
        expires_at=datetime.now(timezone.utc),
    )
    tunnel._config_file = MagicMock()
    tunnel._config_file.exists.return_value = True

    events: list[tuple[TunnelState, TunnelState]] = []
    tunnel.on_state_change(lambda old, new: events.append((old, new)))

    from unittest.mock import patch

    with patch("prime_tunnel.tunnel.httpx.delete"):
        tunnel.sync_stop()

    assert tunnel.state is TunnelState.STOPPED
    assert events[-1] == (TunnelState.CONNECTED, TunnelState.STOPPED)


@pytest.mark.asyncio
async def test_cleanup_transitions_to_stopped():
    tunnel = _make_tunnel()
    tunnel._set_state(TunnelState.DISCONNECTED)

    await tunnel._cleanup()

    assert tunnel.state is TunnelState.STOPPED
