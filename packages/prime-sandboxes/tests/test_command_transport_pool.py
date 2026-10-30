"""Command streams spread over pooled transports below the gateway's stream cap."""

import contextlib
from datetime import datetime, timedelta, timezone
from typing import Any, cast

import pytest

from prime_sandboxes._proto.command_session import command_session_pb2
from prime_sandboxes.sandbox import (
    _COMMAND_STREAMS_PER_TRANSPORT,
    AsyncSandboxClient,
    _CommandTransportPool,
)


def test_pool_opens_a_transport_per_stream_cap_and_reuses_released_slots():
    made: list[object] = []

    def new_client() -> object:
        made.append(object())
        return made[-1]

    pool = _CommandTransportPool(new_client)
    with contextlib.ExitStack() as leases:
        held = [
            leases.enter_context(pool.lease()) for _ in range(_COMMAND_STREAMS_PER_TRANSPORT + 1)
        ]
        assert len(made) == 2
        assert held.count(made[0]) == _COMMAND_STREAMS_PER_TRANSPORT
        assert held[-1] is made[1]

    with pool.lease() as client:
        assert client is made[0]
    assert len(made) == 2


def _auth(gateway_url: str):
    class _Cache:
        async def get_or_refresh(self, _sandbox_id: str):
            return {
                "gateway_url": gateway_url,
                "user_ns": "ns",
                "job_id": "job",
                "token": "tok",
                "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat(),
            }

    return _Cache()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("gateway_url", "pooled"),
    [("https://gateway.example.com", True), ("http://localhost:8080", False)],
)
async def test_async_execute_uses_pooled_client_only_for_tls_gateways(
    monkeypatch, gateway_url, pooled
):
    seen: list[Any] = []

    class _FakeConnectClient:
        def __init__(self, _address: str, **kwargs):
            seen.append(kwargs.get("http_client"))

        def execute_server_stream(self, **_kwargs):
            event_type = command_session_pb2.CommandSessionEvent

            async def events():
                yield command_session_pb2.StartResponse(
                    event=event_type(end=event_type.EndEvent(exit_code=0))
                )

            return events()

        async def close(self):
            pass

    monkeypatch.setattr("prime_sandboxes.sandbox.ConnectClient", _FakeConnectClient)
    client = AsyncSandboxClient(api_key="test-key")
    cast(Any, client)._auth_cache = _auth(gateway_url)
    try:
        await client.execute_command("sbx", "true")
        await client.execute_command("sbx", "true")
    finally:
        await client.aclose()

    if pooled:
        assert seen[0] is not None and seen[0] is seen[1]
    else:
        assert seen == [None, None]
