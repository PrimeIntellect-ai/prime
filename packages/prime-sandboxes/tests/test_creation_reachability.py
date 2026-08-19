"""Creation waits distinguish transient reachability from local SDK failures."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError

from prime_sandboxes.core import APIError
from prime_sandboxes.core.client import APIClient
from prime_sandboxes.exceptions import SandboxNotRunningError
from prime_sandboxes.sandbox import AsyncSandboxClient, SandboxClient


def _running_status(sandbox_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        sandbox_id=sandbox_id,
        status="RUNNING",
        error_type=None,
        error_message=None,
        pending_image_build_id=None,
    )


def _wrapped_rpc_error(
    code: Code,
    message: str,
    cause: BaseException | None = None,
) -> APIError:
    rpc_error = ConnectError(code, message)
    rpc_error.__cause__ = cause
    error = APIError(f"Connect RPC failed ({code.value}): {message}")
    error.__cause__ = rpc_error
    return error


def test_sync_wait_surfaces_local_codec_failure_immediately(monkeypatch) -> None:
    client = SandboxClient(APIClient(api_key="test-key"))
    cast(Any, client)._sandbox_status_batcher = SimpleNamespace(
        get=lambda sandbox_id: _running_status(sandbox_id)
    )
    failure = _wrapped_rpc_error(
        Code.UNAVAILABLE,
        "StartRequest has no attribute to_binary",
        AttributeError("StartRequest has no attribute to_binary"),
    )
    cast(Any, client)._is_sandbox_reachable = lambda _sandbox_id: (_ for _ in ()).throw(failure)
    monotonic = iter([0.0, 0.0, 0.0])
    monkeypatch.setattr("prime_sandboxes.sandbox.time.monotonic", lambda: next(monotonic))
    monkeypatch.setattr(
        "prime_sandboxes.sandbox.time.sleep",
        lambda _delay: pytest.fail("permanent reachability errors must not be retried"),
    )

    with pytest.raises(APIError, match="to_binary") as exc_info:
        client.wait_for_creation("sandbox-a", max_attempts=1)

    assert exc_info.value is failure


def test_sync_wait_reports_last_transient_reachability_error(monkeypatch) -> None:
    client = SandboxClient(APIClient(api_key="test-key"))
    cast(Any, client)._sandbox_status_batcher = SimpleNamespace(
        get=lambda sandbox_id: _running_status(sandbox_id)
    )
    failure = _wrapped_rpc_error(Code.UNAVAILABLE, "connection reset")
    cast(Any, client)._is_sandbox_reachable = lambda _sandbox_id: (_ for _ in ()).throw(failure)
    monotonic = iter([0.0, 0.0, 0.0, 2.0])
    monkeypatch.setattr("prime_sandboxes.sandbox.time.monotonic", lambda: next(monotonic))

    with pytest.raises(SandboxNotRunningError, match="reached RUNNING") as exc_info:
        client.wait_for_creation("sandbox-a", max_attempts=1)

    assert exc_info.value.error_type == "GATEWAY_REACHABILITY_TIMEOUT"
    assert "connection reset" in str(exc_info.value)
    assert exc_info.value.__cause__ is failure


@pytest.mark.asyncio
async def test_async_wait_surfaces_local_codec_failure_immediately(monkeypatch) -> None:
    client = AsyncSandboxClient(api_key="test-key")

    async def status(sandbox_id: str) -> SimpleNamespace:
        return _running_status(sandbox_id)

    failure = _wrapped_rpc_error(
        Code.UNAVAILABLE,
        "StartRequest has no attribute to_binary",
        AttributeError("StartRequest has no attribute to_binary"),
    )

    async def unreachable(_sandbox_id: str) -> bool:
        raise failure

    cast(Any, client)._sandbox_status_batcher = SimpleNamespace(get=status)
    cast(Any, client)._is_sandbox_reachable = unreachable
    monkeypatch.setattr("prime_sandboxes.sandbox._creation_timeout_seconds", lambda _: 0.001)

    try:
        with pytest.raises(APIError, match="to_binary") as exc_info:
            await client.wait_for_creation("sandbox-a", max_attempts=1)
    finally:
        await client.aclose()

    assert exc_info.value is failure


@pytest.mark.asyncio
async def test_async_wait_reports_last_transient_reachability_error(monkeypatch) -> None:
    client = AsyncSandboxClient(api_key="test-key")

    async def status(sandbox_id: str) -> SimpleNamespace:
        return _running_status(sandbox_id)

    failure = _wrapped_rpc_error(Code.UNAVAILABLE, "connection reset")

    async def unreachable(_sandbox_id: str) -> bool:
        raise failure

    cast(Any, client)._sandbox_status_batcher = SimpleNamespace(get=status)
    cast(Any, client)._is_sandbox_reachable = unreachable
    monkeypatch.setattr("prime_sandboxes.sandbox._creation_timeout_seconds", lambda _: 0.001)

    try:
        with pytest.raises(SandboxNotRunningError, match="reached RUNNING") as exc_info:
            await client.wait_for_creation("sandbox-a", max_attempts=1)
    finally:
        await client.aclose()

    assert exc_info.value.error_type == "GATEWAY_REACHABILITY_TIMEOUT"
    assert "connection reset" in str(exc_info.value)
    assert exc_info.value.__cause__ is failure
