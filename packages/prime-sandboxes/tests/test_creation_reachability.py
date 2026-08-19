"""Creation waits distinguish transient reachability from local SDK failures."""

from types import SimpleNamespace
from typing import Any, Callable, cast

import httpx
import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError

from prime_sandboxes.core import APIError
from prime_sandboxes.core.client import APIClient
from prime_sandboxes.exceptions import SandboxNotRunningError
from prime_sandboxes.models import BatchSandboxStatusResponse
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


def _rpc_sandbox_not_found_error() -> SandboxNotRunningError:
    cause = ConnectError(Code.NOT_FOUND, "sandbox not found")
    error = SandboxNotRunningError(
        "sandbox-a",
        status="TERMINATED",
        error_type="SANDBOX_NOT_FOUND",
    )
    error.__cause__ = cause
    return error


def _http_sandbox_not_found_error() -> SandboxNotRunningError:
    request = httpx.Request("POST", "https://gateway.example.com/ns/job/exec")
    response = httpx.Response(
        502,
        request=request,
        json={"error": "sandbox_not_found"},
    )
    cause = httpx.HTTPStatusError("bad gateway", request=request, response=response)
    error = SandboxNotRunningError(
        "sandbox-a",
        status="TERMINATED",
        error_type="SANDBOX_NOT_FOUND",
    )
    error.__cause__ = cause
    return error


def _running_batch_status(sandbox_ids: list[str]) -> BatchSandboxStatusResponse:
    return BatchSandboxStatusResponse.model_validate(
        {
            "statuses": [
                {
                    "sandbox_id": sandbox_id,
                    "status": "RUNNING",
                    "error_type": None,
                    "error_message": None,
                    "pending_image_build_id": None,
                }
                for sandbox_id in sandbox_ids
            ],
            "errors": [],
        }
    )


def _reachable_after(
    failure_factory: Callable[[], SandboxNotRunningError],
) -> tuple[Callable[[str], bool], list[str]]:
    calls: list[str] = []

    def reachable(sandbox_id: str) -> bool:
        calls.append(sandbox_id)
        if len(calls) == 1:
            raise failure_factory()
        return True

    return reachable, calls


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


def test_sync_wait_retries_gateway_not_found_during_reachability(monkeypatch) -> None:
    client = SandboxClient(APIClient(api_key="test-key"))
    cast(Any, client)._sandbox_status_batcher = SimpleNamespace(
        get=lambda sandbox_id: _running_status(sandbox_id)
    )
    reachable, calls = _reachable_after(_rpc_sandbox_not_found_error)
    cast(Any, client)._is_sandbox_reachable = reachable
    monkeypatch.setattr("prime_sandboxes.sandbox._creation_poll_delay", lambda _: 0)

    client.wait_for_creation("sandbox-a", max_attempts=1)

    assert calls == ["sandbox-a", "sandbox-a"]


def test_sync_bulk_wait_retries_gateway_not_found_during_reachability(monkeypatch) -> None:
    client = SandboxClient(APIClient(api_key="test-key"))
    cast(Any, client).get_sandbox_statuses = _running_batch_status
    reachable, calls = _reachable_after(_http_sandbox_not_found_error)
    cast(Any, client)._is_sandbox_reachable = reachable
    monkeypatch.setattr("prime_sandboxes.sandbox.time.sleep", lambda _: None)

    statuses = client.bulk_wait_for_creation(["sandbox-a"], max_attempts=2)

    assert statuses == {"sandbox-a": "RUNNING"}
    assert calls == ["sandbox-a", "sandbox-a"]


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


@pytest.mark.asyncio
async def test_async_wait_retries_gateway_not_found_during_reachability(monkeypatch) -> None:
    client = AsyncSandboxClient(api_key="test-key")

    async def status(sandbox_id: str) -> SimpleNamespace:
        return _running_status(sandbox_id)

    calls: list[str] = []

    async def reachable(sandbox_id: str) -> bool:
        calls.append(sandbox_id)
        if len(calls) == 1:
            raise _http_sandbox_not_found_error()
        return True

    cast(Any, client)._sandbox_status_batcher = SimpleNamespace(get=status)
    cast(Any, client)._is_sandbox_reachable = reachable
    monkeypatch.setattr("prime_sandboxes.sandbox._creation_poll_delay", lambda _: 0)

    try:
        await client.wait_for_creation("sandbox-a", max_attempts=1)
    finally:
        await client.aclose()

    assert calls == ["sandbox-a", "sandbox-a"]


@pytest.mark.asyncio
async def test_async_bulk_wait_retries_gateway_not_found_during_reachability(
    monkeypatch,
) -> None:
    client = AsyncSandboxClient(api_key="test-key")

    async def statuses(sandbox_ids: list[str]) -> BatchSandboxStatusResponse:
        return _running_batch_status(sandbox_ids)

    calls: list[str] = []

    async def reachable(sandbox_id: str) -> bool:
        calls.append(sandbox_id)
        if len(calls) == 1:
            raise _rpc_sandbox_not_found_error()
        return True

    cast(Any, client).get_sandbox_statuses = statuses
    cast(Any, client)._is_sandbox_reachable = reachable
    monkeypatch.setattr("prime_sandboxes.sandbox.asyncio.sleep", lambda _: _async_noop())

    try:
        result = await client.bulk_wait_for_creation(["sandbox-a"], max_attempts=2)
    finally:
        await client.aclose()

    assert result == {"sandbox-a": "RUNNING"}
    assert calls == ["sandbox-a", "sandbox-a"]


async def _async_noop() -> None:
    return None
