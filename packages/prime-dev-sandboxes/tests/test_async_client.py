"""Tests for the async typed client (mocked transport)."""

from typing import Any, Dict, List

import pytest

from prime_dev_sandboxes.client import AsyncDevSandboxClient
from prime_dev_sandboxes.models import Operation, OperationState


class FakeAsyncAPIClient:
    """In-memory double for the core AsyncAPIClient."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.responses: Dict[str, Any] = {}
        self.error: Exception | None = None

    async def _record(self, method: str, endpoint: str, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append({"method": method, "endpoint": endpoint, **kwargs})
        if self.error is not None:
            raise self.error
        if endpoint not in self.responses:
            raise AssertionError(f"unexpected request: {method} {endpoint}")
        return self.responses[endpoint]

    async def get(self, endpoint: str, params: Any = None) -> Dict[str, Any]:
        return await self._record("GET", endpoint, params=params)

    async def post(self, endpoint: str, json: Any = None) -> Dict[str, Any]:
        return await self._record("POST", endpoint, json=json)

    async def delete(self, endpoint: str) -> Dict[str, Any]:
        return await self._record("DELETE", endpoint)


@pytest.mark.asyncio
async def test_async_create_posts_contract_body() -> None:
    api = FakeAsyncAPIClient()
    api.responses["/dev-sandboxes"] = {"environment_id": "dev-env-1", "operation_id": "op-1"}
    client = AsyncDevSandboxClient(client=api)  # type: ignore[arg-type]

    created = await client.create("platform-dev", "origin/develop", ttl_minutes=480)

    assert created.operation_id == "op-1"
    assert api.calls[0]["json"] == {
        "template": "platform-dev",
        "ref": "origin/develop",
        "ttl_minutes": 480,
        "team_id": None,
    }


@pytest.mark.asyncio
async def test_async_wait_for_operation_until_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    ops = [
        Operation.model_validate({"operation_id": "op-1", "kind": "exec", "state": "running"}),
        Operation.model_validate(
            {
                "operation_id": "op-1",
                "kind": "exec",
                "state": "failed",
                "error": {"code": "EXEC_TIMEOUT", "stage": "exec", "message": "timed out"},
            }
        ),
    ]
    it = iter(ops)

    async def fake_get(_self: Any, operation_id: str) -> Operation:
        return next(it)

    client = AsyncDevSandboxClient(client=FakeAsyncAPIClient())  # type: ignore[arg-type]
    monkeypatch.setattr(type(client), "get_operation", fake_get)

    sleeps: List[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    import prime_dev_sandboxes.client as client_module

    monkeypatch.setattr(client_module.asyncio, "sleep", fake_sleep)

    final = await client.wait_for_operation("op-1", poll_interval_seconds=0.01)

    assert final.state is OperationState.FAILED
    assert final.error is not None
    assert final.error.code == "EXEC_TIMEOUT"
    assert sleeps == [0.01]


@pytest.mark.asyncio
async def test_async_get_delete_exec() -> None:
    api = FakeAsyncAPIClient()
    api.responses["/dev-sandboxes/dev-env-1"] = {
        "id": "dev-env-1",
        "state": "READY",
        "template": "platform-dev",
        "ref": "main",
    }
    api.responses["/dev-sandboxes/dev-env-1/exec"] = {"operation_id": "op-exec"}
    api.responses["/dev-sandboxes/dev-env-1"] = api.responses["/dev-sandboxes/dev-env-1"]
    client = AsyncDevSandboxClient(client=api)  # type: ignore[arg-type]

    env = await client.get("dev-env-1")
    assert env.is_ready

    exec_response = await client.exec("dev-env-1", "ls -la")
    assert exec_response.operation_id == "op-exec"
    assert api.calls[-1]["json"] == {"command": "ls -la", "timeout_seconds": 300}

    api.responses["/dev-sandboxes/dev-env-1"] = {"operation_id": "op-del"}
    deleted = await client.delete("dev-env-1")
    assert deleted.operation_id == "op-del"
