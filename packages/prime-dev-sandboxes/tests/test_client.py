"""Tests for the typed dev-sandboxes clients (mocked transport)."""

from typing import Any, Dict, List

import pytest

from prime_dev_sandboxes import DevSandboxClient
from prime_dev_sandboxes.core import APIStatusError
from prime_dev_sandboxes.errors import DevSandboxError
from prime_dev_sandboxes.models import Operation, OperationState


class FakeAPIClient:
    """In-memory double for the core APIClient (sync)."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.responses: Dict[str, Any] = {}
        self.error: Exception | None = None

    def _record(self, method: str, endpoint: str, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append({"method": method, "endpoint": endpoint, **kwargs})
        if self.error is not None:
            raise self.error
        if endpoint not in self.responses:
            raise AssertionError(f"unexpected request: {method} {endpoint}")
        return self.responses[endpoint]

    def get(self, endpoint: str, params: Any = None) -> Dict[str, Any]:
        return self._record("GET", endpoint, params=params)

    def post(self, endpoint: str, json: Any = None) -> Dict[str, Any]:
        return self._record("POST", endpoint, json=json)

    def delete(self, endpoint: str) -> Dict[str, Any]:
        return self._record("DELETE", endpoint)


def _operation(
    state: str = "running", stage: str = "VERIFYING", **overrides: Any
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "operation_id": "op-1",
        "kind": "create",
        "state": state,
        "stage": stage,
    }
    payload.update(overrides)
    return payload


def test_create_posts_contract_body() -> None:
    api = FakeAPIClient()
    api.responses["/dev-sandboxes"] = {"environment_id": "dev-env-1", "operation_id": "op-1"}
    client = DevSandboxClient(client=api)  # type: ignore[arg-type]

    created = client.create("platform-dev", "origin/develop", ttl_minutes=240, team_id="team-9")

    assert created.environment_id == "dev-env-1"
    assert created.operation_id == "op-1"
    assert api.calls[0]["json"] == {
        "template": "platform-dev",
        "ref": "origin/develop",
        "ttl_minutes": 240,
        "team_id": "team-9",
    }


def test_list_passes_pagination() -> None:
    api = FakeAPIClient()
    api.responses["/dev-sandboxes"] = {
        "environments": [
            {
                "id": "dev-env-1",
                "state": "READY",
                "template": "platform-dev",
                "ref": "main",
            }
        ],
        "total": 1,
        "page": 2,
        "per_page": 10,
        "has_next": False,
    }
    client = DevSandboxClient(client=api)  # type: ignore[arg-type]

    page = client.list(page=2, per_page=10)

    assert api.calls[0]["params"] == {"page": 2, "per_page": 10}
    assert page.environments[0].id == "dev-env-1"
    assert page.environments[0].is_ready


def test_get_and_delete_and_exec() -> None:
    api = FakeAPIClient()
    api.responses["/dev-sandboxes/dev-env-1"] = {
        "id": "dev-env-1",
        "state": "READY",
        "template": "platform-dev",
        "ref": "main",
        "urls": {"frontend": "https://fe.example.com"},
    }
    api.responses["/dev-sandboxes/dev-env-1/exec"] = {"operation_id": "op-exec"}
    client = DevSandboxClient(client=api)  # type: ignore[arg-type]

    env = client.get("dev-env-1")
    assert env.urls["frontend"] == "https://fe.example.com"

    exec_response = client.exec("dev-env-1", "make test", timeout_seconds=60)
    assert exec_response.operation_id == "op-exec"
    assert api.calls[-1]["json"] == {"command": "make test", "timeout_seconds": 60}


def test_list_templates() -> None:
    api = FakeAPIClient()
    api.responses["/dev-sandboxes/templates"] = {
        "templates": [
            {
                "id": "platform-dev",
                "image": "prime/prime-engineering/platform-dev:v3",
                "cpu": 8,
                "memory_gb": 32,
                "disk_gb": 60,
            }
        ]
    }
    client = DevSandboxClient(client=api)  # type: ignore[arg-type]

    templates = client.list_templates()
    assert templates[0].id == "platform-dev"
    assert templates[0].cpu == 8


def test_wait_for_operation_polls_until_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    api = FakeAPIClient()
    states = iter(["running", "running", "succeeded"])

    def get_operation(_self: Any, operation_id: str) -> Operation:
        return Operation.model_validate(_operation(next(states)))

    sleeps: List[float] = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    client = DevSandboxClient(client=api)  # type: ignore[arg-type]
    monkeypatch.setattr(type(client), "get_operation", get_operation)

    seen: List[str] = []
    final = client.wait_for_operation(
        "op-1",
        poll_interval_seconds=0.01,
        on_update=lambda op: seen.append(op.state.value),
    )

    assert final.state is OperationState.SUCCEEDED
    assert seen == ["running", "running", "succeeded"]
    assert sleeps == [0.01, 0.01]


def test_wait_for_operation_timeout() -> None:
    api = FakeAPIClient()
    api.responses["/dev-sandboxes/operations/op-1"] = _operation()
    client = DevSandboxClient(client=api)  # type: ignore[arg-type]
    client.poll_interval_seconds = 5.0

    with pytest.raises(DevSandboxError) as excinfo:
        # First snapshot is never terminal, timeout immediately exceeded.
        for _ in client.iter_operation("op-1", timeout_seconds=0.0001):
            pass

    assert excinfo.value.code == "PROVISION_TIMEOUT"


def test_api_status_error_translated_to_taxonomy() -> None:
    api = FakeAPIClient()
    api.error = APIStatusError(
        "HTTP 429: too many environments",
        429,
        {
            "error": {
                "code": "LIMIT_REACHED",
                "stage": "admission",
                "message": "2 active allocations per user",
            }
        },
    )
    client = DevSandboxClient(client=api)  # type: ignore[arg-type]

    with pytest.raises(DevSandboxError) as excinfo:
        client.create("platform-dev", "main")

    assert excinfo.value.code == "LIMIT_REACHED"
    assert excinfo.value.group == "admission"


def test_get_operation_maps_structured_error() -> None:
    api = FakeAPIClient()
    api.error = APIStatusError("HTTP 404: not found", 404, {"detail": "operation not found"})
    client = DevSandboxClient(client=api)  # type: ignore[arg-type]

    with pytest.raises(DevSandboxError) as excinfo:
        client.get_operation("op-missing")

    assert excinfo.value.code == "INVALID_INPUT"
