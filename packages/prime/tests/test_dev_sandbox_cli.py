"""Tests for the `prime dev-sandbox` command (mocked client)."""

import json
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
from prime_cli.commands import dev_sandbox
from prime_cli.main import app
from prime_dev_sandboxes.models import DevEnvironment, Operation
from typer.testing import CliRunner

runner = CliRunner()


def _configure_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_API_KEY", "dummy")
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")


def _operation(**overrides: Any) -> Operation:
    payload: Dict[str, Any] = {
        "operation_id": "op-1",
        "kind": "create",
        "state": "succeeded",
        "result": {},
    }
    payload.update(overrides)
    return Operation.model_validate(payload)


def _environment(**overrides: Any) -> DevEnvironment:
    now_iso = "2026-10-14T10:00:00+00:00"
    base: Dict[str, Any] = dict(
        id="dev-env-1",
        state="READY",
        template="platform-dev",
        template_version="3",
        ref="origin/develop",
        resolved_sha="abc1234",
        ttl_minutes=120,
        team_id=None,
        actor_user_id="user-1",
        sandbox_id="sbx-1",
        urls={"frontend": "https://fe.example.com", "backend": "https://be.example.com"},
        tunnel_ids=["tun-1"],
        credential_ref=None,
        expiry="2026-10-14T12:00:00+00:00",
        last_error=None,
        operation_id="op-1",
        boot_seconds=143.0,
        ready_at=now_iso,
        created_at=now_iso,
        updated_at=now_iso,
    )
    base.update(overrides)
    return DevEnvironment.model_validate(base)


def test_up_polls_to_ready_and_prints_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_cli(monkeypatch)
    created = SimpleNamespace(environment_id="dev-env-1", operation_id="op-1")
    waited: List[str] = []

    def create(self: Any, template: str, ref: str, ttl_minutes: int = 120, team_id: Any = None):
        assert template == "platform-dev"
        assert ref == "origin/develop"
        assert ttl_minutes == 120
        assert team_id is None
        return created

    def wait_for_operation(
        self: Any, operation_id: str, poll_interval_seconds: Any = None, on_update: Any = None
    ):
        waited.append(operation_id)
        assert poll_interval_seconds == 5.0
        return _operation()

    monkeypatch.setattr(dev_sandbox.DevSandboxClient, "create", create)
    monkeypatch.setattr(dev_sandbox.DevSandboxClient, "wait_for_operation", wait_for_operation)
    monkeypatch.setattr(dev_sandbox.DevSandboxClient, "get", lambda self, env_id: _environment())

    result = runner.invoke(app, ["dev-sandbox", "up", "--ref", "origin/develop"])

    assert result.exit_code == 0, result.output
    assert waited == ["op-1"]
    assert "dev-env-1" in result.output
    assert "READY" in result.output
    assert "https://fe.example.com" in result.output


def test_up_json_output(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_cli(monkeypatch)
    created = SimpleNamespace(environment_id="dev-env-1", operation_id="op-1")

    monkeypatch.setattr(
        dev_sandbox.DevSandboxClient,
        "create",
        lambda self, template, ref, ttl_minutes=120, team_id=None: created,
    )
    monkeypatch.setattr(
        dev_sandbox.DevSandboxClient,
        "wait_for_operation",
        lambda self, operation_id, poll_interval_seconds=None, on_update=None: _operation(),
    )
    monkeypatch.setattr(dev_sandbox.DevSandboxClient, "get", lambda self, env_id: _environment())

    result = runner.invoke(app, ["dev-sandbox", "up", "--ref", "main", "--output", "json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["id"] == "dev-env-1"
    assert data["state"] == "READY"
    assert data["urls"]["frontend"] == "https://fe.example.com"
    assert data["operation_state"] == "succeeded"
    assert data["operation_id"] == "op-1"


def test_up_failed_operation_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_cli(monkeypatch)
    created = SimpleNamespace(environment_id="dev-env-1", operation_id="op-1")

    monkeypatch.setattr(
        dev_sandbox.DevSandboxClient,
        "create",
        lambda self, template, ref, ttl_minutes=120, team_id=None: created,
    )
    monkeypatch.setattr(
        dev_sandbox.DevSandboxClient,
        "wait_for_operation",
        lambda self, operation_id, poll_interval_seconds=None, on_update=None: _operation(
            state="failed",
            error={
                "code": "READY_TIMEOUT",
                "stage": "readiness",
                "message": "probes did not pass",
            },
        ),
    )

    result = runner.invoke(app, ["dev-sandbox", "up", "--ref", "main"])

    assert result.exit_code == 1
    assert "READY_TIMEOUT" in result.output


def test_up_rejects_ttl_over_max(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_cli(monkeypatch)
    result = runner.invoke(app, ["dev-sandbox", "up", "--ref", "main", "--ttl-minutes", "481"])
    assert result.exit_code == 1
    assert "480" in result.output


def test_up_requires_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_cli(monkeypatch)
    result = runner.invoke(app, ["dev-sandbox", "up"])
    assert result.exit_code != 0


def test_status_table_and_json(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_cli(monkeypatch)
    monkeypatch.setattr(dev_sandbox.DevSandboxClient, "get", lambda self, env_id: _environment())

    table = runner.invoke(app, ["dev-sandbox", "status", "dev-env-1"])
    assert table.exit_code == 0, table.output
    assert "READY" in table.output and "platform-dev" in table.output

    as_json = runner.invoke(app, ["dev-sandbox", "status", "dev-env-1", "--output", "json"])
    assert as_json.exit_code == 0, as_json.output
    data = json.loads(as_json.output)
    assert data["id"] == "dev-env-1"
    assert data["resolved_sha"] == "abc1234"


def test_list_table_and_json(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_cli(monkeypatch)

    def list_envs(self: Any, page: int = 1, per_page: int = 50):
        assert (page, per_page) == (1, 10)
        return SimpleNamespace(
            environments=[_environment(), _environment(id="dev-env-2", state="DELETING")],
            total=2,
            page=1,
            per_page=10,
            has_next=False,
        )

    monkeypatch.setattr(dev_sandbox.DevSandboxClient, "list", list_envs)

    table = runner.invoke(app, ["dev-sandbox", "list", "--num", "10"])
    assert table.exit_code == 0, table.output
    assert "dev-env-1" in table.output and "dev-env-2" in table.output

    as_json = runner.invoke(app, ["dev-sandbox", "list", "--num", "10", "--output", "json"])
    assert as_json.exit_code == 0, as_json.output
    data = json.loads(as_json.output)
    assert data["total"] == 2
    assert data["environments"][1]["state"] == "DELETING"


def test_exec_streams_diagnostics_and_exits_with_code(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_cli(monkeypatch)
    admitted = SimpleNamespace(operation_id="op-exec")

    captured: Dict[str, Any] = {}

    def do_exec(self: Any, env_id: str, command: str, timeout_seconds: int = 300):
        captured["args"] = (env_id, command, timeout_seconds)
        return admitted

    monkeypatch.setattr(dev_sandbox.DevSandboxClient, "exec", do_exec)

    def wait_for_operation(
        self: Any, operation_id: str, poll_interval_seconds: Any = None, on_update: Any = None
    ):
        assert poll_interval_seconds == 2.0
        operation = _operation(
            operation_id="op-exec",
            kind="exec",
            diagnostics=["dockerd up", "running make test"],
            result={"exit_code": 0, "stdout": "all tests passed"},
        )
        if on_update is not None:
            on_update(operation)
        return operation

    monkeypatch.setattr(dev_sandbox.DevSandboxClient, "wait_for_operation", wait_for_operation)

    result = runner.invoke(
        app,
        ["dev-sandbox", "exec", "dev-env-1", "--timeout-seconds", "60", "make", "test"],
    )

    assert result.exit_code == 0, result.output
    assert captured["args"] == ("dev-env-1", "make test", 60)
    assert "all tests passed" in result.output
    assert "running make test" in result.output


def test_exec_failed_command_propagates_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_cli(monkeypatch)
    monkeypatch.setattr(
        dev_sandbox.DevSandboxClient,
        "exec",
        lambda self, env_id, command, timeout_seconds=300: SimpleNamespace(operation_id="op-exec"),
    )
    monkeypatch.setattr(
        dev_sandbox.DevSandboxClient,
        "wait_for_operation",
        lambda self, operation_id, poll_interval_seconds=None, on_update=None: _operation(
            operation_id="op-exec",
            kind="exec",
            result={"exit_code": 2, "stdout": "", "stderr": "boom"},
        ),
    )

    result = runner.invoke(app, ["dev-sandbox", "exec", "dev-env-1", "false"])

    assert result.exit_code == 1
    assert "boom" in result.output


def test_exec_json_output(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_cli(monkeypatch)
    monkeypatch.setattr(
        dev_sandbox.DevSandboxClient,
        "exec",
        lambda self, env_id, command, timeout_seconds=300: SimpleNamespace(operation_id="op-exec"),
    )
    monkeypatch.setattr(
        dev_sandbox.DevSandboxClient,
        "wait_for_operation",
        lambda self, operation_id, poll_interval_seconds=None, on_update=None: _operation(
            operation_id="op-exec",
            kind="exec",
            result={"exit_code": 0, "stdout": "ok"},
        ),
    )

    result = runner.invoke(
        app, ["dev-sandbox", "exec", "dev-env-1", "--output", "json", "echo", "hi"]
    )

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["state"] == "succeeded"
    assert data["exit_code"] == 0
    assert data["result"]["stdout"] == "ok"


def test_down_deletes_and_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_cli(monkeypatch)
    deleted: List[str] = []

    monkeypatch.setattr(
        dev_sandbox.DevSandboxClient,
        "delete",
        lambda self, env_id: (deleted.append(env_id), SimpleNamespace(operation_id="op-del"))[1],
    )
    monkeypatch.setattr(
        dev_sandbox.DevSandboxClient,
        "wait_for_operation",
        lambda self, operation_id, poll_interval_seconds=None, on_update=None: _operation(
            operation_id="op-del",
            kind="delete",
            result={"state": "DELETED"},
        ),
    )

    result = runner.invoke(app, ["dev-sandbox", "down", "dev-env-1", "--yes"])

    assert result.exit_code == 0, result.output
    assert deleted == ["dev-env-1"]
    assert "DELETED" in result.output


def test_down_requires_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_cli(monkeypatch)
    result = runner.invoke(app, ["dev-sandbox", "down", "dev-env-1"], input="n\n")
    assert result.exit_code == 0
    # No delete call was made.
    assert "Deleting" not in result.output


def test_plain_mode_help_lists_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_cli(monkeypatch)
    result = runner.invoke(app, ["dev-sandbox", "--plain", "--help"])
    assert result.exit_code == 0, result.output
    for command in ("up", "list", "status", "exec", "down"):
        assert command in result.output
