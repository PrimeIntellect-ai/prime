"""Tests for `prime rl logs` (orchestrator + env-server) and `prime rl components`."""

from datetime import datetime, timezone
from typing import Any, Dict, List

import pytest
from prime_cli.api.rl import RLClient
from prime_cli.client import APIError
from prime_cli.main import app
from typer.testing import CliRunner

RUN_ID = "rl-run-123"


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_API_KEY", "dummy")
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("prime_cli.commands.rl.time.sleep", lambda _: None)


def _run_payload(status: str = "RUNNING") -> Dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": RUN_ID,
        "name": "demo",
        "userId": "u1",
        "status": status,
        "rolloutsPerExample": 8,
        "seqLen": 4096,
        "maxSteps": 100,
        "batchSize": 32,
        "baseModel": "some-model",
        "environments": [{"name": "reverse-text"}],
        "createdAt": now,
        "updatedAt": now,
    }


def _make_mock_get(
    responses: Dict[str, Any],
    orch_calls: List[Dict[str, Any]],
    env_calls: List[Dict[str, Any]],
    list_calls: List[Dict[str, Any]],
):
    def mock_get(self: Any, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if endpoint == f"/rft/runs/{RUN_ID}":
            return {"run": _run_payload(responses.get("run_status", "RUNNING"))}
        if endpoint == f"/rft/runs/{RUN_ID}/logs":
            orch_calls.append(params or {})
            return {"logs": responses.get("orchestrator_logs", "")}
        if endpoint == f"/rft/runs/{RUN_ID}/env-server-logs":
            env_calls.append(params or {})
            return {"logs": responses.get("env_server_logs", "")}
        if endpoint == f"/rft/runs/{RUN_ID}/env-servers":
            list_calls.append(params or {})
            return {"env_servers": responses.get("env_servers", [])}
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    return mock_get


# ---------- prime rl logs (orchestrator default) ----------


def test_logs_default_hits_orchestrator(monkeypatch: pytest.MonkeyPatch) -> None:
    orch: List[Dict[str, Any]] = []
    env: List[Dict[str, Any]] = []
    lst: List[Dict[str, Any]] = []
    mock_get = _make_mock_get({"orchestrator_logs": "orch-line-1\norch-line-2"}, orch, env, lst)
    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["rl", "logs", RUN_ID, "--raw"])

    assert result.exit_code == 0, result.output
    assert "orch-line-1" in result.output
    assert "orch-line-2" in result.output
    assert len(orch) == 1
    assert orch[0]["tail_lines"] == 1000
    assert env == []


def test_logs_env_flag_alone_infers_env_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """Passing --env without -c should infer component=env-server."""
    orch: List[Dict[str, Any]] = []
    env: List[Dict[str, Any]] = []
    lst: List[Dict[str, Any]] = []
    mock_get = _make_mock_get({"env_server_logs": "inferred line"}, orch, env, lst)
    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["rl", "logs", RUN_ID, "--env", "reverse-text", "--raw"])

    assert result.exit_code == 0, result.output
    assert "inferred line" in result.output
    assert orch == []
    assert len(env) == 1
    assert env[0]["env_name"] == "reverse-text"
    assert env[0]["env_index"] == 0


def test_logs_explicit_orchestrator_with_env_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit `-c orchestrator --env x` is a real conflict; must error."""

    def mock_get(self: Any, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        raise AssertionError(f"Should not be called: {endpoint}")

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(
        app, ["rl", "logs", RUN_ID, "-c", "orchestrator", "--env", "reverse-text"]
    )
    assert result.exit_code != 0


# ---------- prime rl logs -c env-server ----------


def test_logs_env_server_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    orch: List[Dict[str, Any]] = []
    env: List[Dict[str, Any]] = []
    lst: List[Dict[str, Any]] = []
    mock_get = _make_mock_get(
        {"env_server_logs": "ModuleNotFoundError: no module foo"}, orch, env, lst
    )
    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(
        app,
        [
            "rl",
            "logs",
            RUN_ID,
            "-c",
            "env-server",
            "--env",
            "reverse-text",
            "--tail",
            "50",
            "--raw",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "ModuleNotFoundError" in result.output
    assert orch == []
    assert len(env) == 1
    assert env[0] == {
        "env_name": "reverse-text",
        "env_index": 0,
        "tail_lines": 50,
    }


def test_logs_env_server_qualified_name_parses_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--env reverse-text/2` should split into env_name='reverse-text', env_index=2."""
    orch: List[Dict[str, Any]] = []
    env: List[Dict[str, Any]] = []
    lst: List[Dict[str, Any]] = []
    mock_get = _make_mock_get({"env_server_logs": "line"}, orch, env, lst)
    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(
        app,
        ["rl", "logs", RUN_ID, "-c", "env-server", "--env", "reverse-text/2", "--raw"],
    )

    assert result.exit_code == 0, result.output
    assert env[0]["env_name"] == "reverse-text"
    assert env[0]["env_index"] == 2


def test_logs_env_server_owner_name_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--env primeintellect/reverse-text` is a valid name, not a qualifier.

    Only a trailing ``/<int>`` should be treated as a replica index.
    """
    orch: List[Dict[str, Any]] = []
    env: List[Dict[str, Any]] = []
    lst: List[Dict[str, Any]] = []
    mock_get = _make_mock_get({"env_server_logs": "ok"}, orch, env, lst)
    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(
        app,
        ["rl", "logs", RUN_ID, "--env", "primeintellect/reverse-text", "--raw"],
    )

    assert result.exit_code == 0, result.output
    assert env[0]["env_name"] == "primeintellect/reverse-text"
    assert env[0]["env_index"] == 0


def test_logs_env_server_owner_name_with_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--env primeintellect/reverse-text/1` should split at the last slash."""
    orch: List[Dict[str, Any]] = []
    env: List[Dict[str, Any]] = []
    lst: List[Dict[str, Any]] = []
    mock_get = _make_mock_get({"env_server_logs": "ok"}, orch, env, lst)
    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(
        app,
        ["rl", "logs", RUN_ID, "--env", "primeintellect/reverse-text/1", "--raw"],
    )

    assert result.exit_code == 0, result.output
    assert env[0]["env_name"] == "primeintellect/reverse-text"
    assert env[0]["env_index"] == 1


def test_logs_env_server_requires_env_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_get(self: Any, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        raise AssertionError(f"Should not be called: {endpoint}")

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["rl", "logs", RUN_ID, "-c", "env-server"])
    assert result.exit_code != 0
    assert "--env" in result.output or "env" in result.output.lower()


def test_logs_invalid_component(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_get(self: Any, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        raise AssertionError(f"Should not be called: {endpoint}")

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["rl", "logs", RUN_ID, "-c", "trainer"])
    assert result.exit_code != 0


def test_logs_run_not_started(monkeypatch: pytest.MonkeyPatch) -> None:
    """Queued runs return 404 with 'queued'; CLI should exit 0 with a friendly message."""

    def mock_get(self: Any, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        raise APIError("Failed to get RL run logs: 404 run is queued")

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["rl", "logs", RUN_ID, "--raw"])

    assert result.exit_code == 0, result.output
    assert "has not started yet" in result.output


def test_logs_env_server_follow_dedupes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Follow-mode polls and avoids reprinting overlapping tail lines."""
    call_count = {"n": 0}

    def mock_get_env_server_logs(
        self: Any,
        run_id: str,
        env_name: str,
        env_index: int = 0,
        tail_lines: int = 1000,
    ) -> str:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return "line-1\nline-2"
        if call_count["n"] == 2:
            return "line-2\nline-3"
        raise KeyboardInterrupt

    monkeypatch.setattr(RLClient, "get_env_server_logs", mock_get_env_server_logs)

    result = CliRunner().invoke(
        app,
        [
            "rl",
            "logs",
            RUN_ID,
            "-c",
            "env-server",
            "--env",
            "reverse-text",
            "--follow",
            "--raw",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.output.count("line-1") == 1
    assert result.output.count("line-2") == 1
    assert result.output.count("line-3") == 1
    assert call_count["n"] >= 2


# ---------- prime rl components ----------


def test_components_lists_orchestrator_and_env_servers(monkeypatch: pytest.MonkeyPatch) -> None:
    orch: List[Dict[str, Any]] = []
    env: List[Dict[str, Any]] = []
    lst: List[Dict[str, Any]] = []
    mock_get = _make_mock_get(
        {
            "run_status": "RUNNING",
            "env_servers": [
                {
                    "env_name": "reverse-text",
                    "env_index": 0,
                    "pod_name": "rft-rl-run-123-env-reverse-text-0",
                    "status": "Running",
                },
                {
                    "env_name": "opencode-math",
                    "env_index": 0,
                    "pod_name": "rft-rl-run-123-env-opencode-math-0",
                    "status": "CrashLoopBackOff",
                },
            ],
        },
        orch,
        env,
        lst,
    )
    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["rl", "components", RUN_ID])

    assert result.exit_code == 0, result.output
    out = " ".join(result.output.split())
    assert "orchestrator" in out
    assert "RUNNING" in out
    assert "reverse-text" in out
    assert "opencode-math" in out
    assert "CrashLoopBackOff" in out
    # No qualifier when env names are unique.
    assert "reverse-text/" not in out
    assert "opencode-math/" not in out
    assert len(lst) == 1


def test_components_qualifies_duplicate_env_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the same env name appears twice, the listing must show 'name/N'."""
    orch: List[Dict[str, Any]] = []
    env: List[Dict[str, Any]] = []
    lst: List[Dict[str, Any]] = []
    mock_get = _make_mock_get(
        {
            "run_status": "RUNNING",
            "env_servers": [
                {
                    "env_name": "reverse-text",
                    "env_index": 0,
                    "pod_name": "rft-x-env-reverse-text-0",
                    "status": "Running",
                },
                {
                    "env_name": "reverse-text",
                    "env_index": 1,
                    "pod_name": "rft-x-env-reverse-text-1",
                    "status": "Running",
                },
            ],
        },
        orch,
        env,
        lst,
    )
    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["rl", "components", RUN_ID])

    assert result.exit_code == 0, result.output
    out = " ".join(result.output.split())
    assert "reverse-text/0" in out
    assert "reverse-text/1" in out


def test_components_empty_env_servers(monkeypatch: pytest.MonkeyPatch) -> None:
    orch: List[Dict[str, Any]] = []
    env: List[Dict[str, Any]] = []
    lst: List[Dict[str, Any]] = []
    mock_get = _make_mock_get(
        {"run_status": "QUEUED", "env_servers": []},
        orch,
        env,
        lst,
    )
    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["rl", "components", RUN_ID])

    assert result.exit_code == 0, result.output
    assert "orchestrator" in result.output
    assert "QUEUED" in result.output
