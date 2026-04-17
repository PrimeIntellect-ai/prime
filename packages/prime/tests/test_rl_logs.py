"""Tests for `prime rl logs` (orchestrator and env-server components)."""

from typing import Any, Dict, List

import pytest
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


def _make_mock_get(
    responses: Dict[str, Any],
    orchestrator_call_log: List[Dict[str, Any]],
    env_server_call_log: List[Dict[str, Any]],
    env_servers_list_log: List[int],
):
    def mock_get(self: Any, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if endpoint == f"/rft/runs/{RUN_ID}/logs":
            orchestrator_call_log.append(params or {})
            return {"logs": responses.get("orchestrator_logs", "")}
        if endpoint == f"/rft/runs/{RUN_ID}/env-server-logs":
            env_server_call_log.append(params or {})
            return {
                "logs": responses.get("env_server_logs", ""),
                "env_name": (params or {}).get("env_name"),
                "env_index": (params or {}).get("env_index", 0),
            }
        if endpoint == f"/rft/runs/{RUN_ID}/env-servers":
            env_servers_list_log.append(1)
            return {"env_servers": responses.get("env_servers", [])}
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    return mock_get


def test_logs_defaults_to_orchestrator(monkeypatch: pytest.MonkeyPatch) -> None:
    orch_calls: List[Dict[str, Any]] = []
    env_calls: List[Dict[str, Any]] = []
    list_calls: List[int] = []
    mock_get = _make_mock_get(
        {"orchestrator_logs": "orchestrator line one\norchestrator line two"},
        orch_calls,
        env_calls,
        list_calls,
    )
    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["rl", "logs", RUN_ID, "--raw"])

    assert result.exit_code == 0, result.output
    assert "orchestrator line one" in result.output
    assert "orchestrator line two" in result.output
    # Orchestrator endpoint hit, env-server endpoints untouched.
    assert len(orch_calls) == 1
    assert orch_calls[0]["tail_lines"] == 1000
    assert env_calls == []
    assert list_calls == []


def test_logs_env_server_with_explicit_env(monkeypatch: pytest.MonkeyPatch) -> None:
    orch_calls: List[Dict[str, Any]] = []
    env_calls: List[Dict[str, Any]] = []
    list_calls: List[int] = []
    mock_get = _make_mock_get(
        {"env_server_logs": "env-server crashed: ModuleNotFoundError: no module foo"},
        orch_calls,
        env_calls,
        list_calls,
    )
    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(
        app,
        [
            "rl",
            "logs",
            RUN_ID,
            "--component",
            "env-server",
            "--env",
            "reverse-text",
            "--index",
            "2",
            "--tail",
            "50",
            "--raw",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "ModuleNotFoundError" in result.output
    # Env-server endpoint hit with the right params; orchestrator + list untouched.
    assert orch_calls == []
    assert list_calls == []
    assert len(env_calls) == 1
    assert env_calls[0] == {
        "env_name": "reverse-text",
        "env_index": 2,
        "tail_lines": 50,
    }


def test_logs_env_server_autoselects_single_env(monkeypatch: pytest.MonkeyPatch) -> None:
    orch_calls: List[Dict[str, Any]] = []
    env_calls: List[Dict[str, Any]] = []
    list_calls: List[int] = []
    mock_get = _make_mock_get(
        {
            "env_servers": [
                {
                    "env_name": "reverse-text",
                    "env_index": "0",
                    "pod_name": "rft-env-reverse-text-0",
                    "status": "Running",
                }
            ],
            "env_server_logs": "only-env-line",
        },
        orch_calls,
        env_calls,
        list_calls,
    )
    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["rl", "logs", RUN_ID, "--component", "env-server", "--raw"])

    assert result.exit_code == 0, result.output
    assert "only-env-line" in result.output
    assert list_calls == [1]
    assert len(env_calls) == 1
    assert env_calls[0]["env_name"] == "reverse-text"
    assert env_calls[0]["env_index"] == 0


def test_logs_env_server_lists_when_ambiguous(monkeypatch: pytest.MonkeyPatch) -> None:
    orch_calls: List[Dict[str, Any]] = []
    env_calls: List[Dict[str, Any]] = []
    list_calls: List[int] = []
    mock_get = _make_mock_get(
        {
            "env_servers": [
                {
                    "env_name": "reverse-text",
                    "env_index": "0",
                    "pod_name": "rft-env-reverse-text-0",
                    "status": "Running",
                },
                {
                    "env_name": "opencode-math",
                    "env_index": "0",
                    "pod_name": "rft-env-opencode-math-0",
                    "status": "CrashLoopBackOff",
                },
            ]
        },
        orch_calls,
        env_calls,
        list_calls,
    )
    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(
        app,
        ["rl", "logs", RUN_ID, "--component", "env-server"],
        env={"COLUMNS": "200"},
    )

    assert result.exit_code == 1, result.output
    # Both env names appear in the listing; no logs were fetched.
    assert "reverse-text" in result.output
    assert "opencode-math" in result.output
    assert env_calls == []


def test_logs_rejects_invalid_component(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_get(self: Any, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        raise AssertionError(f"Should not be called: {endpoint}")

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["rl", "logs", RUN_ID, "--component", "worker"])

    assert result.exit_code == 1, result.output
    assert "--component" in result.output
