"""Tests for `prime rl logs` (orchestrator) and `prime rl env-logs` (env-server)."""

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


def _make_mock_get(
    responses: Dict[str, Any],
    orchestrator_call_log: List[Dict[str, Any]],
    env_server_call_log: List[Dict[str, Any]],
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
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    return mock_get


# -------------------- `prime rl logs` (orchestrator) --------------------


def test_logs_fetches_orchestrator_only(monkeypatch: pytest.MonkeyPatch) -> None:
    orch_calls: List[Dict[str, Any]] = []
    env_calls: List[Dict[str, Any]] = []
    mock_get = _make_mock_get(
        {"orchestrator_logs": "orchestrator line one\norchestrator line two"},
        orch_calls,
        env_calls,
    )
    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["rl", "logs", RUN_ID, "--raw"])

    assert result.exit_code == 0, result.output
    assert "orchestrator line one" in result.output
    assert "orchestrator line two" in result.output
    # Orchestrator endpoint hit; env-server endpoint never called.
    assert len(orch_calls) == 1
    assert orch_calls[0]["tail_lines"] == 1000
    assert env_calls == []


def test_logs_rejects_unknown_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make sure `prime rl logs` has not grown env-specific flags."""

    def mock_get(self: Any, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        raise AssertionError(f"Should not be called: {endpoint}")

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    # --env is an env-logs flag; it must not silently be accepted by `logs`.
    result = CliRunner().invoke(app, ["rl", "logs", RUN_ID, "--env", "reverse-text"])
    assert result.exit_code != 0


# -------------------- `prime rl env-logs` --------------------


def test_env_logs_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    orch_calls: List[Dict[str, Any]] = []
    env_calls: List[Dict[str, Any]] = []
    mock_get = _make_mock_get(
        {"env_server_logs": "env-server crashed: ModuleNotFoundError: no module foo"},
        orch_calls,
        env_calls,
    )
    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(
        app,
        [
            "rl",
            "env-logs",
            RUN_ID,
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
    # Env-server endpoint hit with the right params; orchestrator untouched.
    assert orch_calls == []
    assert len(env_calls) == 1
    assert env_calls[0] == {
        "env_name": "reverse-text",
        "env_index": 2,
        "tail_lines": 50,
    }


def test_env_logs_default_index_is_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    orch_calls: List[Dict[str, Any]] = []
    env_calls: List[Dict[str, Any]] = []
    mock_get = _make_mock_get(
        {"env_server_logs": "line"},
        orch_calls,
        env_calls,
    )
    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(
        app,
        ["rl", "env-logs", RUN_ID, "--env", "opencode-math", "--raw"],
    )

    assert result.exit_code == 0, result.output
    assert len(env_calls) == 1
    assert env_calls[0]["env_index"] == 0
    assert env_calls[0]["tail_lines"] == 1000


def test_env_logs_requires_env_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_get(self: Any, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        raise AssertionError(f"Should not be called: {endpoint}")

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["rl", "env-logs", RUN_ID])

    assert result.exit_code != 0, result.output


def test_env_logs_run_not_started(monkeypatch: pytest.MonkeyPatch) -> None:
    """Queued runs return a 404 with a 'queued'/'pending' marker; CLI should
    exit 0 with a friendly message, matching `prime rl logs` behavior."""

    def mock_get(self: Any, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        raise APIError("Failed to get env server logs: 404 run is queued")

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["rl", "env-logs", RUN_ID, "--env", "reverse-text", "--raw"])

    assert result.exit_code == 0, result.output
    assert "has not started yet" in result.output


def test_env_logs_run_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_get(self: Any, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        raise APIError("Failed to get env server logs: run not found")

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["rl", "env-logs", RUN_ID, "--env", "reverse-text", "--raw"])

    assert result.exit_code == 1, result.output
    # Rich may wrap lines; collapse whitespace for the substring check.
    collapsed = " ".join(result.output.lower().split())
    assert "run not found" in collapsed


def test_env_logs_follow_loops_and_dedupes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Follow-mode should re-poll and avoid reprinting overlapping tail lines."""
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
        # Break the loop on the third iteration.
        raise KeyboardInterrupt

    monkeypatch.setattr(RLClient, "get_env_server_logs", mock_get_env_server_logs)

    result = CliRunner().invoke(
        app,
        ["rl", "env-logs", RUN_ID, "--env", "reverse-text", "--follow", "--raw"],
    )

    assert result.exit_code == 0, result.output
    # line-1, line-2, line-3 each appear exactly once.
    assert result.output.count("line-1") == 1
    assert result.output.count("line-2") == 1
    assert result.output.count("line-3") == 1
    assert call_count["n"] >= 2
