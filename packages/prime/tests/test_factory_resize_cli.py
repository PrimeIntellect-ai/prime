"""Tests for `prime factory slurm resize` command."""

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from unittest.mock import patch

import pytest
from prime_cli.api.factory_slurm import FactorySlurmClient
from prime_cli.client import APIError
from prime_cli.main import app
from prime_cli.utils.formatters import strip_ansi
from typer.testing import CliRunner

runner = CliRunner()

TEST_ENV = {
    "PRIME_API_KEY": "dummy",
    "PRIME_DISABLE_VERSION_CHECK": "1",
    "COLUMNS": "220",
    "TERM": "xterm",
}


def _cluster_row(
    cluster_id: str = "job-123",
    display_name: str = "research-cluster",
    gpu_count: int = 32,
    status: str = "running",
) -> Dict[str, Any]:
    return {
        "id": cluster_id,
        "primeClusterId": "cluster-abc",
        "displayName": display_name,
        "status": status,
        "gpuType": "H100",
        "gpuCount": gpu_count,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "startedAt": datetime.now(timezone.utc).isoformat(),
    }


def _clusters_payload(**overrides) -> Dict[str, Any]:
    cluster = _cluster_row()
    cluster.update(overrides)
    return {"data": [cluster]}


class _StubConfig:
    def __init__(self, team_id: Optional[str]) -> None:
        self._team_id = team_id

    @property
    def team_id(self) -> Optional[str]:
        return self._team_id


class _DummyAPIClient:
    def __init__(
        self,
        clusters_payload: Dict[str, Any],
        resize_response: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._clusters_payload = clusters_payload
        self._resize_response = resize_response or {
            "numNodes": 4,
            "added": ["node-3", "node-4"],
            "removed": [],
        }
        self.calls: list[Dict[str, Any]] = []

    def get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None, timeout: Any = None
    ) -> Dict[str, Any]:
        self.calls.append({"method": "GET", "endpoint": endpoint, "params": params})
        return self._clusters_payload

    def post(
        self,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        timeout: Any = None,
    ) -> Dict[str, Any]:
        self.calls.append({"method": "POST", "endpoint": endpoint, "json": json})
        return self._resize_response

    def request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: Any = None,
    ) -> Dict[str, Any]:
        self.calls.append({"method": method, "endpoint": endpoint, "json": json, "params": params})
        if method == "POST":
            return self._resize_response
        return self._clusters_payload

    def delete(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {}


def _install(
    monkeypatch: pytest.MonkeyPatch,
    clusters_payload: Dict[str, Any],
    team_id: Optional[str] = "team-123",
    resize_response: Optional[Dict[str, Any]] = None,
) -> _DummyAPIClient:
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)
    dummy = _DummyAPIClient(clusters_payload, resize_response=resize_response)
    monkeypatch.setattr("prime_cli.commands.factory_slurm.APIClient", lambda: dummy)
    monkeypatch.setattr("prime_cli.commands.factory_slurm.Config", lambda: _StubConfig(team_id))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    return dummy


def test_slurm_resize_requires_cluster_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _clusters_payload())

    result = runner.invoke(app, ["factory", "slurm", "resize", "--nodes", "4"], env=TEST_ENV)

    assert result.exit_code != 0
    assert "Missing argument" in result.output or "CLUSTER_ID" in result.output


def test_slurm_resize_requires_nodes_option(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _clusters_payload())

    result = runner.invoke(app, ["factory", "slurm", "resize", "job-123"], env=TEST_ENV)

    assert result.exit_code != 0
    assert "--nodes" in result.output or "Missing option" in result.output


def test_slurm_resize_with_confirmation_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = _install(monkeypatch, _clusters_payload())

    result = runner.invoke(
        app, ["factory", "slurm", "resize", "job-123", "--nodes", "4", "--yes"], env=TEST_ENV
    )
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "Resize initiated" in output
    assert "research-cluster" in output
    # Verify the API was called
    post_calls = [c for c in dummy.calls if c["method"] == "POST"]
    assert len(post_calls) == 1
    assert "/slurm-clusters/team-123/job-123/resize" in post_calls[0]["endpoint"]
    assert post_calls[0]["json"]["numNodes"] == 4


def test_slurm_resize_shows_confirmation_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _clusters_payload(gpu_count=32))

    # Simulate user declining
    result = runner.invoke(
        app, ["factory", "slurm", "resize", "job-123", "--nodes", "4"], input="n\n", env=TEST_ENV
    )
    output = strip_ansi(result.output)

    assert result.exit_code == 0
    assert "Resize Slurm cluster" in output
    assert "research-cluster" in output
    assert "Current size: 32 GPUs" in output
    assert "New nodes:    4 nodes" in output
    assert "cancelled" in output.lower()


def test_slurm_resize_shows_added_removed_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    resize_response = {
        "numNodes": 4,
        "added": ["node-3", "node-4"],
        "removed": [],
    }
    _install(monkeypatch, _clusters_payload(), resize_response=resize_response)

    result = runner.invoke(
        app, ["factory", "slurm", "resize", "job-123", "--nodes", "4", "--yes"], env=TEST_ENV
    )
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "node-3" in output
    assert "node-4" in output
    assert "Nodes to add:" in output


def test_slurm_resize_scale_down_shows_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    resize_response = {
        "numNodes": 2,
        "added": [],
        "removed": ["node-3", "node-4"],
    }
    _install(monkeypatch, _clusters_payload(), resize_response=resize_response)

    result = runner.invoke(
        app, ["factory", "slurm", "resize", "job-123", "--nodes", "2", "--yes"], env=TEST_ENV
    )
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "node-3" in output
    assert "node-4" in output
    assert "Nodes to remove:" in output


def test_slurm_resize_rejects_zero_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _clusters_payload())

    result = runner.invoke(
        app, ["factory", "slurm", "resize", "job-123", "--nodes", "0", "--yes"], env=TEST_ENV
    )
    output = strip_ansi(result.output)

    assert result.exit_code == 1
    assert "must be a positive integer" in output


def test_slurm_resize_rejects_negative_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _clusters_payload())

    result = runner.invoke(
        app, ["factory", "slurm", "resize", "job-123", "--nodes", "-10", "--yes"], env=TEST_ENV
    )
    output = strip_ansi(result.output)

    assert result.exit_code == 1
    assert "must be a positive integer" in output


def test_slurm_resize_json_output(monkeypatch: pytest.MonkeyPatch) -> None:
    resize_response = {
        "numNodes": 4,
        "added": ["node-3", "node-4"],
        "removed": [],
    }
    _install(monkeypatch, _clusters_payload(), resize_response=resize_response)

    result = runner.invoke(
        app,
        ["factory", "slurm", "resize", "job-123", "--nodes", "4", "--yes", "--json"],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    response = json.loads(result.output)
    assert response["numNodes"] == 4
    assert response["added"] == ["node-3", "node-4"]
    assert response["removed"] == []


def test_slurm_resize_handles_403_error_with_context(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _clusters_payload())

    with patch.object(FactorySlurmClient, "resize_cluster") as mock_resize:
        mock_resize.side_effect = APIError("HTTP 403: Forbidden")

        result = runner.invoke(
            app, ["factory", "slurm", "resize", "job-123", "--nodes", "4", "--yes"], env=TEST_ENV
        )
        output = strip_ansi(result.output)

        assert result.exit_code == 1
        assert "team admin" in output or "self-service" in output


def test_slurm_resize_handles_503_error_with_retry_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _clusters_payload())

    with patch.object(FactorySlurmClient, "resize_cluster") as mock_resize:
        mock_resize.side_effect = APIError("HTTP 503: Service Unavailable")

        result = runner.invoke(
            app, ["factory", "slurm", "resize", "job-123", "--nodes", "4", "--yes"], env=TEST_ENV
        )
        output = strip_ansi(result.output)

        assert result.exit_code == 1
        assert "retry" in output.lower()
        assert "Slurm state unavailable" in output


def test_slurm_resize_handles_409_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _clusters_payload())

    with patch.object(FactorySlurmClient, "resize_cluster") as mock_resize:
        mock_resize.side_effect = APIError("HTTP 409: Conflict")

        result = runner.invoke(
            app, ["factory", "slurm", "resize", "job-123", "--nodes", "4", "--yes"], env=TEST_ENV
        )
        output = strip_ansi(result.output)

        assert result.exit_code == 1
        assert "another operation is in progress" in output
        assert "insufficient idle nodes" in output


def test_slurm_resize_no_team_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _clusters_payload(), team_id=None)

    result = runner.invoke(
        app, ["factory", "slurm", "resize", "job-123", "--nodes", "4", "--yes"], env=TEST_ENV
    )
    output = strip_ansi(result.output)

    assert result.exit_code == 0  # Exits cleanly with message
    assert "No team selected" in output


def test_slurm_resize_cluster_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _clusters_payload())

    result = runner.invoke(
        app,
        ["factory", "slurm", "resize", "nonexistent-job", "--nodes", "4", "--yes"],
        env=TEST_ENV,
    )
    output = strip_ansi(result.output)

    assert result.exit_code == 1
    assert "No Slurm deployment matched" in output or "nonexistent-job" in output


def test_slurm_resize_with_wait_completes(monkeypatch: pytest.MonkeyPatch) -> None:
    # Start with one state
    initial_payload = _clusters_payload(status="scaling")
    # After resize, return running status
    updated_payload = _clusters_payload(status="running")

    dummy = _install(monkeypatch, initial_payload)

    # Mock time.sleep to avoid actual waiting
    with patch("time.sleep"):
        # Replace the get method to return updated status after first call
        call_count = [0]

        def mock_get(endpoint: str, params: Any = None, timeout: Any = None) -> Dict[str, Any]:
            dummy.calls.append({"method": "GET", "endpoint": endpoint, "params": params})
            call_count[0] += 1
            # First two calls return initial state, then return updated
            return initial_payload if call_count[0] <= 2 else updated_payload

        dummy.get = mock_get

        result = runner.invoke(
            app,
            ["factory", "slurm", "resize", "job-123", "--nodes", "4", "--yes", "--wait"],
            env=TEST_ENV,
        )
        output = strip_ansi(result.output)

        assert result.exit_code == 0, result.output
        assert "Waiting for resize" in output or "Status:" in output
