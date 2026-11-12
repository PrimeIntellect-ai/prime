"""Tests for `prime factory slurm resize` command."""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
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
    """GET returns the cluster-list payloads in order, repeating the last one."""

    def __init__(
        self,
        clusters_payloads: List[Dict[str, Any]],
        resize_response: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._clusters_payloads = clusters_payloads
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
        gets = sum(1 for c in self.calls if c["method"] == "GET")
        return self._clusters_payloads[min(gets, len(self._clusters_payloads)) - 1]

    def post(
        self,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        timeout: Any = None,
    ) -> Dict[str, Any]:
        self.calls.append({"method": "POST", "endpoint": endpoint, "json": json})
        return self._resize_response

    def delete(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {}


def _install(
    monkeypatch: pytest.MonkeyPatch,
    clusters_payload: Union[Dict[str, Any], List[Dict[str, Any]]],
    team_id: Optional[str] = "team-123",
    resize_response: Optional[Dict[str, Any]] = None,
) -> _DummyAPIClient:
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)
    payloads = clusters_payload if isinstance(clusters_payload, list) else [clusters_payload]
    dummy = _DummyAPIClient(payloads, resize_response=resize_response)
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
    _install(monkeypatch, _clusters_payload(gpuCount=32))

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


RESIZE_ARGS = ["factory", "slurm", "resize", "job-123", "--nodes", "4"]


def test_slurm_resize_json_with_yes_keeps_stdout_pure(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _clusters_payload())

    result = runner.invoke(app, [*RESIZE_ARGS, "--yes", "--json"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {
        "numNodes": 4,
        "added": ["node-3", "node-4"],
        "removed": [],
    }


def test_slurm_resize_json_prompt_goes_to_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _clusters_payload())

    result = runner.invoke(app, [*RESIZE_ARGS, "--json"], input="y\n", env=TEST_ENV)

    assert result.exit_code == 0, result.output
    # click.confirm reads via input(" ") and CliRunner echoes the typed answer
    # to stdout; a real terminal echoes it on the tty, leaving only the space.
    assert json.loads(result.stdout.removeprefix(" y\n"))["numNodes"] == 4
    stderr = strip_ansi(result.stderr)
    assert "Current size: 32 GPUs" in stderr
    assert "Proceed with resize?" in stderr


def test_slurm_resize_json_declined_prints_nothing_to_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy = _install(monkeypatch, _clusters_payload())

    result = runner.invoke(app, [*RESIZE_ARGS, "--json"], input="n\n", env=TEST_ENV)

    assert result.exit_code == 0
    assert result.stdout.removeprefix(" n\n") == ""
    assert "Resize cancelled." in result.stderr
    assert not [c for c in dummy.calls if c["method"] == "POST"]


def test_slurm_resize_json_cluster_not_found_keeps_stdout_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _clusters_payload())

    result = runner.invoke(
        app,
        ["factory", "slurm", "resize", "nonexistent-job", "--nodes", "4", "--yes", "--json"],
        env=TEST_ENV,
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "No Slurm deployment matched" in strip_ansi(result.stderr)


def test_slurm_resize_handles_403_error_with_context(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _clusters_payload())

    with patch.object(FactorySlurmClient, "resize_cluster") as mock_resize:
        mock_resize.side_effect = APIError("HTTP 403: Forbidden")

        result = runner.invoke(
            app, ["factory", "slurm", "resize", "job-123", "--nodes", "4", "--yes"], env=TEST_ENV
        )
        output = strip_ansi(result.output)

        assert result.exit_code == 1
        assert "factory:write" in output
        assert "team admin" in output
        assert "self-service" in output


def test_slurm_resize_lookup_403_mentions_read_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _clusters_payload())

    with patch.object(FactorySlurmClient, "list_clusters") as mock_list:
        mock_list.side_effect = APIError("HTTP 403: Forbidden")

        result = runner.invoke(app, [*RESIZE_ARGS, "--yes"], env=TEST_ENV)

    assert result.exit_code == 1
    assert "slurm_clusters:read" in strip_ansi(result.output)


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


# 2 nodes x 8 GPUs before; the default resize response adds 2 nodes -> 32 GPUs.
def _sized(gpu_count: int) -> Dict[str, Any]:
    return _clusters_payload(gpuCount=gpu_count)


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> List[float]:
    calls: List[float] = []
    monkeypatch.setattr("prime_cli.commands.factory_slurm.time.sleep", calls.append)
    return calls


def test_slurm_resize_wait_polls_until_target_gpu_count(
    monkeypatch: pytest.MonkeyPatch, sleeps: List[float]
) -> None:
    # Pre-flight lookup, then polls; "running" alone must not end the wait.
    dummy = _install(monkeypatch, [_sized(16), _sized(16), _sized(24), _sized(32)])

    result = runner.invoke(app, [*RESIZE_ARGS, "--yes", "--wait"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "Waiting for the cluster to reach 32 GPUs" in output
    assert "GPUs: 24/32" in output
    assert "Resize complete: 32 GPUs" in output
    assert len([c for c in dummy.calls if c["method"] == "GET"]) == 4
    assert len(sleeps) == 2


def test_slurm_resize_wait_handles_scale_down(
    monkeypatch: pytest.MonkeyPatch, sleeps: List[float]
) -> None:
    resize_response = {"numNodes": 2, "added": [], "removed": ["node-3", "node-4"]}
    _install(monkeypatch, [_sized(32), _sized(32), _sized(16)], resize_response=resize_response)

    result = runner.invoke(
        app,
        ["factory", "slurm", "resize", "job-123", "--nodes", "2", "--yes", "--wait"],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    assert "Resize complete: 16 GPUs" in strip_ansi(result.output)


def test_slurm_resize_wait_timeout_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, sleeps: List[float]
) -> None:
    _install(monkeypatch, _sized(16))

    result = runner.invoke(app, [*RESIZE_ARGS, "--yes", "--wait"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 1
    assert "Timeout:" in output
    assert "Resize complete" not in output
    assert sum(sleeps) == 300


def test_slurm_resize_wait_rejects_underivable_target(
    monkeypatch: pytest.MonkeyPatch, sleeps: List[float]
) -> None:
    _install(monkeypatch, _sized(15))

    result = runner.invoke(app, [*RESIZE_ARGS, "--yes", "--wait"], env=TEST_ENV)

    assert result.exit_code == 1
    assert "Cannot derive the target GPU count" in strip_ansi(result.output)
    assert sleeps == []


def test_slurm_resize_json_wait_prints_response_after_wait(
    monkeypatch: pytest.MonkeyPatch, sleeps: List[float]
) -> None:
    _install(monkeypatch, [_sized(16), _sized(24), _sized(32)])

    result = runner.invoke(app, [*RESIZE_ARGS, "--yes", "--wait", "--json"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["numNodes"] == 4
    assert "Resize complete: 32 GPUs" in strip_ansi(result.stderr)


def test_slurm_resize_json_wait_timeout_keeps_stdout_empty(
    monkeypatch: pytest.MonkeyPatch, sleeps: List[float]
) -> None:
    _install(monkeypatch, _sized(16))

    result = runner.invoke(app, [*RESIZE_ARGS, "--yes", "--wait", "--json"], env=TEST_ENV)

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Timeout:" in strip_ansi(result.stderr)
