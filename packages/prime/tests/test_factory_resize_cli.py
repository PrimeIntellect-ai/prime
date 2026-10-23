"""Tests for `prime factory resize` command."""

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest
from prime_cli.api.factory import FactoryClient
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


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _source(kind: str, status: str = "ok", age_seconds: int = 12) -> Dict[str, Any]:
    return {
        "kind": kind,
        "status": status,
        "observed_at": _iso(datetime.now(timezone.utc)),
    }


def _pool(
    type_: str,
    reserved: Optional[int],
    in_use: Optional[int],
    idle_inside: Optional[int],
    unknown: Optional[int],
) -> Dict[str, Any]:
    return {
        "type": type_,
        "reserved_gpus": reserved,
        "in_use_gpus": in_use,
        "idle_inside_gpus": idle_inside,
        "unknown_gpus": unknown,
    }


def _status_payload(total_gpus: int = 128, **cluster_overrides: Any) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    cluster: Dict[str, Any] = {
        "display_name": "research-b300",
        "gpu_type": "B300",
        "total_gpus": total_gpus,
        "status": "online",
        "unassigned_gpus": 16,
        "unknown_gpus": 0,
        "pools": [
            _pool("training", 32, 32, 0, 0),
            _pool("inference", 32, 32, 0, 0),
            _pool("slurm", 48, 16, 32, 0),
        ],
        "sources": [
            _source("capacity", age_seconds=12),
            _source("training", age_seconds=10),
            _source("inference", age_seconds=10),
            _source("slurm", age_seconds=3),
        ],
    }
    cluster.update(cluster_overrides)
    return {
        "schema_version": 1,
        "as_of": _iso(now),
        "clusters": [cluster],
    }


class _StubConfig:
    def __init__(self, team_id: Optional[str]) -> None:
        self._team_id = team_id

    @property
    def team_id(self) -> Optional[str]:
        return self._team_id


class _DummyAPIClient:
    def __init__(
        self,
        status_payload: Dict[str, Any],
        resize_response: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._status_payload = status_payload
        self._resize_response = resize_response or {"status": "resize_initiated"}
        self.calls: list[Dict[str, Any]] = []

    def get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None, timeout: Any = None
    ) -> Dict[str, Any]:
        self.calls.append({"method": "GET", "endpoint": endpoint, "params": params})
        return self._status_payload

    def patch(
        self,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.calls.append({"method": "PATCH", "endpoint": endpoint, "json": json, "params": params})
        return self._resize_response


def _install(
    monkeypatch: pytest.MonkeyPatch,
    status_payload: Dict[str, Any],
    team_id: Optional[str] = "team-123",
    resize_response: Optional[Dict[str, Any]] = None,
) -> _DummyAPIClient:
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)
    dummy = _DummyAPIClient(status_payload, resize_response=resize_response)
    monkeypatch.setattr("prime_cli.commands.factory.APIClient", lambda: dummy)
    monkeypatch.setattr("prime_cli.commands.factory.Config", lambda: _StubConfig(team_id))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    return dummy


def test_factory_resize_requires_cluster_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _status_payload())

    result = runner.invoke(app, ["factory", "resize", "--nodes", "256"], env=TEST_ENV)

    assert result.exit_code != 0
    assert "Missing argument" in result.output or "CLUSTER" in result.output


def test_factory_resize_requires_nodes_option(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _status_payload())

    result = runner.invoke(app, ["factory", "resize", "research-b300"], env=TEST_ENV)

    assert result.exit_code != 0
    assert "--nodes" in result.output or "Missing option" in result.output


def test_factory_resize_with_confirmation_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = _install(monkeypatch, _status_payload(total_gpus=128))

    result = runner.invoke(
        app, ["factory", "resize", "research-b300", "--nodes", "256", "--yes"], env=TEST_ENV
    )
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "Resize initiated" in output
    assert "research-b300" in output
    assert "256 GPUs" in output
    # Verify the API was called
    patch_calls = [c for c in dummy.calls if c["method"] == "PATCH"]
    assert len(patch_calls) == 1
    assert "/factory/clusters/research-b300/resize" in patch_calls[0]["endpoint"]
    assert patch_calls[0]["json"]["total_gpus"] == 256


def test_factory_resize_shows_confirmation_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _status_payload(total_gpus=128))

    # Simulate user declining
    result = runner.invoke(
        app, ["factory", "resize", "research-b300", "--nodes", "256"], input="n\n", env=TEST_ENV
    )
    output = strip_ansi(result.output)

    assert result.exit_code == 0
    assert "Resize cluster" in output
    assert "Current size: 128 GPUs" in output
    assert "New size:     256 GPUs" in output
    assert "+128 GPUs (scale up)" in output
    assert "cancelled" in output.lower()


def test_factory_resize_scale_down_shows_negative_change(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _status_payload(total_gpus=256))

    result = runner.invoke(
        app, ["factory", "resize", "research-b300", "--nodes", "128"], input="n\n", env=TEST_ENV
    )
    output = strip_ansi(result.output)

    assert result.exit_code == 0
    assert "Current size: 256 GPUs" in output
    assert "New size:     128 GPUs" in output
    assert "-128 GPUs (scale down)" in output


def test_factory_resize_no_change_shows_message(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _status_payload(total_gpus=128))

    result = runner.invoke(
        app, ["factory", "resize", "research-b300", "--nodes", "128"], input="n\n", env=TEST_ENV
    )
    output = strip_ansi(result.output)

    assert result.exit_code == 0
    assert "Current size: 128 GPUs" in output
    assert "New size:     128 GPUs" in output
    assert "no change (current size matches target)" in output


def test_factory_resize_rejects_zero_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _status_payload())

    result = runner.invoke(
        app, ["factory", "resize", "research-b300", "--nodes", "0", "--yes"], env=TEST_ENV
    )
    output = strip_ansi(result.output)

    assert result.exit_code == 1
    assert "must be a positive integer" in output


def test_factory_resize_rejects_negative_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _status_payload())

    result = runner.invoke(
        app, ["factory", "resize", "research-b300", "--nodes", "-10", "--yes"], env=TEST_ENV
    )
    output = strip_ansi(result.output)

    assert result.exit_code != 0


def test_factory_resize_by_cluster_index(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = _install(monkeypatch, _status_payload())

    result = runner.invoke(
        app, ["factory", "resize", "1", "--nodes", "256", "--yes"], env=TEST_ENV
    )
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "Resize initiated" in output
    # Should still use display name in the API call
    patch_calls = [c for c in dummy.calls if c["method"] == "PATCH"]
    assert len(patch_calls) == 1


def test_factory_resize_json_output(monkeypatch: pytest.MonkeyPatch) -> None:
    resize_response = {"status": "resize_initiated", "cluster_id": "research-b300", "target_gpus": 256}
    dummy = _install(monkeypatch, _status_payload(), resize_response=resize_response)

    result = runner.invoke(
        app, ["factory", "resize", "research-b300", "--nodes", "256", "--yes", "--json"], env=TEST_ENV
    )

    assert result.exit_code == 0, result.output
    response = json.loads(result.output)
    assert response["status"] == "resize_initiated"
    assert response["target_gpus"] == 256


def test_factory_resize_handles_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _status_payload())

    # Mock the resize method to raise an error
    with patch.object(FactoryClient, "resize_cluster") as mock_resize:
        mock_resize.side_effect = APIError("Insufficient capacity for resize")

        result = runner.invoke(
            app, ["factory", "resize", "research-b300", "--nodes", "256", "--yes"], env=TEST_ENV
        )
        output = strip_ansi(result.output)

        assert result.exit_code == 1
        assert "Insufficient capacity" in output


def test_factory_resize_no_team_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _status_payload(), team_id=None)

    result = runner.invoke(
        app, ["factory", "resize", "research-b300", "--nodes", "256", "--yes"], env=TEST_ENV
    )
    output = strip_ansi(result.output)

    assert result.exit_code == 0  # Exits cleanly with message
    assert "No team selected" in output


def test_factory_resize_cluster_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _status_payload())

    result = runner.invoke(
        app, ["factory", "resize", "nonexistent-cluster", "--nodes", "256", "--yes"], env=TEST_ENV
    )
    output = strip_ansi(result.output)

    assert result.exit_code == 1
    assert "No cluster matched" in output


def test_factory_resize_with_wait_completes(monkeypatch: pytest.MonkeyPatch) -> None:
    # Start with 128 GPUs
    initial_payload = _status_payload(total_gpus=128)
    # After resize, return 256 GPUs
    updated_payload = _status_payload(total_gpus=256)

    dummy = _install(monkeypatch, initial_payload)

    # Mock time.sleep to avoid actual waiting
    with patch("time.sleep"):
        # Replace the get method to return updated status after first call
        call_count = [0]

        def mock_get(endpoint: str, params: Any = None, timeout: Any = None) -> Dict[str, Any]:
            dummy.calls.append({"method": "GET", "endpoint": endpoint, "params": params})
            call_count[0] += 1
            # First call returns initial state, second returns updated
            return initial_payload if call_count[0] <= 1 else updated_payload

        dummy.get = mock_get

        result = runner.invoke(
            app,
            ["factory", "resize", "research-b300", "--nodes", "256", "--yes", "--wait"],
            env=TEST_ENV,
        )
        output = strip_ansi(result.output)

        assert result.exit_code == 0, result.output
        assert "Waiting for resize to complete" in output
        assert "Resize complete" in output or "256 GPUs" in output


def test_factory_resize_multiple_clusters_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Create payload with duplicate cluster names
    payload = _status_payload()
    payload["clusters"].append(payload["clusters"][0].copy())

    _install(monkeypatch, payload)

    result = runner.invoke(
        app, ["factory", "resize", "research-b300", "--nodes", "256", "--yes"], env=TEST_ENV
    )
    output = strip_ansi(result.output)

    assert result.exit_code == 1
    assert "matches multiple clusters" in output or "Use its 1-based index" in output
