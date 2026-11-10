"""Tests for `prime inference deployments`."""

import json
from typing import Any, Dict, List, Optional

import pytest
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
DEPLOYMENT = {
    "id": "job-1",
    "deploymentName": "qwen-a1b2",
    "status": "RUNNING",
    "servedModelAlias": "frontier/dep-1/qwen",
    "endpointUrl": "https://api.pinference.ai/api/v1",
    "isOnDemand": True,
    "replicas": 2,
    "gpusPerReplica": 1,
    "maxReplicas": 8,
    "scaleState": None,
    "createdAt": "2026-11-09T12:00:00Z",
    "dashboardUrl": "https://app.primeintellect.ai/dashboard/inference/dedicated/job-1",
}


class _StubConfig:
    def __init__(self, team_id: Optional[str]) -> None:
        self.team_id = team_id


class _DummyAPIClient:
    def __init__(self, error: Optional[APIError] = None) -> None:
        self.error = error
        self.calls: List[Dict[str, Any]] = []

    def get(self, endpoint, params=None):
        self.calls.append({"method": "GET", "endpoint": endpoint, "params": params})
        if endpoint == "/inference/deployments":
            return {"data": [DEPLOYMENT]}
        return DEPLOYMENT

    def post(self, endpoint, json=None):
        self.calls.append({"method": "POST", "endpoint": endpoint, "json": json})
        if self.error is not None:
            raise self.error
        return {**DEPLOYMENT, "replicas": json["replicas"], "scaleState": "converging"}


def _install(
    monkeypatch: pytest.MonkeyPatch,
    team_id: Optional[str] = None,
    error: Optional[APIError] = None,
) -> _DummyAPIClient:
    dummy = _DummyAPIClient(error)
    module = "prime_cli.commands.inference_deployments"
    monkeypatch.setattr(f"{module}.APIClient", lambda: dummy)
    monkeypatch.setattr(f"{module}.Config", lambda: _StubConfig(team_id))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    return dummy


@pytest.mark.parametrize(("team_id", "params"), [("team-1", {"teamId": "team-1"}), (None, None)])
def test_list_scopes_to_the_selected_team_or_the_caller(
    monkeypatch: pytest.MonkeyPatch, team_id: Optional[str], params
) -> None:
    dummy = _install(monkeypatch, team_id)

    result = runner.invoke(app, ["inference", "deployments", "list"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert "qwen-a1b2" in strip_ansi(result.output)
    assert "2 x 1 GPU" in strip_ansi(result.output)
    assert dummy.calls == [
        {"method": "GET", "endpoint": "/inference/deployments", "params": params}
    ]


def test_scale_posts_the_target_and_echoes_the_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy = _install(monkeypatch)

    result = runner.invoke(
        app, ["inference", "deployments", "scale", "job-1", "3", "--json"], env=TEST_ENV
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["replicas"] == 3
    assert dummy.calls == [
        {
            "method": "POST",
            "endpoint": "/inference/deployments/job-1/scale",
            "json": {"replicas": 3},
        }
    ]


def test_scale_refusal_is_a_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, error=APIError("HTTP 409: no room for 2 more GPUs"))

    result = runner.invoke(app, ["inference", "deployments", "scale", "job-1", "4"], env=TEST_ENV)

    assert result.exit_code == 1
    assert "no room for 2 more GPUs" in strip_ansi(result.output)
