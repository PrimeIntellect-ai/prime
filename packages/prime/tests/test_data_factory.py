import json
from typing import Any, Dict, Optional

import pytest
from prime_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()

TEST_ENV = {
    "COLUMNS": "200",
    "LINES": "50",
    "PRIME_DISABLE_VERSION_CHECK": "1",
    "PRIME_API_KEY": "test-key",
}
RUN = {
    "id": "run1",
    "graph_id": "g1",
    "version": "a" * 64,
    "status": "RUNNING",
    "error": None,
    "stopped": False,
    "sample": None,
    "outputs": [{"name": "report.md", "node": "report", "bytes": 3}],
    "state": {
        "nodes": {"scout": {"done": 1, "running": 0, "held": 0, "failed": 0, "cost": 0.25}},
        "cards": [{"job": "graph", "node": "review", "question": "Keep or drop each idea."}],
        "cost": {"usd": 0.25},
    },
    "created_at": "2026-11-05T00:00:00",
}


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    calls: Dict[str, Any] = {}

    def post(
        self: Any, endpoint: str, json: Optional[Dict[str, Any]] = None, timeout: Any = None
    ) -> Dict[str, Any]:
        calls["post"] = (endpoint, json)
        return RUN

    def get(self: Any, endpoint: str, params: Any = None, timeout: Any = None) -> Dict[str, Any]:
        calls["get"] = endpoint
        return RUN

    monkeypatch.setattr("prime_cli.client.APIClient.post", post)
    monkeypatch.setattr("prime_cli.client.APIClient.get", get)
    return calls


def test_run_submits_the_graph_with_the_secrets_it_names(api: Dict[str, Any], tmp_path) -> None:
    graph = tmp_path / "g.json"
    graph.write_text(json.dumps({"name": "g", "nodes": {}}))
    env = {**TEST_ENV, "GITHUB_TOKEN": "gh"}
    result = runner.invoke(
        app,
        ["data-factory", "run", str(graph), "--secret", "GITHUB_TOKEN", "--sample", "2"],
        env=env,
    )
    assert result.exit_code == 0, result.output
    endpoint, body = api["post"]
    assert endpoint == "/data-factory/runs"
    assert body["graph"] == {"name": "g", "nodes": {}}
    assert body["secrets"] == {"GITHUB_TOKEN": "gh"} and body["sample"] == 2
    assert "prime data-factory status run1" in result.output


def test_run_refuses_a_secret_missing_from_the_environment(api: Dict[str, Any], tmp_path) -> None:
    graph = tmp_path / "g.json"
    graph.write_text("{}")
    result = runner.invoke(
        app, ["data-factory", "run", str(graph), "--secret", "NOPE"], env=TEST_ENV
    )
    assert result.exit_code == 1 and "NOPE" in result.output and "post" not in api


def test_status_shows_counts_cards_cost_and_outputs(api: Dict[str, Any]) -> None:
    result = runner.invoke(app, ["data-factory", "status", "run1"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    assert api["get"] == "/data-factory/runs/run1"
    for text in ("scout", "Waiting on you", "Keep or drop each idea.", "$0.25", "report/report.md"):
        assert text in result.output


def test_status_as_json(api: Dict[str, Any]) -> None:
    result = runner.invoke(app, ["data-factory", "status", "run1", "-o", "json"], env=TEST_ENV)
    assert result.exit_code == 0 and json.loads(result.output)["id"] == "run1"


def test_stop(api: Dict[str, Any]) -> None:
    result = runner.invoke(app, ["data-factory", "stop", "run1"], env=TEST_ENV)
    assert result.exit_code == 0 and api["post"] == ("/data-factory/runs/run1/stop", None)
