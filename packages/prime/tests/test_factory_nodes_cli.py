"""Tests for `prime factory nodes` (sinfo-style fleet node table)."""

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest
from prime_cli.api.factory import FactoryClient
from prime_cli.main import app
from prime_cli.utils.formatters import strip_ansi
from typer.testing import CliRunner

runner = CliRunner()

TEST_ENV = {
    "PRIME_API_KEY": "dummy",
    "PRIME_DISABLE_VERSION_CHECK": "1",
    "COLUMNS": "220",
    # Rich treats TERM=dumb as a fixed 80-column terminal and then ignores
    # COLUMNS; force a real terminal so the wide table is not truncated.
    "TERM": "xterm",
}


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _source(kind: str, status: str = "ok", age_seconds: int = 12) -> Dict[str, Any]:
    return {
        "kind": kind,
        "status": status,
        "observed_at": _iso(datetime.now(timezone.utc) - timedelta(seconds=age_seconds)),
    }


def _node(name: str, **overrides: Any) -> Dict[str, Any]:
    node: Dict[str, Any] = {
        "name": name,
        "state": "ready",
        "gpu_type": "H200_141GB",
        "gpus_total": 8,
        "gpus_used": 8,
        "assigned_to": "slurm",
    }
    node.update(overrides)
    return node


def _nodes_cluster(
    display_name: str = "research-b300",
    nodes: Optional[List[Dict[str, Any]]] = None,
    status: str = "online",
) -> Dict[str, Any]:
    # The frozen nodes contract: clusters carry no sources; the envelope
    # carries one capacity entry per cluster, in the same order.
    return {
        "display_name": display_name,
        "status": status,
        "nodes": nodes if nodes is not None else [_node("gpu-01"), _node("gpu-02")],
    }


def _nodes_payload(
    clusters: Optional[List[Dict[str, Any]]] = None,
    sources: Optional[List[Dict[str, Any]]] = None,
    **cluster_overrides: Any,
) -> Dict[str, Any]:
    if clusters is None:
        cluster = _nodes_cluster()
        cluster.update(cluster_overrides)
        clusters = [cluster]
    if sources is None:
        sources = [_source("capacity") for _ in clusters]
    return {
        "schema_version": 1,
        "as_of": _iso(datetime.now(timezone.utc)),
        "clusters": clusters,
        "sources": sources,
    }


class _StubConfig:
    def __init__(self, team_id: Optional[str]) -> None:
        self._team_id = team_id

    @property
    def team_id(self) -> Optional[str]:
        return self._team_id


class _DummyAPIClient:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload
        self.calls: List[Dict[str, Any]] = []

    def get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None, timeout: Any = None
    ) -> Dict[str, Any]:
        self.calls.append({"endpoint": endpoint, "params": params})
        return self._payload


def _install(
    monkeypatch: pytest.MonkeyPatch,
    payload: Dict[str, Any],
    team_id: Optional[str] = "team-123",
) -> _DummyAPIClient:
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)
    dummy = _DummyAPIClient(payload)
    monkeypatch.setattr("prime_cli.commands.factory.APIClient", lambda: dummy)
    monkeypatch.setattr("prime_cli.commands.factory.Config", lambda: _StubConfig(team_id))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    return dummy


def test_nodes_table_renders_clusters_and_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _nodes_payload())

    result = runner.invoke(app, ["factory", "nodes"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "CLUSTERS" in output
    assert "research-b300" in output
    assert "online" in output
    for header in ("NODE", "STATE", "HELD", "ASSIGNED TO"):
        assert header in output
    assert "gpu-01" in output and "gpu-02" in output
    assert "ready" in output
    assert "8/8" in output
    assert "slurm" in output
    assert "Error" not in output


def test_nodes_states_and_assignment_render_honestly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _nodes_payload(
        nodes=[
            _node("gpu-01", state="cordoned", assigned_to="slurm"),
            _node("gpu-02", state="offline", gpus_used=None),
            _node("gpu-03", state="unknown", assigned_to=None, gpus_total=None),
        ]
    )
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "nodes"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "cordoned" in output and "slurm" in output
    assert "offline" in output
    # Unobserved counts stay dashes, never fabricated zeros.
    assert "-/8" in output
    assert "unknown" in output
    # Null assignment means placement is NOT observed — rendered as a
    # lowercase dim "unknown", never as unassigned.
    assert "null" not in output
    assert "unknown" in output


def test_nodes_json_prints_exact_api_response(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _nodes_payload()
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "nodes", "--json"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == payload


def test_nodes_empty_fleet_prints_friendly_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(
        monkeypatch,
        {
            "schema_version": 1,
            "as_of": _iso(datetime.now(timezone.utc)),
            "clusters": [],
            "sources": [],
        },
    )

    result = runner.invoke(app, ["factory", "nodes"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "No factory clusters allocated." in output


def test_nodes_cluster_without_nodes_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _nodes_payload(nodes=[]))

    result = runner.invoke(app, ["factory", "nodes"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "no nodes reported" in output


def test_nodes_no_team_selected_prints_switch_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy = _install(monkeypatch, _nodes_payload(), team_id=None)

    result = runner.invoke(app, ["factory", "nodes"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "prime switch" in output
    assert not dummy.calls


def test_nodes_team_flag_overrides_missing_team_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy = _install(monkeypatch, _nodes_payload(), team_id=None)

    result = runner.invoke(app, ["factory", "nodes", "--team", "team-42"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert dummy.calls and dummy.calls[0]["params"] == {"team_id": "team-42"}


def test_nodes_stale_source_suppresses_table_with_friendly_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _nodes_payload(sources=[_source("capacity", status="stale", age_seconds=7200)])
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "nodes"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "node breakdown unavailable" in output
    assert "node data last seen 2h ago" in output
    assert "gpu-01" not in output  # no stale rows rendered
    assert "NODE" not in output  # no table at all


def test_nodes_mixed_fresh_and_stale_clusters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _nodes_payload(
        clusters=[
            _nodes_cluster(display_name="fresh-b300"),
            _nodes_cluster(display_name="stale-h200"),
        ],
        sources=[
            _source("capacity"),
            _source("capacity", status="error", age_seconds=3600),
        ],
    )
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "nodes"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "fresh-b300" in output
    assert "gpu-01" in output
    assert "stale-h200" in output
    assert "node breakdown unavailable — node data last seen 1h ago" in output


def test_nodes_cluster_filter_and_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _nodes_payload(
        clusters=[
            _nodes_cluster(display_name="research-b300"),
            _nodes_cluster(display_name="office-a100", nodes=[_node("a100-01")]),
        ],
    )
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "nodes", "--cluster", "office-a100"], env=TEST_ENV)
    output = strip_ansi(result.output)
    assert result.exit_code == 0, result.output
    assert "a100-01" in output
    assert "gpu-01" not in output

    miss = runner.invoke(app, ["factory", "nodes", "--cluster", "nope"], env=TEST_ENV)
    miss_output = strip_ansi(miss.output)
    assert miss.exit_code == 1, miss.output
    assert "No cluster matched" in miss_output


def test_nodes_state_filter_and_assigned_to_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _nodes_payload(
        nodes=[
            _node("gpu-01", state="cordoned", assigned_to="slurm"),
            _node("gpu-02", state="ready", assigned_to=None),
        ]
    )
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "nodes", "--state", "cordoned"], env=TEST_ENV)
    output = strip_ansi(result.output)
    assert result.exit_code == 0, result.output
    assert "gpu-01" in output
    assert "gpu-02" not in output

    result = runner.invoke(app, ["factory", "nodes", "--assigned-to", "slurm"], env=TEST_ENV)
    output = strip_ansi(result.output)
    assert result.exit_code == 0, result.output
    assert "gpu-01" in output
    assert "gpu-02" not in output

    # training/inference placement is not observed: the filter refuses.
    refused = runner.invoke(app, ["factory", "nodes", "--assigned-to", "training"], env=TEST_ENV)
    refused_output = strip_ansi(refused.output)
    assert refused.exit_code == 1, refused.output
    assert "Only slurm is observable" in refused_output


def test_nodes_filters_apply_to_json_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _nodes_payload(
        nodes=[
            _node("gpu-01", state="cordoned", assigned_to="slurm"),
            _node("gpu-02", state="ready", assigned_to="training"),
        ]
    )
    _install(monkeypatch, payload)

    result = runner.invoke(
        app, ["factory", "nodes", "--assigned-to", "slurm", "--json"], env=TEST_ENV
    )
    data = json.loads(result.stdout)

    assert result.exit_code == 0, result.output
    names = [n["name"] for n in data["clusters"][0]["nodes"]]
    assert names == ["gpu-01"]
    assert data["schema_version"] == 1


@pytest.mark.parametrize(
    "flag,value",
    [("--state", "flying"), ("--assigned-to", "research")],
)
def test_nodes_invalid_filter_values_exit_nonzero(
    monkeypatch: pytest.MonkeyPatch, flag: str, value: str
) -> None:
    dummy = _install(monkeypatch, _nodes_payload())

    result = runner.invoke(app, ["factory", "nodes", flag, value], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 1, result.output
    assert "Invalid" in output
    assert not dummy.calls


def test_nodes_rich_markup_in_dynamic_values_is_escaped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _nodes_payload(nodes=[_node("gpu-[bold]01", state="ready", assigned_to="[red]team")])
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "nodes"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "[bold]" in output
    assert "[red]" in output


def test_nodes_unsupported_schema_version_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _nodes_payload()
    payload["schema_version"] = 2
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "nodes"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 1, result.output
    assert "Unsupported factory nodes schema version" in output


def test_factory_client_nodes_endpoint_and_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy = _DummyAPIClient(_nodes_payload())
    client = FactoryClient(dummy)  # type: ignore[arg-type]

    client.get_nodes("team-9")

    assert dummy.calls == [{"endpoint": "/factory/nodes", "params": {"team_id": "team-9"}}]


def test_nodes_cluster_index_selector_filters_raw_json_by_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Duplicate display names: index selection must pick exactly the
    # selected cluster in --json, like table mode, not name-match both.
    payload = _nodes_payload(
        clusters=[
            _nodes_cluster(display_name="twin"),
            _nodes_cluster(display_name="twin", nodes=[_node("gpu-99")]),
        ],
        sources=[
            _source("capacity", age_seconds=12),
            _source("capacity", age_seconds=5),
        ],
    )
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "nodes", "--cluster", "2", "--json"], env=TEST_ENV)
    data = json.loads(result.stdout)

    assert result.exit_code == 0, result.output
    assert len(data["clusters"]) == 1
    assert [n["name"] for n in data["clusters"][0]["nodes"]] == ["gpu-99"]
    # The envelope-level sources follow the same index selection.
    assert len(data["sources"]) == 1
    assert data["sources"][0]["observed_at"] == payload["sources"][1]["observed_at"]


def test_nodes_missing_capacity_entry_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    # A cluster with no source entry at all has unknown evidence: no node
    # table, one honest degraded line — never a silently fresh rendering.
    payload = _nodes_payload(sources=[])
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "nodes"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "node breakdown unavailable" in output
    assert "gpu-01" not in output
    assert "no nodes reported" not in output


def test_nodes_filter_match_empty_distinguished_from_no_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _nodes_payload())

    filtered = runner.invoke(app, ["factory", "nodes", "--state", "cordoned"], env=TEST_ENV)
    output = strip_ansi(filtered.output)
    assert filtered.exit_code == 0, filtered.output
    assert "no nodes match the given filters" in output
    assert "no nodes reported" not in output

    plain = runner.invoke(app, ["factory", "nodes"], env=TEST_ENV)
    plain_output = strip_ansi(plain.output)
    assert plain.exit_code == 0, plain.output
    assert "no nodes reported" not in plain_output  # fixture has nodes
    assert "no nodes match" not in plain_output


def test_nodes_client_parses_real_backend_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    # Realistic served shape (PR #6339): sources at the TOP level, one
    # capacity entry per cluster in cluster order; clusters carry no
    # sources field. Validation must succeed against a populated response.
    realistic = {
        "schema_version": 1,
        "as_of": _iso(datetime.now(timezone.utc)),
        "clusters": [
            {
                "display_name": "research-b300",
                "status": "online",
                "nodes": [
                    {
                        "name": "gpu-01",
                        "state": "ready",
                        "gpu_type": "B300",
                        "gpus_total": 8,
                        "gpus_used": 8,
                        "assigned_to": "slurm",
                    },
                    {
                        "name": "gpu-02",
                        "state": "cordoned",
                        "gpu_type": "B300",
                        "gpus_total": 8,
                        "gpus_used": None,
                        "assigned_to": None,
                    },
                ],
            },
            {
                "display_name": "office-a100",
                "status": "offline",
                "nodes": [],
            },
        ],
        "sources": [
            _source("capacity", age_seconds=10),
            _source("capacity", status="stale", age_seconds=86400),
        ],
    }
    dummy = _DummyAPIClient(realistic)
    client = FactoryClient(dummy)  # type: ignore[arg-type]

    nodes = client.get_nodes("team-1")

    assert [c.display_name for c in nodes.clusters] == ["research-b300", "office-a100"]
    assert [s.status for s in nodes.sources] == ["ok", "stale"]
    assert nodes.raw_response is realistic
    # And the CLI renders it: fresh cluster gets its table, stale one its line.
    _install(monkeypatch, realistic)
    result = runner.invoke(app, ["factory", "nodes"], env=TEST_ENV)
    output = strip_ansi(result.output)
    assert result.exit_code == 0, result.output
    assert "gpu-01" in output
    # The second cluster's paired capacity entry is stale: its breakdown
    # stays visible instead of a quiet empty table.
    assert "node breakdown unavailable — node data last seen 1d ago" in output
    assert "office-a100" in output
