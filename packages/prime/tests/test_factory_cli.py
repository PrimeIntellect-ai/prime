"""Tests for `prime factory status` (fleet node usage glance)."""

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

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


def _status_payload(**cluster_overrides: Any) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    cluster: Dict[str, Any] = {
        "display_name": "cluster-a",
        "gpu_type": "B300",
        "total_gpus": 128,
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


def _default_nodes_payload() -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "schema_version": 1,
        "as_of": _iso(now),
        "sources": [_source("capacity", age_seconds=12)],
        "clusters": [
            {
                "display_name": "cluster-a",
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
                        "state": "ready",
                        "gpu_type": "B300",
                        "gpus_total": 8,
                        "gpus_used": 4,
                        "assigned_to": "slurm",
                    },
                ],
            }
        ],
    }


def _workloads_payload(*workloads: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "as_of": _iso(datetime.now(timezone.utc)),
        "workloads": list(workloads),
        "sources": [_source("slurm", age_seconds=3)],
    }


class _DummyAPIClient:
    def __init__(
        self,
        payload: Dict[str, Any],
        nodes_payload: Optional[Dict[str, Any]] = None,
        workloads_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._payload = payload
        self._nodes_payload = (
            nodes_payload if nodes_payload is not None else _default_nodes_payload()
        )
        self._workloads_payload = (
            workloads_payload if workloads_payload is not None else _workloads_payload()
        )
        self.calls: list[Dict[str, Any]] = []

    def get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None, timeout: Any = None
    ) -> Dict[str, Any]:
        self.calls.append({"endpoint": endpoint, "params": params})
        if endpoint == "/factory/nodes":
            return self._nodes_payload
        if endpoint == "/factory/workloads":
            return self._workloads_payload
        return self._payload


def _install(
    monkeypatch: pytest.MonkeyPatch,
    payload: Dict[str, Any],
    team_id: Optional[str] = "team-123",
    nodes_payload: Optional[Dict[str, Any]] = None,
    workloads_payload: Optional[Dict[str, Any]] = None,
) -> _DummyAPIClient:
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)
    dummy = _DummyAPIClient(
        payload, nodes_payload=nodes_payload, workloads_payload=workloads_payload
    )
    monkeypatch.setattr("prime_cli.commands.factory.APIClient", lambda: dummy)
    monkeypatch.setattr("prime_cli.commands.factory.Config", lambda: _StubConfig(team_id))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    return dummy


def _node(
    name: str,
    state: Optional[str] = "ready",
    assigned_to: Optional[str] = None,
    gpus_used: Optional[int] = 8,
) -> Dict[str, Any]:
    return {
        "name": name,
        "state": state,
        "gpu_type": "B300",
        "gpus_total": 8,
        "gpus_used": gpus_used,
        "assigned_to": assigned_to,
    }


def _nodes_payload(*clusters: Tuple[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "as_of": _iso(datetime.now(timezone.utc)),
        "sources": [_source("capacity") for _ in clusters],
        "clusters": [
            {"display_name": name, "status": "online", "nodes": nodes} for name, nodes in clusters
        ],
    }


def _slurm_job(gpus: int, cluster: str = "cluster-a", scheduler: str = "slurm-a") -> Dict[str, Any]:
    return {
        "id": f"slurm:{scheduler}-{gpus}",
        "type": "slurm",
        "cluster_display_name": cluster,
        "name": "train",
        "state": "running",
        "owner": {"kind": "slurm", "display_name": "alice"},
        "requested_gpus": gpus,
        "allocated_gpus": gpus,
        "scheduler_display_name": scheduler,
        "source": _source("slurm"),
    }


def _plain_status_rows(*args: str) -> List[List[str]]:
    result = runner.invoke(app, ["factory", "status", "--plain", *args], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    return [line.split() for line in result.output.splitlines()]


def test_factory_status_all_fresh_sources_show_no_unavailable_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _status_payload())

    result = runner.invoke(app, ["factory", "status"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "unavailable" not in output
    assert "last seen" not in output
    assert "Warning" not in output


def test_factory_status_json_prints_exact_api_response(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _status_payload()
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "status", "--json"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == payload


def test_factory_status_output_json_flag_matches_json_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _status_payload()
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "status", "--output", "json"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == payload


def test_factory_status_empty_response_prints_friendly_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(
        monkeypatch,
        {"schema_version": 1, "as_of": _iso(datetime.now(timezone.utc)), "clusters": []},
    )

    result = runner.invoke(app, ["factory", "status"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "No factory clusters allocated." in output


def test_factory_status_empty_response_json_keeps_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(
        monkeypatch,
        {"schema_version": 1, "as_of": _iso(datetime.now(timezone.utc)), "clusters": []},
    )

    result = runner.invoke(app, ["factory", "status", "--json"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["clusters"] == []


def test_factory_status_no_team_selected_prints_switch_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _status_payload(), team_id=None)

    result = runner.invoke(app, ["factory", "status"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "prime switch" in output
    assert "No factory clusters allocated." not in output


def test_factory_status_team_flag_overrides_missing_team_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy = _install(monkeypatch, _status_payload(), team_id=None)

    result = runner.invoke(app, ["factory", "status", "--team", "team-42"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert {"endpoint": "/factory/status", "params": {"team_id": "team-42"}} in dummy.calls
    assert all(call["params"]["team_id"] == "team-42" for call in dummy.calls)


def test_factory_status_cluster_filter_by_display_name_and_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _status_payload()
    second = dict(_status_payload()["clusters"][0], display_name="cluster-b", gpu_type="H200")
    payload["clusters"].append(second)
    _install(monkeypatch, payload)

    by_name = runner.invoke(
        app, ["factory", "status", "--verbose", "--cluster", "cluster-b"], env=TEST_ENV
    )
    assert by_name.exit_code == 0, by_name.output
    out = strip_ansi(by_name.output)
    assert "cluster-b" in out
    assert "cluster-a" not in out
    assert "H200" in out

    by_index = runner.invoke(
        app, ["factory", "status", "--verbose", "--cluster", "1"], env=TEST_ENV
    )
    assert by_index.exit_code == 0, by_index.output
    out = strip_ansi(by_index.output)
    assert "cluster-a" in out
    assert "cluster-b" not in out

    # Compact mode: --cluster selects table rows too.
    compact = runner.invoke(app, ["factory", "status", "--cluster", "cluster-b"], env=TEST_ENV)
    assert compact.exit_code == 0, compact.output
    out = strip_ansi(compact.output)
    assert "cluster-b" in out
    assert "cluster-a" not in out


def test_factory_status_unknown_cluster_selector_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _status_payload())

    result = runner.invoke(app, ["factory", "status", "--cluster", "nope"], env=TEST_ENV)

    assert result.exit_code == 1
    assert "No cluster matched 'nope'" in strip_ansi(result.output)


def test_factory_status_json_with_cluster_filter_keeps_exact_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _status_payload()
    payload["clusters"].append(
        dict(payload["clusters"][0], display_name="cluster-b", gpu_type="H200")
    )
    _install(monkeypatch, payload)

    result = runner.invoke(
        app, ["factory", "status", "--json", "--cluster", "cluster-b"], env=TEST_ENV
    )

    assert result.exit_code == 0, result.output
    # The filtered view passes through the raw API cluster object byte-exact,
    # not a re-serialization of the parsed model.
    assert json.loads(result.stdout) == {
        **payload,
        "clusters": [payload["clusters"][1]],
    }


def test_factory_status_json_no_team_keeps_stdout_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _status_payload(), team_id=None)

    result = runner.invoke(app, ["factory", "status", "--json"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert result.stdout == ""
    assert "prime switch" in strip_ansi(result.stderr)


def test_factory_status_json_cluster_miss_keeps_stdout_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _status_payload())

    result = runner.invoke(app, ["factory", "status", "--json", "--cluster", "nope"], env=TEST_ENV)

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "No cluster matched 'nope'" in strip_ansi(result.stderr)


def test_factory_status_json_ambiguous_cluster_keeps_stdout_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _status_payload()
    payload["clusters"].append(dict(payload["clusters"][0], gpu_type="H200"))
    _install(monkeypatch, payload)

    result = runner.invoke(
        app, ["factory", "status", "--json", "--cluster", "cluster-a"], env=TEST_ENV
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "matches multiple clusters" in strip_ansi(result.stderr)


def test_factory_status_reports_api_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FailingAPIClient:
        def get(self, endpoint: str, params: Any = None, timeout: Any = None) -> Dict[str, Any]:
            raise APIError("HTTP 503: factory status unavailable")

    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)
    monkeypatch.setattr("prime_cli.commands.factory.APIClient", _FailingAPIClient)
    monkeypatch.setattr("prime_cli.commands.factory.Config", lambda: _StubConfig("team-123"))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    result = runner.invoke(app, ["factory", "status"], env=TEST_ENV)

    assert result.exit_code == 1
    assert "factory status unavailable" in strip_ansi(result.output)


def test_factory_client_get_status_calls_frozen_endpoint() -> None:
    payload = _status_payload()
    dummy = _DummyAPIClient(payload)

    status = FactoryClient(dummy).get_status("team-123")  # type: ignore[arg-type]

    assert dummy.calls == [{"endpoint": "/factory/status", "params": {"team_id": "team-123"}}]
    assert status.raw_response is payload
    assert status.schema_version == 1
    assert len(status.clusters) == 1
    cluster = status.clusters[0]
    assert cluster.display_name == "cluster-a"
    assert [p.type for p in cluster.pools] == ["training", "inference", "slurm"]
    assert [s.kind for s in cluster.sources] == [
        "capacity",
        "training",
        "inference",
        "slurm",
    ]


def test_factory_status_rejects_response_missing_clusters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A 200 without `clusters` is a malformed payload, not an empty fleet.
    _install(monkeypatch, {"schema_version": 1, "as_of": _iso(datetime.now(timezone.utc))})

    result = runner.invoke(app, ["factory", "status"], env=TEST_ENV)

    assert result.exit_code == 1
    assert "Unexpected factory status response shape" in strip_ansi(result.output)
    assert "No factory clusters allocated." not in result.output


def test_factory_status_rejects_malformed_response_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, {"as_of": _iso(datetime.now(timezone.utc)), "clusters": []})

    result = runner.invoke(app, ["factory", "status"], env=TEST_ENV)

    assert result.exit_code == 1
    assert "Unexpected factory status response shape" in strip_ansi(result.output)
    assert "Traceback" not in result.output


def test_factory_client_validation_failure_raises_api_error() -> None:
    dummy = _DummyAPIClient({"schema_version": 1})  # missing required `clusters`

    with pytest.raises(APIError, match="Unexpected factory status response shape"):
        FactoryClient(dummy).get_status("team-123")  # type: ignore[arg-type]


def test_factory_status_rejects_unsupported_schema_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _status_payload()
    payload["schema_version"] = 2
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "status"], env=TEST_ENV)

    assert result.exit_code == 1
    assert "Unsupported factory status schema version: 2" in strip_ansi(result.output)


def test_factory_status_rejects_cluster_missing_pools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _status_payload()
    del payload["clusters"][0]["pools"]
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "status"], env=TEST_ENV)

    assert result.exit_code == 1
    assert "Unexpected factory status response shape" in strip_ansi(result.output)
    assert "allocations" in strip_ansi(result.output)


def test_factory_status_rejects_cluster_missing_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _status_payload()
    del payload["clusters"][0]["sources"]
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "status"], env=TEST_ENV)

    assert result.exit_code == 1
    assert "Unexpected factory status response shape" in strip_ansi(result.output)
    assert "sources" in strip_ansi(result.output)


def test_factory_client_validation_error_message_has_no_rich_brackets() -> None:
    # Pydantic messages embed "[type=...]" metadata; those brackets would
    # crash Rich rendering when printed. The wrapped APIError must not.
    dummy = _DummyAPIClient({"clusters": [{"display_name": "x"}]})

    with pytest.raises(APIError) as excinfo:
        FactoryClient(dummy).get_status("team-123")  # type: ignore[arg-type]

    assert "[" not in str(excinfo.value)
    assert "allocations" in str(excinfo.value)


def test_factory_client_rejects_future_schema_version() -> None:
    dummy = _DummyAPIClient({**_status_payload(), "schema_version": 3})

    with pytest.raises(APIError, match="Unsupported factory status schema version: 3"):
        FactoryClient(dummy).get_status("team-123")  # type: ignore[arg-type]


def test_factory_status_markup_in_cluster_selector_does_not_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A selector containing Rich closing markup must print a normal error,
    # not raise rich.errors.MarkupError.
    _install(monkeypatch, _status_payload())

    result = runner.invoke(app, ["factory", "status", "--cluster", "[/bold]"], env=TEST_ENV)

    assert result.exit_code == 1
    # Exit must come from the command (SystemExit), not an escaped MarkupError.
    assert isinstance(result.exception, SystemExit)
    assert "No cluster matched" in strip_ansi(result.output)


def test_factory_status_ambiguous_selector_with_markup_does_not_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _status_payload()
    payload["clusters"][0]["display_name"] = "[/bold]"
    payload["clusters"].append(dict(payload["clusters"][0], gpu_type="H200"))
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "status", "--cluster", "[/bold]"], env=TEST_ENV)

    assert result.exit_code == 1
    # Exit must come from the command (SystemExit), not an escaped MarkupError.
    assert isinstance(result.exception, SystemExit)
    assert "matches multiple clusters" in strip_ansi(result.output)


def test_factory_status_markup_in_backend_values_does_not_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # API-supplied gpu_type / source kind / source status containing Rich
    # markup must render as literal text, not raise MarkupError.
    payload = _status_payload(
        gpu_type="[/bold]B300",
        sources=[_source("[/bold]capacity", status="[/bold]stale", age_seconds=30)],
    )
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "status", "--verbose"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert isinstance(result.exception, (type(None), SystemExit))
    output = strip_ansi(result.output)
    assert "cluster-a" in output
    # The degraded (markup-named) source surfaces as a last-seen phrase,
    # with its markup rendered as literal text instead of crashing.
    assert "last seen" in output
    assert "[/bold]capacity data last seen" in output


@pytest.mark.parametrize("digit", ["\u00b2", "\u2460"])
def test_factory_status_unicode_digit_selector_misses_cleanly(
    monkeypatch: pytest.MonkeyPatch, digit: str
) -> None:
    # Unicode digits pass str.isdigit() but int() cannot parse them; they
    # must be treated as display-name selectors and miss cleanly.
    _install(monkeypatch, _status_payload())

    result = runner.invoke(app, ["factory", "status", "--cluster", digit], env=TEST_ENV)

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "No cluster matched" in strip_ansi(result.output)


def test_factory_status_ascii_index_still_selects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _status_payload()
    payload["clusters"].append(dict(payload["clusters"][0], display_name="cluster-b"))
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "status", "--cluster", "2"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert "cluster-b" in strip_ansi(result.output)
    assert "cluster-a" not in strip_ansi(result.output)


def test_factory_status_accepts_allocations_envelope_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The backend renamed the status envelope key from `pools` to
    # `allocations` (user vocabulary); the CLI parses the new primary shape
    # and tolerates the legacy `pools` key, with --json an exact passthrough.
    payload = _status_payload()
    for cluster in payload["clusters"]:
        cluster["allocations"] = cluster.pop("pools")
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "status", "--verbose"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "cluster-a" in output
    assert "slurm" in output

    json_result = runner.invoke(app, ["factory", "status", "--json"], env=TEST_ENV)
    assert json.loads(json_result.stdout) == payload

    # Legacy `pools` key still parses (tolerated alias, not the primary).
    legacy_payload = _status_payload()  # fixture still uses the `pools` key
    _install(monkeypatch, legacy_payload)
    legacy = runner.invoke(app, ["factory", "status", "--verbose"], env=TEST_ENV)
    assert legacy.exit_code == 0, legacy.output
    assert "slurm" in strip_ansi(legacy.output)


def test_factory_status_mixed_naive_and_aware_timestamps_compare_correctly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A response mixing naive and offset-aware timestamps must not crash
    # source comparison, and ages must be computed on one UTC base.
    now = datetime.now(timezone.utc)
    naive_two_hours_ago = (now - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
    aware_one_hour_ago = (
        (now - timedelta(hours=1)).astimezone(timezone(timedelta(hours=2))).isoformat()
    )
    payload = _status_payload(
        sources=[
            {"kind": "capacity", "status": "stale", "observed_at": naive_two_hours_ago},
            {"kind": "slurm", "status": "stale", "observed_at": aware_one_hour_ago},
            _source("training", age_seconds=10),
            _source("inference", age_seconds=10),
        ],
    )
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "status", "--verbose"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    # The oldest observation (naive -2h, read as UTC) drives the phrase.
    assert "last seen 2h ago" in output
    assert "last seen 1h ago" not in output


def test_factory_status_default_table_counts_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _status_payload())

    result = runner.invoke(app, ["factory", "status"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    for header in ("CLUSTER", "GPU", "NODES", "FREE", "SLURM", "BUSY", "UNAVAILABLE"):
        assert header in output
    for dropped in ("TOTAL", "HELD", "UNUSED", "STATUS", "DATA"):
        assert dropped not in output
    # Two ready Slurm nodes and no running jobs: 0 free, 2 slurm (idle), 0 busy, 0 unavailable.
    # Without --verbose or degraded data there are no sub-rows, notes, or warnings.
    rows = _plain_status_rows()
    assert rows == [
        ["CLUSTER", "GPU", "NODES", "FREE", "SLURM", "BUSY", "UNAVAILABLE"],
        ["cluster-a", "B300", "2", "0", "2", "0", "0"],
    ]
    # Verify four-bucket invariant: FREE + SLURM + BUSY + UNAVAILABLE == NODES
    assert int(rows[1][3]) + int(rows[1][4]) + int(rows[1][5]) + int(rows[1][6]) == int(rows[1][2])


def test_factory_status_counts_nodes_by_state_and_running_slurm_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nodes = [
        _node("slurm-1", assigned_to="slurm"),
        _node("slurm-2", assigned_to="slurm"),
        _node("slurm-3", assigned_to="slurm"),
        _node("slurm-cordoned", state="cordoned", assigned_to="slurm"),
        _node("slurm-draining", state="draining", assigned_to="slurm"),
        _node("busy-1"),
        _node("busy-2"),
        _node("idle", gpus_used=0),
        _node("offline", state="offline", gpus_used=None),
        _node("unknown", state="unknown", gpus_used=0),
    ]
    jobs = [
        _slurm_job(12),  # spans 2 nodes
        _slurm_job(0),  # CPU-only: no node
        _slurm_job(64, cluster="elsewhere"),
    ]
    _install(
        monkeypatch,
        _status_payload(),
        nodes_payload=_nodes_payload(("cluster-a", nodes)),
        workloads_payload=_workloads_payload(*jobs),
    )

    # NODES 10 = FREE (1 idle direct) + SLURM (5 total: 3 ready + cordoned + draining)
    # + BUSY (2 direct busy) + UNAVAILABLE (2 direct: offline + unknown).
    rows = _plain_status_rows()
    assert rows[1] == ["cluster-a", "B300", "10", "1", "5", "2", "2"]
    # Verify four-bucket invariant
    assert int(rows[1][3]) + int(rows[1][4]) + int(rows[1][5]) + int(rows[1][6]) == int(rows[1][2])


def test_factory_status_slurm_estimate_is_capped_at_ready_slurm_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nodes = [
        _node("slurm-1", assigned_to="slurm"),
        _node("slurm-2", assigned_to="slurm"),
        _node("slurm-offline", state="offline", assigned_to="slurm"),
    ]
    _install(
        monkeypatch,
        _status_payload(),
        nodes_payload=_nodes_payload(("cluster-a", nodes)),
        workloads_payload=_workloads_payload(_slurm_job(64)),
    )

    # 3 nodes: 0 free, 3 slurm (all Slurm nodes), 0 busy, 0 unavailable
    assert _plain_status_rows()[1] == ["cluster-a", "B300", "3", "0", "3", "0", "0"]


def test_factory_status_verbose_pool_rows_add_up_to_the_cluster_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _status_payload(
        pools=[
            _pool("training", 4, 4, 0, 0),  # half a node still occupies one
            _pool("inference", 0, 0, 0, 0),
            _pool("slurm", 24, 16, 8, 0),
        ]
    )
    nodes = [
        _node("slurm-1", assigned_to="slurm"),
        _node("slurm-2", assigned_to="slurm"),
        _node("slurm-cordoned", state="cordoned", assigned_to="slurm"),
        _node("busy-1"),
        _node("busy-2"),
        _node("busy-3"),
        _node("idle", gpus_used=0),
        _node("offline", state="offline", gpus_used=0),
    ]
    jobs = [_slurm_job(8, scheduler="slurm-a"), _slurm_job(8, scheduler="slurm-b")]
    _install(
        monkeypatch,
        payload,
        nodes_payload=_nodes_payload(("cluster-a", nodes)),
        workloads_payload=_workloads_payload(*jobs),
    )

    rows = _plain_status_rows("--verbose")

    # The all-zero inference pool is omitted; one note closes the view.
    # Cluster: 8 nodes = 1 free + 3 slurm + 3 busy + 1 unavailable
    # Slurm pool shows four-bucket: 3 nodes, all in slurm bucket (internal breakdown separate)
    assert rows[1:6] == [
        ["cluster-a", "B300", "8", "1", "3", "3", "1"],  # nodes, free, slurm, busy, unavailable
        ["slurm", "(slurm-a,", "slurm-b)", "3", "0", "3", "0", "0"],  # Slurm: all in slurm bucket
        ["training", "1", "0", "0", "1", "0"],  # training: 1 busy node
        ["other", "2", "0", "0", "2", "0"],  # other: 2 busy nodes
        ["unassigned", "2", "1", "0", "0", "1"],  # unassigned: 1 free, 1 unavailable
    ]
    # Verify arithmetic: last 5 columns are nodes, free, slurm, busy, unavailable
    cluster = [int(cell) for cell in rows[1][-5:]]
    pools = [[int(cell) for cell in row[-5:]] for row in rows[2:6]]
    assert [sum(column) for column in zip(*pools)] == cluster
    # Verify four-bucket invariant for cluster and each pool
    assert all(nodes == free + slurm + busy + unavail 
               for nodes, free, slurm, busy, unavail in [cluster, *pools])
    assert len(rows) == 7 and "estimated from running jobs" in " ".join(rows[6])


def test_factory_status_warns_once_per_offline_or_stale_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clusters = [
        _status_payload(display_name=name, status=status)["clusters"][0]
        for name, status in (("healthy", "online"), ("dark", "offline"), ("stale", "online"))
    ]
    nodes_payload = _nodes_payload(*((c["display_name"], [_node("gpu-1")]) for c in clusters))
    nodes_payload["sources"][2] = _source("capacity", status="stale", age_seconds=2 * 3600)
    _install(monkeypatch, {**_status_payload(), "clusters": clusters}, nodes_payload=nodes_payload)

    result = runner.invoke(app, ["factory", "status", "--plain"], env=TEST_ENV)
    lines = result.output.splitlines()

    assert result.exit_code == 0, result.output
    # The stale cluster keeps its last-known counts; the warning says how old.
    assert lines[3].split() == ["[3]", "stale", "B300", "1", "1", "0", "0"]
    assert lines[4:] == ["[2] dark: offline", "[3] stale: node data last seen 2h ago"]


@pytest.mark.parametrize("endpoint", ["/factory/nodes", "/factory/workloads"])
def test_factory_status_nodes_or_workloads_failure_is_a_clean_error(
    monkeypatch: pytest.MonkeyPatch, endpoint: str
) -> None:
    class _FailingClient(_DummyAPIClient):
        def get(self, requested, params=None, timeout=None):
            if requested == endpoint:
                raise APIError("HTTP 503: unavailable")
            return super().get(requested, params, timeout)

    _install(monkeypatch, _status_payload())
    monkeypatch.setattr(
        "prime_cli.commands.factory.APIClient", lambda: _FailingClient(_status_payload())
    )

    result = runner.invoke(app, ["factory", "status"], env=TEST_ENV)

    assert result.exit_code == 1
    assert "HTTP 503: unavailable" in strip_ansi(result.output)
    assert "Traceback" not in result.output


def test_factory_status_json_makes_no_nodes_call(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _status_payload()
    dummy = _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "status", "--json"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert [call["endpoint"] for call in dummy.calls] == ["/factory/status"]
    assert json.loads(result.stdout) == payload


def test_factory_status_compact_duplicate_names_pair_positionally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Duplicate display names: node summaries must pair by position, not by
    # name — a name key would let the later cluster overwrite the earlier
    # one's counts.
    now = datetime.now(timezone.utc)
    status_payload = {
        "schema_version": 1,
        "as_of": _iso(now),
        "clusters": [
            {
                **_status_payload()["clusters"][0],
                "display_name": "twin",
                "pools": [
                    _pool("training", 32, 32, 0, 0),
                    _pool("inference", 32, 32, 0, 0),
                    _pool("slurm", 48, 16, 32, 0),
                ],
                "sources": [
                    _source("capacity"),
                    _source("training"),
                    _source("inference"),
                    _source("slurm"),
                ],
            },
            {
                **_status_payload()["clusters"][0],
                "display_name": "twin",
                "pools": [
                    _pool("training", 8, 8, 0, 0),
                ],
                "sources": [
                    _source("capacity"),
                    _source("training"),
                ],
            },
        ],
    }
    nodes_payload = {
        "schema_version": 1,
        "as_of": _iso(now),
        "sources": [_source("capacity"), _source("capacity")],
        "clusters": [
            {
                "display_name": "twin",
                "status": "online",
                "nodes": [
                    {
                        "name": "gpu-1",
                        "state": "ready",
                        "gpu_type": "B300",
                        "gpus_total": 8,
                        "gpus_used": 8,
                        "assigned_to": "slurm",
                    },
                    {
                        "name": "gpu-2",
                        "state": "ready",
                        "gpu_type": "B300",
                        "gpus_total": 8,
                        "gpus_used": 8,
                        "assigned_to": None,
                    },
                ],
            },
            {
                "display_name": "twin",
                "status": "online",
                "nodes": [
                    {
                        "name": "gpu-9",
                        "state": "cordoned",
                        "gpu_type": "B300",
                        "gpus_total": 8,
                        "gpus_used": 8,
                        "assigned_to": None,
                    },
                ],
            },
        ],
    }
    _install(monkeypatch, status_payload, nodes_payload=nodes_payload)

    result = runner.invoke(app, ["factory", "status"], env=TEST_ENV)
    output = strip_ansi(result.output)
    assert result.exit_code == 0, result.output
    # Duplicate display names make the equal-name identity check blind to
    # replacement between the two requests: both rows degrade to em-dashes
    # instead of risking cross-wired counts.
    assert output.count("—") == 8
    assert "[1] twin: node data unavailable" in output
    assert "[2] twin: node data unavailable" in output

    # index selection over duplicates degrades the same way
    selected = runner.invoke(app, ["factory", "status", "--cluster", "2"], env=TEST_ENV)
    selected_output = strip_ansi(selected.output)
    assert selected.exit_code == 0, selected.output
    assert selected_output.count("—") == 4
    assert "[2] twin: node data unavailable" in selected_output


def test_factory_nodes_help_documents_top_level_sources() -> None:
    # --help makes no API call; no fixture install needed.
    result = runner.invoke(app, ["factory", "nodes", "--help"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    # clusters[] carries no sources; the envelope documents them explicitly.
    assert "clusters[] = {display_name, status, nodes[]}" in result.output
    assert "clusters[] = {display_name, status, nodes[], sources[]}" not in result.output


def test_factory_status_compact_node_join_verifies_cluster_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The status and nodes responses are separate requests: when their
    # cluster lists cannot be aligned (drift between the requests), the
    # positional join must NOT display one cluster's node counts on
    # another cluster's row — the counts degrade to em-dashes instead.
    now = datetime.now(timezone.utc)
    status_payload = {
        "schema_version": 1,
        "as_of": _iso(now),
        "clusters": [
            {
                **_status_payload()["clusters"][0],
                "display_name": "cluster-a",
                "sources": [_source("capacity"), _source("training")],
                "pools": [_pool("training", 32, 32, 0, 0)],
            },
            {
                **_status_payload()["clusters"][0],
                "display_name": "office-a100",
                "sources": [_source("capacity"), _source("training")],
                "pools": [_pool("training", 8, 8, 0, 0)],
            },
        ],
    }
    # Nodes response drifted: reversed order vs the status clusters.
    nodes_payload = {
        "schema_version": 1,
        "as_of": _iso(now),
        "sources": [_source("capacity"), _source("capacity")],
        "clusters": [
            {
                "display_name": "office-a100",
                "status": "online",
                "nodes": [
                    {
                        "name": "a100-1",
                        "state": "ready",
                        "gpu_type": "A100",
                        "gpus_total": 4,
                        "gpus_used": 4,
                        "assigned_to": "slurm",
                    }
                ],
            },
            {
                "display_name": "cluster-a",
                "status": "online",
                "nodes": [
                    {
                        "name": "gpu-1",
                        "state": "ready",
                        "gpu_type": "B300",
                        "gpus_total": 8,
                        "gpus_used": 8,
                        "assigned_to": "slurm",
                    }
                ],
            },
        ],
    }
    _install(monkeypatch, status_payload, nodes_payload=nodes_payload)

    result = runner.invoke(app, ["factory", "status"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    # Position 1 pairs with office-a100's node data on cluster-a's row:
    # identity mismatch -> em-dash cells on both rows, never cross-wired
    # counts, and a warning line per cluster.
    assert output.count("—") == 8
    assert "[1] cluster-a: node data unavailable" in output
    assert "[2] office-a100: node data unavailable" in output


def test_factory_status_malformed_success_body_is_clean_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A 200 with a non-JSON body on a PRIMARY endpoint must surface as a
    # coarse APIError (exit 1), never an unhandled ValueError traceback.
    class _MalformedStatusClient:
        def get(self, endpoint, params=None, timeout=None):
            raise ValueError("Expecting value: line 1 column 1 (char 0)")

    monkeypatch.setattr("prime_cli.commands.factory.APIClient", lambda: _MalformedStatusClient())
    monkeypatch.setattr("prime_cli.commands.factory.Config", lambda: _StubConfig("team-123"))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    result = runner.invoke(app, ["factory", "status"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 1, result.output
    assert "malformed response body" in output
    assert "Traceback" not in result.output
    assert "ValueError" not in result.output


def test_factory_status_compact_multi_cluster_shows_original_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With multiple clusters, each compact row is prefixed with its
    # original 1-based index — the selector --cluster error messages direct
    # users to (and duplicate names make name selection impossible).
    now = datetime.now(timezone.utc)
    status_payload = {
        "schema_version": 1,
        "as_of": _iso(now),
        "clusters": [
            {
                **_status_payload()["clusters"][0],
                "display_name": "twin",
                "sources": [_source("capacity"), _source("training")],
                "pools": [_pool("training", 32, 32, 0, 0)],
            },
            {
                **_status_payload()["clusters"][0],
                "display_name": "twin",
                "sources": [_source("capacity"), _source("training")],
                "pools": [_pool("training", 8, 8, 0, 0)],
            },
        ],
    }
    _install(monkeypatch, status_payload)

    result = runner.invoke(app, ["factory", "status"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "[1] twin" in output and "[2] twin" in output

    # Index selection after filtering keeps the original indices visible.
    selected = runner.invoke(app, ["factory", "status", "--cluster", "2"], env=TEST_ENV)
    selected_output = strip_ansi(selected.output)
    assert selected.exit_code == 0, selected.output
    assert "[2] twin" in selected_output
    assert "[1] twin" not in selected_output


def test_factory_status_json_help_documents_legacy_pools_key() -> None:
    # --json is a byte-exact passthrough: against a legacy server the
    # envelope still carries `pools`. The advertised shape must mention
    # both keys so users do not query a missing field.
    result = runner.invoke(app, ["factory", "status", "--help"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert "allocations[]" in result.output
    assert "pools[]" in result.output


def test_api_client_omits_unset_timeout_to_preserve_default() -> None:
    # The client configures httpx.Timeout(30s, 10s connect); explicitly
    # forwarding timeout=None would disable it. request() must omit the
    # kwarg when unset and pass it through when provided.
    from prime_cli.core.client import APIClient

    api = APIClient(api_key="dummy", require_auth=False)
    recorded: Dict[str, Any] = {}

    class _FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True}

    def fake_request(method, url, **kwargs):
        recorded.update(kwargs)
        recorded["_url"] = url
        return _FakeResponse()

    monkey = pytest.MonkeyPatch()
    monkey.setattr(api.client, "request", fake_request)
    try:
        api.get("/factory/status", params={"team_id": "t"})
        assert "timeout" not in recorded  # client default stays in force

        api.get("/factory/nodes", params={"team_id": "t"}, timeout=5.0)
        assert recorded.get("timeout") == 5.0
    finally:
        monkey.undo()


def test_factory_status_compact_gpu_type_strips_memory_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The number after the underscore is noise next to a TOTAL column.
    payload = _status_payload(gpu_type="B300_262GB")
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "status"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "B300" in output
    assert "B300_262GB" not in output


def test_factory_status_compact_nodes_snapshot_duplicates_disable_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The status payload has ONE "twin" cluster, but the independently
    # fetched nodes snapshot carries TWO same-named clusters: the
    # positional name match cannot prove which counts belong to the status
    # row — the join is disabled, the counts dash, and a warning line
    # explains it.
    now = datetime.now(timezone.utc)
    status_payload = _status_payload(display_name="twin")
    nodes_payload = {
        "schema_version": 1,
        "as_of": _iso(now),
        "sources": [_source("capacity"), _source("capacity")],
        "clusters": [
            {
                "display_name": "twin",
                "status": "online",
                "nodes": [
                    {
                        "name": "gpu-1",
                        "state": "ready",
                        "gpu_type": "B300",
                        "gpus_total": 8,
                        "gpus_used": 8,
                        "assigned_to": "slurm",
                    }
                ],
            },
            {
                "display_name": "twin",
                "status": "online",
                "nodes": [
                    {
                        "name": "gpu-9",
                        "state": "ready",
                        "gpu_type": "B300",
                        "gpus_total": 8,
                        "gpus_used": 4,
                        "assigned_to": None,
                    }
                ],
            },
        ],
    }
    _install(monkeypatch, status_payload, nodes_payload=nodes_payload)

    result = runner.invoke(app, ["factory", "status"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert output.count("—") == 4  # never attach either duplicate's counts
    assert "twin: node data unavailable" in output


def test_factory_nodes_state_cell_renders_draining_dim() -> None:
    from prime_cli.api.factory import FactoryNode
    from prime_cli.commands.factory import _node_state_cell

    node = FactoryNode(
        name="gpu-01",
        state="draining",
        gpu_type="B300",
        gpus_total=8,
        gpus_used=8,
        assigned_to="slurm",
    )
    cell = _node_state_cell(node)
    assert "draining" in strip_ansi(cell)
    assert "[dim]" in cell  # lowercase dim, per the v1.5.2 user spec


def test_factory_nodes_state_filter_accepts_draining() -> None:
    from prime_cli.commands.factory import NODE_STATES

    assert "draining" in NODE_STATES
