"""Tests for `prime factory workloads` (squeue-style fleet job table)."""

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


def _workload(
    id_: str,
    type_: str,
    state: str = "running",
    **overrides: Any,
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    native_states = {
        "running": "RUNNING",
        "queued": "PENDING",
        "completed": "COMPLETED",
        "failed": "FAILED",
    }
    workload: Dict[str, Any] = {
        "id": id_,
        "type": type_,
        "cluster_display_name": "cluster-a",
        "name": f"{type_}-job",
        "state": state,
        "native_state": native_states.get(state, state.upper()),
        "owner": {"kind": "slurm" if type_ == "slurm" else "prime", "display_name": "carol"},
        "requested_gpus": 32,
        "allocated_gpus": 32 if state in ("running", "completed", "failed") else None,
        "created_at": _iso(now - timedelta(hours=2)),
        "started_at": _iso(now - timedelta(minutes=30)) if state == "running" else None,
        "ended_at": _iso(now - timedelta(minutes=5)) if state in ("completed", "failed") else None,
        "source": _source(type_),
    }
    workload.update(overrides)
    return workload


def _workloads_payload(
    workloads: List[Dict[str, Any]],
    sources: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    kinds: List[str] = []
    for workload in workloads:
        if workload["source"]["kind"] not in kinds:
            kinds.append(workload["source"]["kind"])
    return {
        "schema_version": 1,
        "as_of": _iso(datetime.now(timezone.utc)),
        "workloads": workloads,
        "sources": sources if sources is not None else [_source(k) for k in kinds],
    }


def _default_rows() -> List[Dict[str, Any]]:
    return [
        _workload("slurm:ac12:8421", "slurm", "running"),
        _workload("training:run-7", "training", "running"),
        _workload("inference:job-3", "inference", "queued"),
    ]


class _StubConfig:
    def __init__(self, team_id: Optional[str]) -> None:
        self._team_id = team_id

    @property
    def team_id(self) -> Optional[str]:
        return self._team_id


CLUSTERS = ("cluster-a", "office-a100")


def _status_payload() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "clusters": [
            {
                "display_name": name,
                "gpu_type": "B300_262GB",
                "allocations": [],
                "sources": [_source("capacity")],
            }
            for name in CLUSTERS
        ],
    }


def _nodes_payload(nodes: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    if nodes is None:
        nodes = [{"name": "gpu-1", "state": "ready", "gpus_total": 8, "gpus_used": 8}]
    return {
        "schema_version": 1,
        "clusters": [{"display_name": name, "nodes": nodes} for name in CLUSTERS],
        "sources": [_source("capacity") for _ in CLUSTERS],
    }


class _DummyAPIClient:
    def __init__(
        self, payload: Dict[str, Any], nodes_payload: Optional[Dict[str, Any]] = None
    ) -> None:
        self._payloads = {
            "/factory/workloads": payload,
            "/factory/status": _status_payload(),
            "/factory/nodes": nodes_payload or _nodes_payload(),
        }
        self.calls: List[Dict[str, Any]] = []

    def get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None, timeout: Any = None
    ) -> Dict[str, Any]:
        self.calls.append({"endpoint": endpoint, "params": params})
        return self._payloads[endpoint]

    def workloads_params(self) -> Optional[Dict[str, Any]]:
        return next(c["params"] for c in self.calls if c["endpoint"] == "/factory/workloads")


def _install(
    monkeypatch: pytest.MonkeyPatch,
    payload: Dict[str, Any],
    team_id: Optional[str] = "team-123",
    nodes_payload: Optional[Dict[str, Any]] = None,
) -> _DummyAPIClient:
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)
    # Freeze the command's clock just after the rows were built, so live
    # ages render to the exact second.
    frozen = datetime.now(timezone.utc)

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> datetime:
            return frozen

    monkeypatch.setattr("prime_cli.commands.factory.datetime", _FrozenDatetime)
    dummy = _DummyAPIClient(payload, nodes_payload)
    monkeypatch.setattr("prime_cli.commands.factory.APIClient", lambda: dummy)
    monkeypatch.setattr("prime_cli.commands.factory.Config", lambda: _StubConfig(team_id))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    return dummy


def _counts(free: int, slurm: int, busy: int, unavailable: int) -> List[str]:
    return ["·", str(free), "free", "·", str(slurm), "in", "slurm", "·", str(busy), "busy", "·", str(unavailable), "unavailable"]


def _plain_rows(*args: str) -> List[List[str]]:
    result = runner.invoke(app, ["factory", "workloads", "--plain", *args], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    return [line.split() for line in result.output.splitlines()]


def test_workloads_groups_jobs_by_cluster_and_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    main = _workload("slurm:1", "slurm", allocated_gpus=16, scheduler_display_name="slurm-a")
    dev = _workload("slurm:2", "slurm", allocated_gpus=8, scheduler_display_name="slurm-b")
    training = _workload("training:1", "training")
    inference = _workload("inference:1", "inference", "queued", requested_gpus=8)
    inference["cluster_display_name"] = "office-a100"
    _install(monkeypatch, _workloads_payload([inference, training, main, dev]))

    # One table per cluster in `prime factory status` order, titled like
    # the status row; one block per pool (Slurm clusters by name, then
    # training, inference); the ID last.
    # Default nodes: 1 ready direct node with gpus_used=8 (busy)
    counts = _counts(0, 0, 1, 0)  # free, slurm, busy, unavailable
    assert _plain_rows() == [
        ["[1]", "cluster-a", "·", "B300", *counts],
        ["NAME", "USER", "STATE", "NODES", "AGE", "ID"],
        ["slurm", "(slurm-a)"],
        ["slurm-job", "carol", "running", "2", "30:00", "slurm:1"],
        ["slurm", "(slurm-b)"],
        ["slurm-job", "carol", "running", "1", "30:00", "slurm:2"],
        ["training"],
        ["training-job", "carol", "running", "4", "30:00", "training:1"],
        ["other", *counts],
        [],
        ["[2]", "office-a100", "·", "B300", *counts],
        ["NAME", "USER", "STATE", "NODES", "AGE", "ID"],
        ["inference"],
        ["inference-job", "carol", "queued", "1", "2:00:00", "inference:1"],
        ["other", *counts],
    ]
    output = runner.invoke(app, ["factory", "workloads", "--plain"], env=TEST_ENV).output
    headers = [line for line in output.splitlines() if line.startswith("NAME")]
    assert headers[0] == headers[1]  # the tables share column widths


def test_workloads_nodes_column_and_order_within_a_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime.now(timezone.utc)
    rows = [
        _workload(
            "slurm:16", "slurm", allocated_gpus=16, started_at=_iso(now - timedelta(hours=2))
        ),
        _workload(
            "slurm:0",
            "slurm",
            allocated_gpus=0,
            requested_gpus=0,
            started_at=_iso(now - timedelta(hours=3)),
        ),
        _workload("slurm:q64", "slurm", "queued", requested_gpus=64),
        _workload("slurm:12", "slurm", allocated_gpus=12, requested_gpus=12),
        _workload(
            "slurm:4",
            "slurm",
            allocated_gpus=4,
            requested_gpus=4,
            started_at=_iso(now - timedelta(hours=1)),
        ),
    ]
    _install(monkeypatch, _workloads_payload(rows))

    # Like squeue: queued before running, then the newest first, whatever
    # the size. NODES: GPUs / 8 rounded up, GPUs below one node, cpu for
    # none; queued jobs count their request.
    assert _plain_rows()[3:8] == [
        ["slurm-job", "carol", "queued", "8", "2:00:00", "slurm:q64"],
        ["slurm-job", "carol", "running", "2", "30:00", "slurm:12"],
        ["slurm-job", "carol", "running", "4", "gpu", "1:00:00", "slurm:4"],
        ["slurm-job", "carol", "running", "2", "2:00:00", "slurm:16"],
        ["slurm-job", "carol", "running", "cpu", "3:00:00", "slurm:0"],
    ]


USAGE_NODES = [
    *(
        {
            "name": f"slurm-{i}",
            "state": "ready",
            "gpus_total": 8,
            "gpus_used": 8,
            "assigned_to": "slurm",
        }
        for i in range(3)
    ),
    {
        "name": "slurm-down",
        "state": "offline",
        "gpus_total": 8,
        "gpus_used": 0,
        "assigned_to": "slurm",
    },
    {"name": "busy", "state": "ready", "gpus_total": 8, "gpus_used": 8},
    {"name": "idle", "state": "ready", "gpus_total": 8, "gpus_used": 0},
]


def test_workloads_titles_and_pool_headers_count_nodes_like_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs = [
        _workload("slurm:1", "slurm", allocated_gpus=16, scheduler_display_name="slurm-a"),
        _workload("slurm:2", "slurm", allocated_gpus=0, scheduler_display_name="slurm-b"),
    ]
    _install(monkeypatch, _workloads_payload(jobs), nodes_payload=_nodes_payload(USAGE_NODES))

    status = runner.invoke(app, ["factory", "status", "--plain"], env=TEST_ENV)
    rows = _plain_rows()

    # The title carries the status row's FREE / SLURM / BUSY / UNAVAILABLE.
    # USAGE_NODES: 1 idle, 4 slurm, 1 busy, 0 unavailable
    assert status.output.splitlines()[1].split() == ["[1]", "cluster-a", "B300", "6", "1", "4", "1", "0"]
    assert rows[0] == ["[1]", "cluster-a", "·", "B300", *_counts(1, 4, 1, 0)]
    # All Slurm nodes count on one "slurm" row: the API cannot split them
    # between Slurm clusters, so the named groups carry no counts. Pools
    # without listed jobs close the table so every node is accounted for.
    assert [row[:2] for row in rows[2:7]] == [
        ["slurm", "·"],
        ["slurm", "(slurm-a)"],
        ["slurm-job", "carol"],
        ["slurm", "(slurm-b)"],
        ["slurm-job", "carol"],
    ]
    # Slurm pool header shows allocated/idle/draining (internal Slurm breakdown)
    assert rows[2] == ["slurm", "·", "2", "allocated", "·", "1", "idle", "·", "1", "draining"]
    # Other pools show four-bucket format
    assert rows[7:9] == [["other", *_counts(0, 0, 1, 0)], ["unassigned", *_counts(1, 0, 0, 0)]]
    assert "slurm allocated/idle = internal Slurm breakdown" in " ".join(rows[-1])


def test_workloads_slurm_counts_head_the_table_without_slurm_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(
        monkeypatch,
        _workloads_payload([_workload("training:1", "training")]),
        nodes_payload=_nodes_payload(USAGE_NODES),
    )

    rows = _plain_rows("--type", "training")

    # Slurm pool with no jobs: 0 allocated, 3 idle, 1 draining
    assert rows[2:4] == [["slurm", "·", "0", "allocated", "·", "3", "idle", "·", "1", "draining"], ["training"]]


def test_workloads_warns_when_slurm_jobs_are_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    # Jobs cut from the running Slurm list would count their nodes as idle.
    payload = _workloads_payload([_workload("slurm:1", "slurm")])
    payload["truncated"] = True
    _install(monkeypatch, payload, nodes_payload=_nodes_payload(USAGE_NODES))

    output = " ".join(" ".join(row) for row in _plain_rows())

    assert "allocated Slurm nodes may be undercounted" in output


def test_workloads_shows_idle_clusters_with_their_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    job = _workload("slurm:1", "slurm", scheduler_display_name="slurm-a")
    _install(monkeypatch, _workloads_payload([job]), nodes_payload=_nodes_payload(USAGE_NODES))

    rows = _plain_rows()

    # office-a100 has no jobs: its title and counts still show.
    # USAGE_NODES: free=1, slurm=4, busy=1, unavailable=0
    assert rows[rows.index([]) + 1 :][:2] == [
        ["[2]", "office-a100", "·", "B300", *_counts(1, 4, 1, 0)],
        ["no", "running", "or", "queued", "jobs"],
    ]
    # A filtered view shows only clusters with matching jobs.
    assert "office-a100" not in str(_plain_rows("--type", "slurm"))


def test_workloads_long_job_names_are_shortened_in_the_middle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    long_name = "a-very-long-experiment-name-with-many-parts-and-a-retry-suffix-retry1"
    forty = "nightly-eval-sweep-checkpoint-0042-final"
    rows = [
        _workload("slurm:1", "slurm", name=long_name),
        _workload("slurm:2", "slurm", name=forty),
    ]
    _install(monkeypatch, _workloads_payload(rows))

    names = {row[-1]: row[0] for row in _plain_rows() if row and row[-1].startswith("slurm:")}

    assert names == {"slurm:1": "a-very-long-experime…retry-suffix-retry1", "slurm:2": forty}
    json_result = runner.invoke(app, ["factory", "workloads", "--json"], env=TEST_ENV)
    assert json.loads(json_result.stdout)["workloads"][0]["name"] == long_name


def test_workloads_default_view_hides_unknown_state_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    stale = _workload("training:stale", "training", "unknown", allocated_gpus=0)
    _install(monkeypatch, _workloads_payload([_workload("slurm:1", "slurm"), stale]))

    assert "training:stale" not in str(_plain_rows())
    assert "training:stale" in str(_plain_rows("--state", "all"))


def test_workloads_age_is_run_age_for_running_and_wait_age_for_queued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _workloads_payload(_default_rows()))

    result = runner.invoke(app, ["factory", "workloads"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    # started 30m ago (running rows) vs. created 2h ago (queued row)
    assert "30:00" in output
    assert "2:00:00" in output


def test_workloads_json_prints_exact_api_response(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _workloads_payload(_default_rows())
    dummy = _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "workloads", "--json"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == payload
    # Without --cluster, JSON needs nothing but the workloads endpoint.
    assert [call["endpoint"] for call in dummy.calls] == ["/factory/workloads"]


def test_workloads_output_json_flag_matches_json_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _workloads_payload(_default_rows())
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "workloads", "--output", "json"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == payload


def test_workloads_empty_response_prints_friendly_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty = {
        "schema_version": 1,
        "as_of": _iso(datetime.now(timezone.utc)),
        "workloads": [],
        "sources": [],
    }
    # Clusters without nodes have no idle capacity to show either.
    _install(monkeypatch, empty, nodes_payload=_nodes_payload([]))

    result = runner.invoke(app, ["factory", "workloads"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "No factory workloads found." in output


def test_workloads_empty_response_json_keeps_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty = {
        "schema_version": 1,
        "as_of": _iso(datetime.now(timezone.utc)),
        "workloads": [],
        "sources": [],
    }
    _install(monkeypatch, empty)

    result = runner.invoke(app, ["factory", "workloads", "--json"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["workloads"] == []


def test_workloads_no_team_selected_prints_switch_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy = _install(monkeypatch, _workloads_payload(_default_rows()), team_id=None)

    result = runner.invoke(app, ["factory", "workloads"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "prime switch" in output
    assert not dummy.calls


def test_workloads_team_flag_overrides_missing_team_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy = _install(monkeypatch, _workloads_payload(_default_rows()), team_id=None)

    result = runner.invoke(app, ["factory", "workloads", "--team", "team-42"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert dummy.calls and dummy.workloads_params() == {"team_id": "team-42"}


def test_workloads_type_and_state_filters_are_server_side(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy = _install(monkeypatch, _workloads_payload(_default_rows()))

    result = runner.invoke(
        app,
        ["factory", "workloads", "--type", "slurm", "--state", "queued"],
        env=TEST_ENV,
    )
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert dummy.workloads_params() == {
        "team_id": "team-123",
        "workload_type": "slurm",
        "state": "queued",
    }
    # The (stubbed) response is rendered as-is; the filter itself was the
    # server's job, so the unfiltered stub rows still appear.
    assert "slurm:ac12:8421" in output


@pytest.mark.parametrize("flag,value", [("--type", "gpu"), ("--state", "zombie")])
def test_workloads_invalid_filter_values_exit_nonzero(
    monkeypatch: pytest.MonkeyPatch, flag: str, value: str
) -> None:
    dummy = _install(monkeypatch, _workloads_payload(_default_rows()))

    result = runner.invoke(app, ["factory", "workloads", flag, value], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 1, result.output
    assert "Invalid" in output
    assert not dummy.calls


def test_workloads_user_filter_matches_owner_display_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _default_rows()
    rows[1]["owner"] = {"kind": "prime", "display_name": "alice"}
    _install(monkeypatch, _workloads_payload(rows))

    result = runner.invoke(app, ["factory", "workloads", "--user", "carol"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "carol" in output
    assert "training:run-7" not in output  # alice's row filtered out
    assert "training-job" not in output


def test_workloads_user_filter_applies_to_json_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _default_rows()
    rows[1]["owner"] = {"kind": "prime", "display_name": "alice"}
    payload = _workloads_payload(rows)
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "workloads", "--user", "alice", "--json"], env=TEST_ENV)
    data = json.loads(result.stdout)

    assert result.exit_code == 0, result.output
    assert [w["id"] for w in data["workloads"]] == ["training:run-7"]
    assert data["schema_version"] == 1


def test_workloads_cluster_filter_and_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = _default_rows()
    rows[1]["cluster_display_name"] = "office-a100"
    _install(monkeypatch, _workloads_payload(rows))

    for selector in ("office-a100", "2"):
        result = runner.invoke(app, ["factory", "workloads", "--cluster", selector], env=TEST_ENV)
        output = strip_ansi(result.output)
        assert result.exit_code == 0, result.output
        assert "training:run-7" in output
        assert "slurm:ac12:8421" not in output

    miss = runner.invoke(app, ["factory", "workloads", "--cluster", "nope"], env=TEST_ENV)
    miss_output = strip_ansi(miss.output)
    assert miss.exit_code == 1, miss.output
    assert "No cluster matched" in miss_output
    assert "Available clusters" in miss_output


def test_workloads_stale_source_suppresses_its_rows_with_friendly_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _default_rows()
    sources = [
        _source("training"),
        _source("inference"),
        _source("slurm", status="stale", age_seconds=7200),
    ]
    for row in rows:
        if row["type"] == "slurm":
            row["source"] = _source("slurm", status="stale", age_seconds=7200)
    _install(monkeypatch, _workloads_payload(rows, sources))

    result = runner.invoke(app, ["factory", "workloads"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "slurm jobs unavailable" in output
    assert "scheduler data last seen 2h ago" in output
    # slurm rows suppressed; fresh rows still render
    assert "slurm:ac12:8421" not in output
    assert "training:run-7" in output
    assert "inference:job-3" in output


def test_workloads_all_stale_sources_skip_table_entirely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _default_rows()
    sources = [
        _source("training", status="error"),
        _source("inference", status="error"),
        _source("slurm", status="stale", age_seconds=86400),
    ]
    status_by_kind = {s["kind"]: s["status"] for s in sources}
    for row in rows:
        row["source"] = _source(row["type"], status=status_by_kind[row["type"]])
    _install(monkeypatch, _workloads_payload(rows, sources))

    result = runner.invoke(app, ["factory", "workloads"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "ID" not in output  # no table
    assert "unavailable" in output
    assert "training jobs unavailable" in output
    assert "inference jobs unavailable" in output
    assert "slurm jobs unavailable" in output
    assert "No factory workloads found." not in output


def test_workloads_stale_source_with_zero_rows_still_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A degraded source with no rows must not masquerade as a quiet fleet.
    payload = _workloads_payload(
        [_workload("training:run-7", "training", "running")],
        sources=[_source("training"), _source("slurm", status="stale", age_seconds=3600)],
    )
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "workloads"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "slurm jobs unavailable" in output
    assert "training:run-7" in output


def test_workloads_rich_markup_in_dynamic_values_is_escaped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _workload(
        "slurm:[weird]:1",
        "slurm",
        "running",
        name="eval [bold] run",
        native_state="[RUN]",
        reason="priority [P]",
    )
    row["owner"] = {"kind": "slurm", "display_name": "[red]evil"}
    _install(monkeypatch, _workloads_payload([row]))

    result = runner.invoke(app, ["factory", "workloads"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "[bold]" in output and "[red]" in output and "[P]" in output
    assert "REASON" in output


def test_workloads_reason_column_hidden_when_never_supplied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _workloads_payload(_default_rows()))

    result = runner.invoke(app, ["factory", "workloads"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "REASON" not in output


def test_workloads_unsupported_schema_version_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _workloads_payload(_default_rows())
    payload["schema_version"] = 2
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "workloads"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 1, result.output
    assert "Unsupported factory workloads schema version" in output


def test_workloads_null_values_render_dashes_never_zeros(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _workload(
        "slurm:ac12:8422",
        "slurm",
        "queued",
        name=None,
        allocated_gpus=None,
        requested_gpus=None,
        created_at=_iso(datetime.now(timezone.utc) - timedelta(minutes=5)),
    )
    row["owner"] = {"kind": "unknown", "display_name": None}
    _install(monkeypatch, _workloads_payload([row]))

    result = runner.invoke(app, ["factory", "workloads"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "5:00" in output
    # Unobserved GPUs render "-", never a fabricated zero ("cpu").
    assert "cpu" not in output
    assert "None" not in output


def test_factory_client_workloads_endpoint_and_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy = _DummyAPIClient(_workloads_payload(_default_rows()))
    client = FactoryClient(dummy)  # type: ignore[arg-type]

    client.get_workloads("team-9", workload_type="slurm", state="running")

    assert dummy.calls == [
        {
            "endpoint": "/factory/workloads",
            "params": {"team_id": "team-9", "workload_type": "slurm", "state": "running"},
        }
    ]


def test_workloads_queued_age_counts_from_created_at(monkeypatch: pytest.MonkeyPatch) -> None:
    # Slurm reports a pending job's start ESTIMATE as started_at (seen
    # ~8729h ahead), and a requeued row keeps its old start: the wait age
    # comes from created_at.
    now = datetime.now(timezone.utc)
    estimate = _workload("slurm:1", "slurm", "queued", started_at=_iso(now + timedelta(hours=8729)))
    requeued = _workload(
        "slurm:2",
        "slurm",
        "queued",
        started_at=_iso(now - timedelta(minutes=30)),
        created_at=_iso(now - timedelta(hours=3)),
    )
    _install(monkeypatch, _workloads_payload([estimate, requeued]))

    ages = {row[-1]: row[-2] for row in _plain_rows() if row and row[-1].startswith("slurm:")}

    assert ages == {"slurm:1": "2:00:00", "slurm:2": "3:00:00"}


def test_workloads_live_age_reads_like_squeue_time(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime.now(timezone.utc)
    rows = [
        _workload("slurm:1", "slurm", started_at=_iso(now - timedelta(minutes=2, seconds=52))),
        _workload(
            "slurm:2", "slurm", started_at=_iso(now - timedelta(hours=5, minutes=34, seconds=21))
        ),
        _workload(
            "slurm:3",
            "slurm",
            started_at=_iso(now - timedelta(days=3, hours=9, minutes=54, seconds=8)),
        ),
    ]
    _install(monkeypatch, _workloads_payload(rows))

    ages = {row[-1]: row[-2] for row in _plain_rows() if row and row[-1].startswith("slurm:")}

    assert ages == {"slurm:1": "2:52", "slurm:2": "5:34:21", "slurm:3": "3-09:54:08"}


def test_workloads_filtered_empty_distinguished_from_genuinely_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # --type/--state are server-side: a zero-row response under filters is
    # "filters matched nothing", not "the team has no workloads".
    empty = {
        "schema_version": 1,
        "as_of": _iso(datetime.now(timezone.utc)),
        "workloads": [],
        "sources": [],
    }
    _install(monkeypatch, empty)

    filtered = runner.invoke(app, ["factory", "workloads", "--type", "slurm"], env=TEST_ENV)
    output = strip_ansi(filtered.output)
    assert filtered.exit_code == 0, filtered.output
    assert "No factory workloads match the given filters." in output
    assert "No factory workloads found." not in output

    _install(monkeypatch, empty, nodes_payload=_nodes_payload([]))
    plain = runner.invoke(app, ["factory", "workloads"], env=TEST_ENV)
    plain_output = strip_ansi(plain.output)
    assert plain.exit_code == 0, plain.output
    assert "No factory workloads found." in plain_output
    assert "match the given filters" not in plain_output


def test_workloads_newest_degraded_row_source_per_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    # Envelope omits the slurm entry; several degraded rows provide fallback
    # evidence. The warning must use the NEWEST observation for the kind,
    # not the first row's.
    rows = _default_rows()
    for row in rows:
        row["source"] = _source("slurm", status="stale", age_seconds=5 * 3600)
    rows[2]["source"] = _source("slurm", status="stale", age_seconds=120)  # newest
    payload = _workloads_payload(rows, sources=[_source("training"), _source("inference")])

    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "workloads"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "slurm jobs unavailable" in output
    assert "scheduler data last seen 2m ago" in output
    assert "5h ago" not in output


def test_workloads_degraded_fallback_prefers_observed_over_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A first degraded row without observed_at never beats an observed one.
    rows = _default_rows()
    for row in rows:
        row["source"] = {"kind": "slurm", "status": "stale", "observed_at": None}
    rows[2]["source"] = _source("slurm", status="stale", age_seconds=180)
    payload = _workloads_payload(rows, sources=[_source("training"), _source("inference")])

    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "workloads"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "scheduler data last seen 3m ago" in output
    assert "unavailable" in output


def test_workloads_fresh_row_renders_despite_global_worst_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Roast: per-row freshness, not global-worst-source erasure. The
    # envelope training source is stale (some other deployment), but this
    # training row's own evidence is fresh -> it must render.
    row = _workload("training:run-1", "training", "running")
    row["source"] = _source("training")  # fresh per-row evidence
    bob = _workload("training:run-2", "training", "running")
    bob["source"] = _source("training", status="stale", age_seconds=7200)
    payload = _workloads_payload(
        [row, bob],
        sources=[_source("training", status="stale", age_seconds=7200)],
    )
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "workloads"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "training:run-1" in output  # fresh row survives the stale aggregate
    assert "training:run-2" not in output  # the stale row itself is suppressed
    # Mixed coverage (fresh training rows render alongside the degraded
    # aggregate): partial-data wording, no age.
    assert "some training job data is unavailable" in output
    assert "results may be incomplete" in output
    assert "training data last seen 2h ago" not in output


def test_workloads_unobserved_allocation_row_renders_with_dash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A running row whose GPU allocation is unobserved still renders:
    # identity, state, and owner are known facts; NODES is "-", never a
    # guess from the request.
    row = _workload("training:run-5", "training", "running", requested_gpus=64)
    row["allocated_gpus"] = None
    _install(monkeypatch, _workloads_payload([row]))

    rows = _plain_rows()

    assert ["training-job", "carol", "running", "-", "30:00", "training:run-5"] in rows
    assert all("unavailable" not in row for row in rows)  # not degradation


def test_workloads_degradation_computed_on_narrowed_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # --user narrowing: degradation follows the narrowed view, not the
    # global fleet. Alice has fresh training rows; bob's stale slurm rows
    # are outside her view and must not warn on it.
    alice = _workload("training:run-1", "training", "running")
    alice["owner"] = {"kind": "prime", "display_name": "alice"}
    bob = _workload("slurm:ac12:8421", "slurm", "running")
    bob["owner"] = {"kind": "slurm", "display_name": "bob"}
    bob["source"] = _source("slurm", status="stale", age_seconds=7200)
    payload = _workloads_payload(
        [alice, bob],
        sources=[_source("training"), _source("slurm", status="stale", age_seconds=7200)],
    )
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "workloads", "--user", "alice"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "training:run-1" in output  # alice's fresh row renders
    assert "slurm:ac12:8421" not in output
    # bob's stale slurm evidence is not part of alice's narrowed view
    assert "slurm jobs unavailable" not in output

    # without narrowing, bob's stale row warns and is suppressed
    fleet = runner.invoke(app, ["factory", "workloads"], env=TEST_ENV)
    fleet_output = strip_ansi(fleet.output)
    assert "slurm jobs unavailable" in fleet_output
    assert "training:run-1" in fleet_output  # healthy peer still renders


def test_workloads_degraded_aggregate_with_only_fresh_rows_still_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Envelope reports the kind degraded while every RETURNED row of that
    # kind has fresh row-level evidence: the degraded portion produced no
    # rows. The aggregate warning must surface — missing workloads would
    # otherwise vanish silently — while the fresh rows still render.
    rows = _default_rows()  # training, inference, slurm rows, all fresh
    payload = _workloads_payload(
        rows,
        sources=[
            _source("training", status="stale", age_seconds=2 * 3600),
            _source("inference"),
            _source("slurm"),
        ],
    )
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "workloads"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "training:run-7" in output  # fresh rows still render
    # Mixed coverage: partial-data wording above healthy rows, no age.
    assert "some training job data is unavailable" in output
    assert "results may be incomplete" in output
    assert "training data last seen 2h ago" not in output


def test_workloads_type_filter_excludes_other_kinds_from_zero_row_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # --type is a server-side filter: with --type training, the absence of
    # slurm rows is expected filtering — a degraded slurm source must NOT
    # emit a misleading "slurm jobs unavailable" warning.
    rows = [_workload("training:run-1", "training", "running")]
    payload = _workloads_payload(
        rows,
        sources=[
            _source("training"),
            _source("slurm", status="stale", age_seconds=2 * 3600),
        ],
    )
    _install(monkeypatch, payload)

    filtered = runner.invoke(app, ["factory", "workloads", "--type", "training"], env=TEST_ENV)
    output = strip_ansi(filtered.output)
    assert filtered.exit_code == 0, filtered.output
    assert "training:run-1" in output
    assert "slurm jobs unavailable" not in output

    # Without --type, the degraded slurm source with zero slurm rows is a
    # failed read and must warn.
    plain = runner.invoke(app, ["factory", "workloads"], env=TEST_ENV)
    plain_output = strip_ansi(plain.output)
    assert plain.exit_code == 0, plain.output
    assert "training:run-1" in plain_output
    assert "slurm jobs unavailable" in plain_output


def test_workloads_malformed_success_body_is_clean_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _MalformedClient:
        def get(self, endpoint, params=None, timeout=None):
            raise ValueError("Expecting value: line 1 column 1 (char 0)")

    monkeypatch.setattr("prime_cli.commands.factory.APIClient", lambda: _MalformedClient())
    monkeypatch.setattr("prime_cli.commands.factory.Config", lambda: _StubConfig("team-123"))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    result = runner.invoke(app, ["factory", "workloads"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 1, result.output
    assert "malformed response body" in output
    assert "ValueError" not in result.output


def test_workloads_json_filter_does_not_collapse_duplicate_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two rows share an id but have different owners: --user must keep
    # exactly the matching raw row, not pull the twin in via the shared id.
    row_a = _workload("shared-id", "slurm", "running")
    row_a["owner"] = {"kind": "slurm", "display_name": "alice"}
    row_b = dict(row_a)
    row_b["owner"] = {"kind": "slurm", "display_name": "bob"}
    payload = _workloads_payload([row_a, row_b])
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "workloads", "--user", "alice", "--json"], env=TEST_ENV)
    data = json.loads(result.stdout)

    assert result.exit_code == 0, result.output
    assert len(data["workloads"]) == 1
    assert data["workloads"][0]["owner"]["display_name"] == "alice"


def test_workloads_terminal_state_renders_duration_and_ended_age(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A completed run 25 minutes ago that ran for 25 minutes: DURATION
    # column present, AGE ended-relative ("25m ago"), STATE terminal.
    row = _workload("training:run-1", "training", "completed")
    row["started_at"] = _iso(datetime.now(timezone.utc) - timedelta(minutes=50))
    row["ended_at"] = _iso(datetime.now(timezone.utc) - timedelta(minutes=25))
    _install(monkeypatch, _workloads_payload([row]))

    result = runner.invoke(app, ["factory", "workloads", "--state", "completed"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "DURATION" in output
    assert "25m ago" in output
    assert "25m" in output  # duration span
    assert "completed" in output
    # the live-view columns are unchanged
    assert "NODES" in output and "AGE" in output


def test_workloads_default_view_has_no_duration_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _workloads_payload(_default_rows()))

    result = runner.invoke(app, ["factory", "workloads"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "DURATION" not in output
    assert "ago" not in output  # live ages stay bare, e.g. "30:00"


def test_workloads_state_all_shows_duration_and_default_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _workload("training:run-1", "training", "completed")
    dummy = _install(monkeypatch, _workloads_payload([row]))

    result = runner.invoke(app, ["factory", "workloads", "--state", "all"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "DURATION" in output
    # terminal queries page by default
    assert dummy.workloads_params()["limit"] == 50


def test_workloads_live_state_sends_no_default_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy = _install(monkeypatch, _workloads_payload(_default_rows()))

    result = runner.invoke(app, ["factory", "workloads", "--state", "running"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert "limit" not in dummy.workloads_params()


def test_workloads_since_parsing_forms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Nd form
    dummy = _install(monkeypatch, _workloads_payload(_default_rows()))
    result = runner.invoke(app, ["factory", "workloads", "--since", "2d"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    since_value = dummy.workloads_params()["since"]
    parsed = datetime.fromisoformat(since_value.replace("Z", "+00:00"))
    assert abs((datetime.now(timezone.utc) - parsed).total_seconds() - 2 * 86400) < 60

    # Nh form
    _install(monkeypatch, _workloads_payload(_default_rows()))
    result = runner.invoke(app, ["factory", "workloads", "--since", "3h"], env=TEST_ENV)
    assert result.exit_code == 0, result.output

    # ISO form
    _install(monkeypatch, _workloads_payload(_default_rows()))
    iso = "2026-10-09T00:00:00Z"
    result = runner.invoke(app, ["factory", "workloads", "--since", iso], env=TEST_ENV)
    assert result.exit_code == 0, result.output

    # bad input: plain error, exit 1
    _install(monkeypatch, _workloads_payload(_default_rows()))
    bad = runner.invoke(app, ["factory", "workloads", "--since", "yesterday-ish"], env=TEST_ENV)
    bad_output = strip_ansi(bad.output)
    assert bad.exit_code == 1, bad.output
    assert "Invalid --since" in bad_output
    assert "Use Nd, Nh, or an ISO 8601 timestamp." in bad_output


def test_workloads_limit_bounding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy = _install(monkeypatch, _workloads_payload(_default_rows()))
    ok = runner.invoke(
        app, ["factory", "workloads", "--state", "completed", "--limit", "200"], env=TEST_ENV
    )
    assert ok.exit_code == 0, ok.output
    assert dummy.workloads_params()["limit"] == 200

    _install(monkeypatch, _workloads_payload(_default_rows()))
    too_big = runner.invoke(app, ["factory", "workloads", "--limit", "500"], env=TEST_ENV)
    too_big_output = strip_ansi(too_big.output)
    assert too_big.exit_code == 1, too_big.output
    assert "between 1 and 200" in too_big_output

    _install(monkeypatch, _workloads_payload(_default_rows()))
    zero = runner.invoke(app, ["factory", "workloads", "--limit", "0"], env=TEST_ENV)
    assert zero.exit_code == 1, zero.output


def test_workloads_null_ended_at_renders_dashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A terminal row without ended_at: AGE and DURATION stay dashes —
    # never invented times.
    row = _workload("training:run-1", "training", "failed")
    row["started_at"] = _iso(datetime.now(timezone.utc) - timedelta(hours=1))
    row["ended_at"] = None
    _install(monkeypatch, _workloads_payload([row]))

    result = runner.invoke(app, ["factory", "workloads", "--state", "failed"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "failed" in output
    assert " ago" not in output  # no invented ended age


def test_workloads_user_cluster_filters_work_over_terminal_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _workload("training:run-1", "training", "completed"),
        _workload("training:run-2", "training", "completed"),
    ]
    rows[1]["owner"] = {"kind": "prime", "display_name": "alice"}
    rows[1]["cluster_display_name"] = "office-a100"
    _install(monkeypatch, _workloads_payload(rows))

    result = runner.invoke(
        app,
        [
            "factory",
            "workloads",
            "--state",
            "completed",
            "--user",
            "alice",
            "--cluster",
            "office-a100",
        ],
        env=TEST_ENV,
    )
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "training:run-2" in output
    assert "training:run-1" not in output


def test_workloads_since_filtered_empty_uses_filter_wording(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A zero-row --since window is the filter matching nothing, not an
    # empty fleet.
    empty = {
        "schema_version": 1,
        "as_of": _iso(datetime.now(timezone.utc)),
        "workloads": [],
        "sources": [],
    }
    _install(monkeypatch, empty)

    result = runner.invoke(app, ["factory", "workloads", "--since", "2d"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "No factory workloads match the given filters." in output
    assert "No factory workloads found." not in output


def test_workloads_since_overflow_is_plain_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # A syntactically valid but out-of-range relative window must render the
    # normal invalid---since diagnostic, never an OverflowError traceback.
    _install(monkeypatch, _workloads_payload(_default_rows()))

    result = runner.invoke(app, ["factory", "workloads", "--since", "1000000000d"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 1, result.output
    assert "Invalid --since" in output
    assert "Use Nd, Nh, or an ISO 8601 timestamp." in output
    assert "OverflowError" not in result.output
    assert "Traceback" not in result.output


def test_workloads_json_help_documents_terminal_fields() -> None:
    # Integrations built from --help must not miss the terminal timestamp
    # and the backend-supplied reason.
    result = runner.invoke(app, ["factory", "workloads", "--help"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert "ended_at" in result.output
    assert "reason" in result.output


def test_workloads_slurm_cluster_view_scopes_degraded_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A selected Slurm cluster scopes the view to Slurm: an unrelated
    # training outage stays quiet, and a stale Slurm read still warns even
    # when the other filters leave no rows.
    row = _workload("slurm:1", "slurm", scheduler_display_name="slurm-a")
    training_down = _workloads_payload(
        [row], sources=[_source("slurm"), _source("training", status="stale", age_seconds=7200)]
    )
    _install(monkeypatch, training_down)
    output = strip_ansi(
        runner.invoke(app, ["factory", "workloads", "--cluster", "slurm-a"], env=TEST_ENV).output
    )
    assert "slurm:1" in output
    assert "training jobs unavailable" not in output

    slurm_stale = _workloads_payload(
        [row], sources=[_source("slurm", status="stale", age_seconds=7200)]
    )
    _install(monkeypatch, slurm_stale)
    result = runner.invoke(
        app, ["factory", "workloads", "--cluster", "slurm-a", "--user", "nobody"], env=TEST_ENV
    )
    output = strip_ansi(result.output)
    assert "slurm jobs unavailable" in output
    assert "No factory workloads match" not in output


def test_workloads_slurm_cluster_name_escaped(monkeypatch: pytest.MonkeyPatch) -> None:
    # Rich markup in the Slurm cluster name must never crash rendering.
    row = _workload("slurm:ac12:8421", "slurm", "running")
    row["scheduler_display_name"] = "slurm-[bold]team"
    _install(monkeypatch, _workloads_payload([row]))

    result = runner.invoke(app, ["factory", "workloads"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "slurm-[bold]team" in output


def test_workloads_scheduler_null_slurm_renders_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A failed Slurm deployment lookup still lands in a Slurm pool, labeled
    # "unknown" — never mistaken for direct placement.
    row = _workload("slurm:ac12:8422", "slurm", "completed")
    row["scheduler_display_name"] = None
    _install(monkeypatch, _workloads_payload([row]))

    result = runner.invoke(app, ["factory", "workloads", "--state", "completed"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "slurm (unknown)" in output
    assert "None" not in output  # nulls never leak


def test_workloads_help_documents_slurm_cluster_field() -> None:
    result = runner.invoke(app, ["factory", "workloads", "--help"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    # wrap-safe fragments of the documented line
    assert "scheduler_display_name" in result.output
    assert "slurm cluster" in result.output.lower()
    assert "deployment" in result.output
    assert "placement" in result.output
    # the help describes BOTH meanings of null: direct placement AND
    # unidentified deployments (rendered unknown)
    assert "unidentified deployment" in result.output
    assert "unknown" in result.output


def test_workloads_cluster_selects_a_slurm_cluster(monkeypatch: pytest.MonkeyPatch) -> None:
    # --cluster also takes a Slurm cluster name: only that pool shows, in
    # the table and in the JSON envelope.
    slurm_a = _workload("slurm:ac12:1", "slurm", "running")
    slurm_a["scheduler_display_name"] = "slurm-a"
    slurm_b = _workload("slurm:ac12:2", "slurm", "running")
    slurm_b["scheduler_display_name"] = "slurm-b"
    training = _workload("training:run-1", "training", "running")
    payload = _workloads_payload([slurm_a, slurm_b, training])
    _install(monkeypatch, payload)

    assert _plain_rows("--cluster", "slurm-a") == [
        ["[1]", "cluster-a", "·", "B300", *_counts(0, 0, 1, 0)],
        ["NAME", "USER", "STATE", "NODES", "AGE", "ID"],
        ["slurm", "(slurm-a)"],
        ["slurm-job", "carol", "running", "4", "30:00", "slurm:ac12:1"],
        ["other", *_counts(0, 0, 1, 0)],
    ]

    dummy = _install(monkeypatch, payload)
    json_result = runner.invoke(
        app, ["factory", "workloads", "--cluster", "slurm-a", "--json"], env=TEST_ENV
    )
    data = json.loads(json_result.stdout)
    assert [w["id"] for w in data["workloads"]] == ["slurm:ac12:1"]
    # --cluster needs the status payload to resolve; nodes never matter for JSON.
    assert {call["endpoint"] for call in dummy.calls} == {"/factory/workloads", "/factory/status"}

    miss = runner.invoke(app, ["factory", "workloads", "--cluster", "nope"], env=TEST_ENV)
    assert miss.exit_code == 1, miss.output
    assert "No cluster matched 'nope'" in strip_ansi(miss.output)


def test_workloads_duplicate_scheduler_labels_render_distinctly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two rows can share a Slurm cluster: they share one pool block but
    # are never merged or deduplicated — IDs stay distinct.
    row_a = _workload("slurm:ac12:1", "slurm", "running")
    row_a["scheduler_display_name"] = "slurm-a"
    row_b = _workload("slurm:ac12:2", "slurm", "running")
    row_b["scheduler_display_name"] = "slurm-a"
    _install(monkeypatch, _workloads_payload([row_a, row_b]))

    result = runner.invoke(app, ["factory", "workloads"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "slurm:ac12:1" in output and "slurm:ac12:2" in output
    assert output.count("slurm-a") == 1  # one pool block holds both rows


def test_workloads_cluster_with_terminal_state_uses_widest_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A client-side --cluster narrow happens AFTER the server pages: the
    # implicit history limit must be the widest bounded page (200), never
    # the default 50 that could truncate the cluster's history before the
    # filter sees it.
    row = _workload("slurm:ac12:1", "slurm", "completed")
    row["scheduler_display_name"] = "slurm-a"
    dummy = _install(monkeypatch, _workloads_payload([row]))

    result = runner.invoke(
        app,
        ["factory", "workloads", "--state", "completed", "--cluster", "slurm-a"],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    assert dummy.workloads_params()["limit"] == 200
    # without --cluster, the terminal default stays 50
    dummy = _install(monkeypatch, _workloads_payload([row]))
    plain = runner.invoke(app, ["factory", "workloads", "--state", "completed"], env=TEST_ENV)
    assert plain.exit_code == 0, plain.output
    assert dummy.workloads_params()["limit"] == 50


def test_workloads_cluster_terminal_empty_says_bounded_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Beyond one server page a client-side --cluster match cannot be
    # proven absent — the empty result names the fetched window, never a
    # bare "no matches".
    empty = {
        "schema_version": 1,
        "as_of": _iso(datetime.now(timezone.utc)),
        "workloads": [],
        "sources": [_source("slurm")],
        "truncated": True,
    }
    _install(monkeypatch, empty)

    result = runner.invoke(
        app,
        ["factory", "workloads", "--state", "completed", "--cluster", "office-a100"],
        env=TEST_ENV,
    )
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "in the fetched window (200 rows)" in output
    assert "No factory workloads found." not in output

    # merely SUPPLYING the limit proves nothing: an unflagged short
    # response was not cut, so the miss is definitive
    unflagged = {
        "schema_version": 1,
        "as_of": _iso(datetime.now(timezone.utc)),
        "workloads": [],
        "sources": [_source("slurm")],
    }
    _install(monkeypatch, unflagged)
    short = runner.invoke(
        app,
        ["factory", "workloads", "--state", "completed", "--cluster", "office-a100"],
        env=TEST_ENV,
    )
    short_output = strip_ansi(short.output)
    assert short.exit_code == 0, short.output
    assert "No factory workloads match the given filters." in short_output
    assert "fetched window" not in short_output

    # An explicit --limit on a LIVE --cluster query is bounded too: the
    # server pages before the client predicate, so a miss is unprovable.
    _install(monkeypatch, empty)
    live_limited = runner.invoke(
        app,
        ["factory", "workloads", "--cluster", "office-a100", "--limit", "25"],
        env=TEST_ENV,
    )
    live_output = strip_ansi(live_limited.output)
    assert live_limited.exit_code == 0, live_limited.output
    assert "in the fetched window (25 rows)" in live_output

    # An UNFLAGGED live --cluster miss (no truncation, short response) is
    # definitive: nothing was cut, so the plain filter wording stands (a
    # cluster with nodes would show its idle capacity instead).
    _install(monkeypatch, unflagged, nodes_payload=_nodes_payload([]))
    live_plain = runner.invoke(
        app, ["factory", "workloads", "--cluster", "office-a100"], env=TEST_ENV
    )
    plain_output = strip_ansi(live_plain.output)
    assert live_plain.exit_code == 0, live_plain.output
    assert "No factory workloads match the given filters." in plain_output
    assert "fetched window" not in plain_output


def test_workloads_matching_but_stale_rows_are_not_a_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A filter that MATCHES rows whose evidence is stale is not a no-match:
    # the warning carries the unavailability and nothing else prints —
    # never a contradictory "No factory workloads match the given filters."
    stale_row = _workload("slurm:ac12:7", "slurm", "running")
    stale_row["scheduler_display_name"] = "slurm-a"
    stale_row["source"] = _source("slurm", status="stale", age_seconds=7200)
    payload = _workloads_payload(
        [stale_row],
        sources=[_source("slurm", status="stale", age_seconds=7200)],
    )
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "workloads", "--cluster", "slurm-a"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "slurm jobs unavailable" in output  # or mixed/unavailable wording
    assert "No factory workloads match" not in output

    # same for --user: a matching stale row is not a miss
    stale_training = _workload("training:run-9", "training", "running")
    stale_training["owner"] = {"kind": "prime", "display_name": "alice"}
    stale_training["source"] = _source("training", status="stale", age_seconds=7200)
    payload2 = _workloads_payload(
        [stale_training],
        sources=[_source("training", status="stale", age_seconds=7200)],
    )
    _install(monkeypatch, payload2)
    result2 = runner.invoke(app, ["factory", "workloads", "--user", "alice"], env=TEST_ENV)
    output2 = strip_ansi(result2.output)
    assert result2.exit_code == 0, result2.output
    assert "No factory workloads match" not in output2


def test_workloads_degraded_zero_row_result_is_warning_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A failed or partial read cannot establish that no workload matches:
    # with the relevant training source stale and zero rows returned, the
    # unavailability warning stands alone — never a "no match" claim.
    payload = {
        "schema_version": 1,
        "as_of": _iso(datetime.now(timezone.utc)),
        "workloads": [],
        "sources": [_source("training", status="stale", age_seconds=7200)],
    }
    _install(monkeypatch, payload)

    result = runner.invoke(
        app, ["factory", "workloads", "--type", "training", "--user", "alice"], env=TEST_ENV
    )
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "training jobs unavailable" in output
    assert "No factory workloads match" not in output
    assert "No factory workloads found." not in output


def test_workloads_truncated_envelope_renders_cut_note(monkeypatch: pytest.MonkeyPatch) -> None:
    # The backend caps the default window at 200 rows and flags it: one dim
    # line under the table says so and points at the refiners.
    payload = _workloads_payload(_default_rows())
    payload["truncated"] = True
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "workloads"], env=TEST_ENV)
    output = strip_ansi(result.output)
    assert result.exit_code == 0, result.output
    assert "narrow the server query with --type/--state/--since" in output

    # derived from the active limit: a small --limit names its own page
    # and still offers the raise advice
    _install(monkeypatch, payload)
    small = runner.invoke(app, ["factory", "workloads", "--limit", "25"], env=TEST_ENV)
    small_output = strip_ansi(small.output)
    assert small.exit_code == 0, small.output
    assert "Showing first 25 workloads" in small_output
    assert "raise --limit" in small_output

    # at the 200-row maximum the raise advice disappears
    _install(monkeypatch, payload)
    capped = runner.invoke(app, ["factory", "workloads", "--limit", "200"], env=TEST_ENV)
    capped_output = strip_ansi(capped.output)
    assert capped.exit_code == 0, capped.output
    assert "Showing first 200 workloads" in capped_output
    assert "raise --limit" not in capped_output

    # absent flag (older backends) or False: no cut note
    _install(monkeypatch, _workloads_payload(_default_rows()))
    plain = runner.invoke(app, ["factory", "workloads"], env=TEST_ENV)
    plain_output = strip_ansi(plain.output)
    assert plain.exit_code == 0, plain.output
    assert "Showing first 200" not in plain_output

    # --json stays the exact envelope incl. the flag
    _install(monkeypatch, payload)
    json_result = runner.invoke(app, ["factory", "workloads", "--json"], env=TEST_ENV)
    assert json.loads(json_result.stdout)["truncated"] is True


def test_workloads_state_cell_native_state_has_space(monkeypatch: pytest.MonkeyPatch) -> None:
    # PENDING only restates "queued" and is dropped; a native state that
    # adds detail renders with a space (the UI once shipped
    # 'queued(PENDING)').
    queued = _workload("slurm:ac12:8421", "slurm", "queued")
    failed = _workload("slurm:ac12:8422", "slurm", "failed", native_state="NODE_FAIL")
    _install(monkeypatch, _workloads_payload([queued, failed]))

    result = runner.invoke(app, ["factory", "workloads", "--state", "all"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "PENDING" not in output
    assert "failed (NODE_FAIL)" in output


def test_workloads_truncated_empty_filtered_result_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Bugbot: the truncated flag was ignored on empty client-filtered
    # results — a --user miss inside a backend-capped page claimed a
    # definitive no-match. With the envelope flagging truncation, the
    # miss wording names the fetched window.
    payload = {
        "schema_version": 1,
        "as_of": _iso(datetime.now(timezone.utc)),
        "workloads": [],
        "sources": [_source("training")],
        "truncated": True,
    }
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "workloads", "--user", "nobody"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "in the fetched window (200 rows)" in output
    assert "No factory workloads found." not in output

    # without truncation the same live miss stays definitive
    payload.pop("truncated")
    _install(monkeypatch, payload)
    plain = runner.invoke(app, ["factory", "workloads", "--user", "nobody"], env=TEST_ENV)
    plain_output = strip_ansi(plain.output)
    assert plain.exit_code == 0, plain.output
    assert "No factory workloads match the given filters." in plain_output
    assert "fetched window" not in plain_output


def test_workloads_server_filtered_empty_history_is_definitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Terminal --state queries always send a default limit, but the server
    # applies --state/--type/--since BEFORE the cap: a zero-row result is
    # a genuine empty history, not a bounded-window miss.
    payload = {
        "schema_version": 1,
        "as_of": _iso(datetime.now(timezone.utc)),
        "workloads": [],
        "sources": [_source("training")],
    }
    _install(monkeypatch, payload)

    result = runner.invoke(
        app, ["factory", "workloads", "--state", "completed", "--since", "7d"], env=TEST_ENV
    )
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "No factory workloads match the given filters." in output
    assert "fetched window" not in output
