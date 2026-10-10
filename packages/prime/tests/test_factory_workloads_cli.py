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
    workload: Dict[str, Any] = {
        "id": id_,
        "type": type_,
        "cluster_display_name": "research-b300",
        "name": f"{type_}-job",
        "state": state,
        "native_state": "RUNNING" if state == "running" else "PENDING",
        "owner": {"kind": "slurm" if type_ == "slurm" else "prime", "display_name": "carol"},
        "requested_gpus": 32,
        "allocated_gpus": 32 if state == "running" else None,
        "created_at": _iso(now - timedelta(hours=2)),
        "started_at": _iso(now - timedelta(minutes=30)) if state == "running" else None,
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


def test_workloads_table_renders_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _workloads_payload(_default_rows()))

    result = runner.invoke(app, ["factory", "workloads"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    for header in ("ID", "TYPE", "NAME", "OWNER", "STATE", "GPU A/R", "AGE"):
        assert header in output
    assert "slurm:ac12:8421" in output
    assert "training" in output and "inference" in output
    assert "carol" in output
    assert "running" in output and "queued" in output
    # allocated/requested split, and null allocation rendered as "-" not 0
    assert "32/32" in output
    assert "-/32" in output
    assert "in-use = GPUs held by running jobs (not GPU-activity measurements)" in output
    assert "unavailable" not in output
    assert "Error" not in output


def test_workloads_age_is_run_age_for_running_and_wait_age_for_queued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _workloads_payload(_default_rows()))

    result = runner.invoke(app, ["factory", "workloads"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    # started 30m ago (running rows) vs. created 2h ago (queued row)
    assert "30m" in output
    assert "2h" in output


def test_workloads_json_prints_exact_api_response(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _workloads_payload(_default_rows())
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "workloads", "--json"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == payload


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
    _install(monkeypatch, empty)

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
    assert dummy.calls and dummy.calls[0]["params"] == {"team_id": "team-42"}


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
    assert dummy.calls[0]["params"] == {
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

    result = runner.invoke(app, ["factory", "workloads", "--cluster", "office-a100"], env=TEST_ENV)
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
    assert "-/-" in output
    assert "5m" in output
    # No fabricated zeros anywhere in the GPU column
    assert "0/0" not in output


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


def test_workloads_queued_age_uses_wait_age_despite_historical_started_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A requeued row keeps its old started_at; the documented wait age
    # must come from created_at, not from the previous run's start.
    row = _workload("training:run-9", "training", "queued")
    _install(monkeypatch, _workloads_payload([row]))

    result = runner.invoke(app, ["factory", "workloads"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    # created 2h ago -> wait age 2h (not the 30m-old historical start)
    assert "2h" in output
    assert "30m" not in output


def test_workloads_cluster_miss_with_degraded_source_shows_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The workloads envelope has no cluster list, only row identities: a
    # selector can miss because a degraded source omitted a cluster's rows.
    # The degraded-source warning must surface instead of a clean miss.
    rows = _default_rows()
    for row in rows:
        row["cluster_display_name"] = "research-b300"
    sources = [
        _source("training"),
        _source("inference"),
        _source("slurm", status="stale", age_seconds=7200),
    ]
    payload = _workloads_payload(rows, sources)
    # Backend omitted every row of the (valid) cluster "office-a100" whose
    # slurm source is stale; --cluster office-a100 misses all rows.
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "workloads", "--cluster", "office-a100"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 1, result.output
    # The payload has a fresh slurm row: mixed coverage, partial wording,
    # and no age (the aggregate timestamp describes the healthy read).
    assert "some slurm job data is unavailable" in output
    assert "results may be incomplete" in output
    assert "scheduler data last seen 2h ago" not in output
    assert "No cluster matched" in output


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

    _install(monkeypatch, empty)
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
    # identity, state, and owner are known facts; the allocation cell is
    # "-/64" (em-dash requested, GPU count from requested).
    row = _workload("training:run-5", "training", "running", requested_gpus=64)
    row["allocated_gpus"] = None
    _install(monkeypatch, _workloads_payload([row]))

    result = runner.invoke(app, ["factory", "workloads"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "training:run-5" in output
    assert "running" in output and "carol" in output
    assert "-/64" in output
    assert "unavailable" not in output  # unobserved allocation is not degradation


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


def test_workloads_table_renders_cluster_column(monkeypatch: pytest.MonkeyPatch) -> None:
    row = _workload("slurm:ac12:8421", "slurm", "running")
    row["cluster_display_name"] = "research-b300"
    _install(monkeypatch, _workloads_payload([row]))

    result = runner.invoke(app, ["factory", "workloads"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "CLUSTER" in output
    assert "research-b300" in output


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


def test_workloads_cluster_miss_honors_type_filter_in_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # --type training + an unmatched --cluster: the slurm source narrowed
    # out by --type must NOT warn on the miss; the requested kind still does.
    rows = _default_rows()
    for row in rows:
        row["cluster_display_name"] = "research-b300"
    sources = [
        _source("training", status="error"),
        _source("inference"),
        _source("slurm", status="stale", age_seconds=7200),
    ]
    # server-side --type training would return only training rows
    training_rows = [r for r in rows if r["type"] == "training"]
    payload = _workloads_payload(training_rows, sources)
    _install(monkeypatch, payload)

    result = runner.invoke(
        app,
        ["factory", "workloads", "--type", "training", "--cluster", "nope"],
        env=TEST_ENV,
    )
    output = strip_ansi(result.output)

    assert result.exit_code == 1, result.output
    assert "slurm jobs unavailable" not in output  # narrowed out by --type
    # The requested kind still warns (fresh training rows exist -> mixed
    # coverage wording, no age).
    assert "some training job data is unavailable" in output
    assert "results may be incomplete" in output
    assert "No cluster matched" in output


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
