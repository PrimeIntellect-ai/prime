"""Tests for `prime factory status` (fleet allocation glance)."""

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

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
        "display_name": "research-b300",
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


class _DummyAPIClient:
    def __init__(
        self, payload: Dict[str, Any], nodes_payload: Optional[Dict[str, Any]] = None
    ) -> None:
        self._payload = payload
        self._nodes_payload = (
            nodes_payload if nodes_payload is not None else _default_nodes_payload()
        )
        self.calls: list[Dict[str, Any]] = []

    def get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None, timeout: Any = None
    ) -> Dict[str, Any]:
        self.calls.append({"endpoint": endpoint, "params": params})
        if endpoint == "/factory/nodes":
            return self._nodes_payload
        return self._payload


def _install(
    monkeypatch: pytest.MonkeyPatch,
    payload: Dict[str, Any],
    team_id: Optional[str] = "team-123",
    nodes_payload: Optional[Dict[str, Any]] = None,
) -> _DummyAPIClient:
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)
    dummy = _DummyAPIClient(payload, nodes_payload=nodes_payload)
    monkeypatch.setattr("prime_cli.commands.factory.APIClient", lambda: dummy)
    monkeypatch.setattr("prime_cli.commands.factory.Config", lambda: _StubConfig(team_id))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    return dummy


def test_factory_status_table_renders_pools_and_allocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _status_payload())

    result = runner.invoke(app, ["factory", "status", "--verbose"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "research-b300" in output
    assert "128 B300 GPUs" in output
    assert "online" in output
    assert "IDLE INSIDE" in output
    assert "unassigned" in output and "16" in output
    for pool_type in ("training", "inference", "slurm"):
        assert pool_type in output
    # reserved -> in-use split per pool
    assert "48" in output and "32" in output and "16" in output
    assert "in-use = GPUs held by running jobs (not GPU-activity measurements)" in output
    assert "Error" not in output


def test_factory_status_all_fresh_sources_show_no_unavailable_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _status_payload())

    result = runner.invoke(app, ["factory", "status"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "breakdown unavailable" not in output
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
    assert dummy.calls and dummy.calls[0]["params"] == {"team_id": "team-42"}


def test_factory_status_all_stale_sources_skip_table_with_friendly_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Stale node data suppresses every pool row: no table, no '?' cells,
    # one plain-language line with the age of the last observation.
    payload = _status_payload(
        sources=[_source("capacity", status="stale", age_seconds=26 * 3600)],
        unknown_gpus=8,
    )
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "status", "--verbose"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "workload breakdown unavailable — node data last seen 1d ago" in output
    # The pool table is skipped entirely.
    assert "RESERVED" not in output
    assert "unassigned" not in output
    # Suppression replaces placeholder cells; unknown values never show '?'.
    assert "?" not in output
    assert "Warning" not in output
    # No footnote without a table.
    assert "in-use = GPUs held by running jobs" not in output


def test_factory_status_mixed_fresh_and_stale_sources_render_only_fresh_pools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _status_payload(
        sources=[
            _source("capacity", age_seconds=12),
            _source("training", age_seconds=10),
            _source("inference", age_seconds=10),
            _source("slurm", status="stale", age_seconds=2 * 3600),
        ],
        pools=[
            _pool("training", 32, 32, 0, 0),
            _pool("inference", 32, 32, 0, 0),
            _pool("slurm", 48, None, None, 48),
        ],
    )
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "status", "--verbose"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    # Fresh pools render.
    assert "training" in output and "inference" in output
    assert "IDLE INSIDE" in output and "unassigned" in output
    assert "in-use = GPUs held by running jobs" in output
    # The stale pool is aggregated into one friendly line naming it.
    assert "slurm breakdown unavailable — scheduler data last seen 2h ago" in output
    # The suppressed row's numbers never render, and no '?' placeholders.
    assert "?" not in output
    assert "Warning" not in output
    assert "48" not in output


def test_factory_status_stale_pool_source_visible_with_header_phrase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A degraded source without a suppressed pool still surfaces in the
    # header via the last-seen phrase.
    payload = _status_payload(
        sources=[
            _source("capacity", age_seconds=12),
            _source("training", status="error", age_seconds=4000),
            _source("inference", age_seconds=10),
            _source("slurm", age_seconds=3),
        ],
        pools=[
            _pool("inference", 32, 32, 0, 0),
            _pool("slurm", 48, 16, 32, 0),
        ],
    )
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "status", "--verbose"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "training data last seen 1h ago" in output
    assert "breakdown unavailable" not in output
    assert "Warning" not in output


def test_factory_status_cluster_filter_by_display_name_and_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _status_payload()
    second = dict(_status_payload()["clusters"][0], display_name="research-h200", gpu_type="H200")
    payload["clusters"].append(second)
    _install(monkeypatch, payload)

    by_name = runner.invoke(
        app, ["factory", "status", "--verbose", "--cluster", "research-h200"], env=TEST_ENV
    )
    assert by_name.exit_code == 0, by_name.output
    out = strip_ansi(by_name.output)
    assert "research-h200" in out
    assert "research-b300" not in out
    assert "H200" in out

    by_index = runner.invoke(
        app, ["factory", "status", "--verbose", "--cluster", "1"], env=TEST_ENV
    )
    assert by_index.exit_code == 0, by_index.output
    out = strip_ansi(by_index.output)
    assert "research-b300" in out
    assert "research-h200" not in out

    # Compact mode: --cluster selects table rows too.
    compact = runner.invoke(app, ["factory", "status", "--cluster", "research-h200"], env=TEST_ENV)
    assert compact.exit_code == 0, compact.output
    out = strip_ansi(compact.output)
    assert "research-h200" in out
    assert "research-b300" not in out


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
        dict(payload["clusters"][0], display_name="research-h200", gpu_type="H200")
    )
    _install(monkeypatch, payload)

    result = runner.invoke(
        app, ["factory", "status", "--json", "--cluster", "research-h200"], env=TEST_ENV
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
        app, ["factory", "status", "--json", "--cluster", "research-b300"], env=TEST_ENV
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
    assert cluster.display_name == "research-b300"
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
    assert "research-b300" in output
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
    payload["clusters"].append(dict(payload["clusters"][0], display_name="research-h200"))
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "status", "--cluster", "2"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert "research-h200" in strip_ansi(result.output)
    assert "research-b300" not in strip_ansi(result.output)


def test_factory_status_splits_clusters_and_workloads_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _status_payload())

    result = runner.invoke(app, ["factory", "status", "--verbose"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    # Clear section headers answer "what do I have" then "what is running".
    assert "CLUSTERS" in output
    assert "WORKLOADS" in output
    assert output.index("CLUSTERS") < output.index("WORKLOADS")
    # The inventory line with display name, GPU type + count and status
    # lives in the CLUSTERS section, above the WORKLOADS header.
    assert "research-b300 · 128 B300 GPUs · online" in output
    assert output.index("research-b300 · 128 B300 GPUs · online") < output.index("WORKLOADS")


def test_factory_status_unassigned_and_unknown_are_cluster_summary_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _status_payload())

    result = runner.invoke(app, ["factory", "status", "--verbose"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    # Cluster-level summary lines, not pool table rows with dash placeholders.
    assert "unassigned: 16 GPUs" in output
    assert "unknown: 0 GPUs" in output
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith(("unassigned", "unknown")):
            assert ":" in stripped
            assert " - " not in stripped


def test_factory_status_degraded_note_stays_on_cluster_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A degraded source without a suppressed pool surfaces as a
    # last-seen phrase on the cluster's CLUSTERS line.
    payload = _status_payload(
        sources=[
            _source("capacity", age_seconds=12),
            _source("training", status="error", age_seconds=4000),
        ],
        pools=[
            _pool("inference", 32, 32, 0, 0),
            _pool("slurm", 48, 16, 32, 0),
        ],
    )
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "status", "--verbose"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    cluster_lines = [
        line for line in output.splitlines() if line.strip().startswith("research-b300 ·")
    ]
    assert cluster_lines, output
    assert "training data last seen 1h ago" in cluster_lines[0]


def test_factory_status_multi_cluster_indices_in_both_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _status_payload()
    payload["clusters"].append(
        dict(payload["clusters"][0], display_name="research-h200", gpu_type="H200")
    )
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "status", "--verbose"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    # Indices shown in CLUSTERS match the WORKLOADS labels and the
    # --cluster index selector.
    assert "[1] research-b300" in output
    assert "[2] research-h200" in output
    assert output.count("[1] research-b300") == 2
    assert output.count("[2] research-h200") == 2
    # The plain-words footnote prints once for the whole section.
    assert output.count("in-use = GPUs held by running jobs") == 1


def test_factory_status_no_pools_renders_summary_lines_without_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _status_payload(pools=[], unassigned_gpus=64, unknown_gpus=None)
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "status", "--verbose"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "unassigned: 64 GPUs" in output
    # No pool rows exist, so no table and no footnote.
    assert "RESERVED" not in output
    assert "no workloads reported" not in output
    assert "in-use = GPUs held by running jobs" not in output


def test_factory_status_empty_cluster_prints_no_pools_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _status_payload(pools=[], unassigned_gpus=None, unknown_gpus=None)
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "status", "--verbose"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "no workloads reported" in output
    assert "RESERVED" not in output
    assert "breakdown unavailable" not in output


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
    assert "research-b300" in output
    assert "slurm" in output
    assert "drill down: prime factory nodes" in output

    json_result = runner.invoke(app, ["factory", "status", "--json"], env=TEST_ENV)
    assert json.loads(json_result.stdout) == payload

    # Legacy `pools` key still parses (tolerated alias, not the primary).
    legacy_payload = _status_payload()  # fixture still uses the `pools` key
    _install(monkeypatch, legacy_payload)
    legacy = runner.invoke(app, ["factory", "status", "--verbose"], env=TEST_ENV)
    assert legacy.exit_code == 0, legacy.output
    assert "slurm" in strip_ansi(legacy.output)


def test_factory_status_drill_down_hint_only_when_table_rendered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Fresh data renders the WORKLOADS table -> the dim drill-down hint shows.
    _install(monkeypatch, _status_payload())
    shown = runner.invoke(app, ["factory", "status", "--verbose"], env=TEST_ENV)
    assert "drill down: prime factory nodes" in strip_ansi(shown.output)

    # All-suppressed (stale) data renders no table -> no hint.
    stale = _status_payload()
    for cluster in stale["clusters"]:
        cluster["sources"] = [_source("capacity", status="stale", age_seconds=86400)]
    _install(monkeypatch, stale)
    hidden = runner.invoke(app, ["factory", "status", "--verbose"], env=TEST_ENV)
    output = strip_ansi(hidden.output)
    assert "workload breakdown unavailable" in output
    assert "drill down: prime factory nodes" not in output


def test_factory_status_stale_capacity_with_no_allocations_not_an_empty_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Degraded capacity with zero allocation rows is a coverage failure,
    # not a valid empty cluster: the breakdown-unavailable line with the
    # last-observed age must show, and the empty-cluster wording must not.
    payload = _status_payload(
        pools=[],
        unassigned_gpus=None,
        unknown_gpus=None,
        sources=[_source("capacity", status="stale", age_seconds=86400)],
    )
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "status", "--verbose"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "workload breakdown unavailable" in output
    assert "node data last seen 1d ago" in output
    assert "no workloads reported" not in output


def test_factory_status_fresh_empty_cluster_still_says_no_workloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Fresh capacity + zero allocation rows IS a valid empty cluster.
    payload = _status_payload(pools=[], unassigned_gpus=None, unknown_gpus=None)
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "status", "--verbose"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "no workloads reported" in output
    assert "breakdown unavailable" not in output


def test_factory_status_present_allocation_without_source_entry_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A present allocation whose source entry is omitted from the envelope
    # is unknown evidence — never silently fresh.
    payload = _status_payload(
        sources=[
            _source("capacity", age_seconds=12),
            _source("training", age_seconds=10),
            _source("inference", age_seconds=10),
            # slurm source entry omitted although the slurm pool claims GPUs
        ],
        pools=[
            _pool("training", 32, 32, 0, 0),
            _pool("inference", 32, 32, 0, 0),
            _pool("slurm", 48, 16, 32, 0),
        ],
    )
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "status", "--verbose"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "slurm breakdown unavailable" in output
    # fresh pools still render; the unproven one does not
    assert "training" in output and "inference" in output
    assert "48" not in output


def test_factory_status_inactive_all_zero_pool_without_source_renders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The backend's designed inactive-group signal: an all-zero allocation
    # row with no source entry is complete evidence of nothing.
    payload = _status_payload(
        sources=[
            _source("capacity", age_seconds=12),
            _source("slurm", age_seconds=3),
        ],
        pools=[
            _pool("training", 0, 0, 0, 0),  # inactive: zeros, no source
            _pool("slurm", 48, 16, 32, 0),
        ],
    )
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "status", "--verbose"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "training" in output  # inactive zeros render
    assert "breakdown unavailable" not in output


def test_factory_status_missing_capacity_entry_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _status_payload(sources=[])
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "status", "--verbose"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "workload breakdown unavailable" in output
    assert "no workloads reported" not in output


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


def test_factory_status_compact_table_is_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _status_payload())

    result = runner.invoke(app, ["factory", "status"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    # One compact table: no sections, no footnotes, no drill-down hints.
    assert "CLUSTERS" not in output and "WORKLOADS" not in output
    assert "in-use = GPUs held" not in output
    assert "drill down" not in output
    assert "RESERVED" not in output  # the detailed allocation table is gone
    for header in ("CLUSTER", "GPU", "STATUS", "HELD", "IN USE", "IDLE", "NODES", "DATA"):
        assert header in output
    assert "research-b300" in output
    assert "B300" in output
    assert "online" in output
    # glance facts: reserved sum 112, observed leaf sum 80, idle 32
    assert "112" in output and "80" in output and "32" in output
    # node summary from the nodes endpoint (2 ready of 2)
    assert "2/2" in output
    assert "fresh" in output
    # exactly one row: no dim degraded line under the table
    assert "degraded sources" not in output


def test_factory_status_compact_degraded_row_renders_dashes_and_data_age(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _status_payload(sources=[_source("capacity", status="stale", age_seconds=3600)])
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "status"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    # Stale node evidence: em-dash counts, never zeros, plus a plain age.
    assert "node data 1h ago" in output
    assert "fresh" not in output
    # no stale group numbers render in HELD/IN USE/IDLE
    assert "112" not in output and "80" not in output
    # The single dim line under the table points at --verbose.
    assert "degraded sources — details: prime factory status --verbose" in output
    assert output.count("degraded sources") == 1


def test_factory_status_compact_nodes_column_counts_cordoned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nodes_payload = {
        "schema_version": 1,
        "as_of": _iso(datetime.now(timezone.utc)),
        "sources": [_source("capacity", age_seconds=12)],
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
                        "assigned_to": None,
                    },
                    {
                        "name": "gpu-02",
                        "state": "cordoned",
                        "gpu_type": "B300",
                        "gpus_total": 8,
                        "gpus_used": 8,
                        "assigned_to": "slurm",
                    },
                    {
                        "name": "gpu-03",
                        "state": "offline",
                        "gpu_type": "B300",
                        "gpus_total": 8,
                        "gpus_used": None,
                        "assigned_to": None,
                    },
                ],
            }
        ],
    }
    _install(monkeypatch, _status_payload(), nodes_payload=nodes_payload)

    result = runner.invoke(app, ["factory", "status"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "1/3, 1 cgdn" in output


def test_factory_status_compact_nodes_fetch_failure_degrades_to_dash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _SelectiveClient:
        def __init__(self, status_payload):
            self._status_payload = status_payload
            self.calls = []

        def get(self, endpoint, params=None, timeout=None):
            self.calls.append({"endpoint": endpoint, "params": params})
            if endpoint == "/factory/nodes":
                raise APIError("node view unavailable")
            return self._status_payload

    selective = _SelectiveClient(_status_payload())
    monkeypatch.setattr("prime_cli.commands.factory.APIClient", lambda: selective)
    monkeypatch.setattr("prime_cli.commands.factory.Config", lambda: _StubConfig("team-123"))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    result = runner.invoke(app, ["factory", "status"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "—" in output  # NODES column degraded, status row still renders
    # DATA describes the whole row: a missing node view is not "fresh".
    assert "fresh" not in output
    assert "node view unavailable" in output
    assert "degraded sources" in output  # the dim line shows


def test_factory_status_compact_partial_degradation_shows_only_that_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _status_payload(
        sources=[
            _source("capacity", age_seconds=12),
            _source("training", age_seconds=10),
            _source("inference", age_seconds=10),
            _source("slurm", status="stale", age_seconds=2 * 3600),
        ],
        pools=[
            _pool("training", 32, 32, 0, 0),
            _pool("inference", 32, 32, 0, 0),
            _pool("slurm", 48, None, None, 48),
        ],
    )
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "status"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "scheduler data 2h ago" in output
    assert "fresh" not in output
    # An active stale contributor makes every aggregate unknowable: em-dash
    # totals, never a partial sum presented as complete (and never the
    # stale 48 reserved either).
    assert "—" in output
    assert "64" not in output and "48" not in output


def test_factory_status_json_makes_no_nodes_call(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _status_payload()
    dummy = _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "status", "--json"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert [call["endpoint"] for call in dummy.calls] == ["/factory/status"]
    assert json.loads(result.stdout) == payload


def test_factory_status_verbose_keeps_detailed_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _status_payload())

    result = runner.invoke(app, ["factory", "status", "--verbose"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "CLUSTERS" in output and "WORKLOADS" in output
    assert "RESERVED" in output  # detailed allocation table
    assert "drill down: prime factory nodes" in output
    assert "in-use = GPUs held by running jobs" in output


def test_factory_status_compact_null_allocation_does_not_poison_aggregate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A running training row with null in_use must not erase the healthy
    # peers' contribution from the used-GPUs aggregate.
    payload = _status_payload(
        pools=[
            _pool("training", 32, None, None, 32),
            _pool("inference", 32, 32, 0, 0),
            _pool("slurm", 48, 16, 32, 0),
        ],
    )
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "status"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    # HELD: 32+32+48; IN USE: 32 (inference) + 16 (slurm); null training adds 0
    # IDLE: inference 0 + slurm 32; null training idle adds nothing.
    assert "112" in output and "48" in output and "32" in output
    assert "—" not in output


def test_factory_status_compact_stale_source_numbers_never_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Numbers from a stale source must not appear in the GPUS column: the
    # stale slurm in_use (last-known) is excluded; the DATA column carries
    # the age instead.
    payload = _status_payload(
        sources=[
            _source("capacity", age_seconds=12),
            _source("training", age_seconds=10),
            _source("inference", age_seconds=10),
            _source("slurm", status="stale", age_seconds=2 * 3600),
        ],
        pools=[
            _pool("training", 32, 32, 0, 0),
            _pool("inference", 32, 32, 0, 0),
            _pool("slurm", 48, 8, 40, 0),  # stale last-known numbers
        ],
    )
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "status"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    # An active stale contributor makes every aggregate unknowable: em-dash
    # totals — never the stale 48/8/40, and never a partial sum presented
    # as complete.
    assert "—" in output
    assert "64" not in output and "48" not in output and "40" not in output
    assert "scheduler data 2h ago" in output


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
    # replacement between the two requests: NODES degrades to an em-dash
    # for both rows instead of risking cross-wired counts.
    assert "2/2" not in output and "0/1, 1 cgdn" not in output
    assert "—" in output

    # index selection over duplicates degrades the same way
    selected = runner.invoke(app, ["factory", "status", "--cluster", "2"], env=TEST_ENV)
    selected_output = strip_ansi(selected.output)
    assert selected.exit_code == 0, selected.output
    assert "0/1, 1 cgdn" not in selected_output
    assert "—" in selected_output


def test_factory_status_compact_nodes_fetch_uses_short_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: Dict[str, Any] = {}

    class _RecordingClient:
        def get(self, endpoint, params=None, timeout=None):
            recorded[endpoint] = timeout
            if endpoint == "/factory/nodes":
                return _default_nodes_payload()
            return _status_payload()

    monkeypatch.setattr("prime_cli.commands.factory.APIClient", lambda: _RecordingClient())
    monkeypatch.setattr("prime_cli.commands.factory.Config", lambda: _StubConfig("team-123"))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    result = runner.invoke(app, ["factory", "status"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    # The best-effort fetch is capped so a stalling nodes call degrades
    # to an em-dash instead of hanging the glance.
    assert recorded.get("/factory/nodes") is not None
    assert recorded["/factory/nodes"] <= 5.0
    # the status fetch itself is uncapped
    assert recorded.get("/factory/status") is None


def test_factory_nodes_help_documents_top_level_sources() -> None:
    # --help makes no API call; no fixture install needed.
    result = runner.invoke(app, ["factory", "nodes", "--help"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    # clusters[] carries no sources; the envelope documents them explicitly.
    assert "clusters[] = {display_name, status, nodes[]}" in result.output
    assert "clusters[] = {display_name, status, nodes[], sources[]}" not in result.output


def test_factory_status_compact_malformed_nodes_json_degrades_to_dash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # HTTP 200 with a non-JSON body escapes the client as a raw ValueError
    # (not an APIError): the best-effort handler must still degrade the
    # NODES column instead of crashing the whole status render.
    class _MalformedNodesClient:
        def get(self, endpoint, params=None, timeout=None):
            if endpoint == "/factory/nodes":
                raise ValueError("Expecting value: line 1 column 1 (char 0)")
            return _status_payload()

    monkeypatch.setattr("prime_cli.commands.factory.APIClient", lambda: _MalformedNodesClient())
    monkeypatch.setattr("prime_cli.commands.factory.Config", lambda: _StubConfig("team-123"))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    result = runner.invoke(app, ["factory", "status"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "research-b300" in output  # the status row renders
    assert "—" in output  # NODES degrades to an em-dash
    # DATA describes the whole row: a missing node view is not "fresh".
    assert "fresh" not in output
    assert "node view unavailable" in output
    assert "degraded sources" in output


def test_factory_status_compact_node_join_verifies_cluster_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The status and nodes responses are separate requests: when their
    # cluster lists cannot be aligned (drift between the requests), the
    # positional join must NOT display one cluster's node counts on
    # another cluster's row — NODES degrades to an em-dash instead.
    now = datetime.now(timezone.utc)
    status_payload = {
        "schema_version": 1,
        "as_of": _iso(now),
        "clusters": [
            {
                **_status_payload()["clusters"][0],
                "display_name": "research-b300",
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
                "display_name": "research-b300",
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
    assert "research-b300" in output and "office-a100" in output
    # Position 1 pairs with office-a100's node data on research-b300's row:
    # identity mismatch -> em-dash NODES, never the cross-wired count.
    assert "—" in output
    assert "1/1" not in output


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


def test_factory_status_compact_stale_slurm_plus_inactive_zeros(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Backend-produced shape: inactive training/inference are all-zero
    # allocation rows WITHOUT source entries (the backend only emits
    # per-kind sources when active); slurm is stale holding 16/8.
    # Incomplete evidence must never render as complete totals: all three
    # aggregates em-dash, never "0".
    payload = _status_payload(
        sources=[
            _source("capacity", age_seconds=12),
            _source("slurm", status="stale", age_seconds=2 * 3600),
        ],
        pools=[
            _pool("training", 0, 0, 0, 0),  # inactive: zeros, no source entry
            _pool("inference", 0, 0, 0, 0),  # inactive: zeros, no source entry
            _pool("slurm", 16, 8, 8, 0),  # active, stale: 16/8 last-known
        ],
    )
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "status"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    # No aggregate renders a number: the stale 16/8 never shows, and the
    # inactive zeros never masquerade as a complete "0" total.
    assert "16" not in output and "8" not in output
    # a bare zero in a right-justified metric cell would render " 0 │"
    assert " 0 │" not in output
    assert "    — " in output
    assert "scheduler data 2h ago" in output


def test_factory_status_compact_missing_training_plus_healthy_slurm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Backend-produced shape: the training source entry is missing while
    # the allocation claims GPUs (unprovable evidence); slurm is healthy.
    # The aggregates cannot be known, and DATA must describe the whole row.
    payload = _status_payload(
        sources=[
            _source("capacity", age_seconds=12),
            _source("slurm", age_seconds=3),
        ],
        pools=[
            _pool("training", 32, 32, 0, 0),  # claims GPUs, no source entry
            _pool("inference", 0, 0, 0, 0),  # inactive: zeros, no source entry
            _pool("slurm", 48, 16, 32, 0),  # healthy
        ],
    )
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "status"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    # An active unprovable contributor makes every aggregate unknowable.
    assert "—" in output
    assert "48" not in output and "32" not in output and "16" not in output
    # DATA describes the whole row, not just the listed sources.
    assert "training data unavailable" in output
    assert "fresh" not in output
