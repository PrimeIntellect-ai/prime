"""Tests for `prime factory status` (fleet allocation glance)."""

import json
import re
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
    "COLUMNS": "200",
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
    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload
        self.calls: list[Dict[str, Any]] = []

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


def test_factory_status_table_renders_pools_and_allocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _status_payload())

    result = runner.invoke(app, ["factory", "status"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "research-b300" in output
    assert "128 B300 GPUs" in output
    assert "online" in output
    assert re.search(r"capacity \d+s ago", output)
    assert re.search(r"slurm \d+s ago", output)
    assert "IDLE INSIDE" in output
    assert "unassigned" in output and "16" in output
    for pool_type in ("training", "inference", "slurm"):
        assert pool_type in output
    # reserved -> in-use split per pool
    assert "48" in output and "32" in output and "16" in output
    assert "IN USE = allocated to leaf workloads" in output
    assert "Error" not in output


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


def test_factory_status_stale_and_error_sources_are_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _status_payload(
        sources=[
            _source("capacity", age_seconds=30),
            _source("slurm", status="stale", age_seconds=905),
            _source("training", status="error", age_seconds=4000),
        ],
        pools=[_pool("slurm", 48, None, None, 48)],
        unknown_gpus=8,
    )
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "status"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "stale" in output
    assert "error" in output
    assert "Warning" in output
    # Unknown evidence stays '?' instead of collapsing into a tidy zero.
    assert "?" in output
    assert "48" in output
    assert "unknown" in output


def test_factory_status_cluster_filter_by_display_name_and_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _status_payload()
    second = dict(_status_payload()["clusters"][0], display_name="research-h200", gpu_type="H200")
    payload["clusters"].append(second)
    _install(monkeypatch, payload)

    by_name = runner.invoke(app, ["factory", "status", "--cluster", "research-h200"], env=TEST_ENV)
    assert by_name.exit_code == 0, by_name.output
    out = strip_ansi(by_name.output)
    assert "research-h200" in out
    assert "research-b300" not in out
    assert "H200" in out

    by_index = runner.invoke(app, ["factory", "status", "--cluster", "1"], env=TEST_ENV)
    assert by_index.exit_code == 0, by_index.output
    out = strip_ansi(by_index.output)
    assert "research-b300" in out
    assert "research-h200" not in out


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
    assert [s.kind for s in cluster.sources] == ["capacity", "slurm"]


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
    assert "pools" in strip_ansi(result.output)


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
    assert "pools" in str(excinfo.value)


def test_factory_client_rejects_future_schema_version() -> None:
    dummy = _DummyAPIClient({**_status_payload(), "schema_version": 3})

    with pytest.raises(APIError, match="Unsupported factory status schema version: 3"):
        FactoryClient(dummy).get_status("team-123")  # type: ignore[arg-type]
