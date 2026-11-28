"""Tests for `prime factory metrics` (per-node cluster metrics)."""

import json
import math
from typing import Any, Dict, List, Optional

import pytest
from prime_cli.api.factory_metrics import FactoryMetricsClient, MetricPoint
from prime_cli.client import APIError
from prime_cli.commands import factory_clusters
from prime_cli.commands.factory_clusters import sparkline
from prime_cli.core.client import NotFoundError, ServerError, UnauthorizedError
from prime_cli.main import app
from prime_cli.utils.formatters import strip_ansi
from typer.testing import CliRunner

runner = CliRunner()

TEST_ENV = {
    "PRIME_API_KEY": "dummy",
    "PRIME_DISABLE_VERSION_CHECK": "1",
    "COLUMNS": "240",
    "TERM": "xterm",
}

GIB = 1024**3
STATUS_ENDPOINT = "/factory/status"
METRICS_ENDPOINT = "/model-factory/clusters/pc_123/metrics/nodes"


def _series(base: float, amp: float, n: int = 60) -> List[Dict[str, Any]]:
    return [
        {"timestamp": 1_764_000_000 + i * 60, "value": base + amp * math.sin(i / 6)}
        for i in range(n)
    ]


def _gpu(index: int, util: float) -> Dict[str, Any]:
    return {
        "index": index,
        "current": {
            "gpu_util_percent": util,
            "gpu_memory_used_bytes": 61 * GIB,
            "gpu_memory_total_bytes": 80 * GIB,
            "gpu_temperature_celsius": 58 + index,
            "gpu_power_watts": 610,
        },
        "series": {
            "gpu_util_percent": _series(util, 8),
            "gpu_temperature_celsius": _series(58 + index, 3),
        },
    }


def _ib_port(device: str, receive_errors: int = 0, detail: bool = False) -> Dict[str, Any]:
    port: Dict[str, Any] = {
        "device": device,
        "port": "1",
        "state": "active",
        "physical_state": "link_up",
        "rate_bytes_per_second": 50e9,
        "symbol_errors": 0,
        "link_downed": 0,
        "receive_errors": receive_errors,
    }
    if detail:
        port["current"] = {
            "ib_rx_bytes_per_second": 10.5 * GIB,
            "ib_tx_bytes_per_second": 10.2 * GIB,
            "ib_errors_per_second": 0.0,
        }
        port["series"] = {"ib_rx_bytes_per_second": _series(10.5 * GIB, GIB)}
    return port


def _node(
    name: str,
    util: float = 92.0,
    state: str = "ready",
    with_gpus: bool = False,
    receive_errors: int = 0,
) -> Dict[str, Any]:
    return {
        "name": name,
        "gpu_type": "H100_80GB",
        "gpu_count": 8,
        "state": state,
        "current": {
            "gpu_util_percent": util,
            "gpu_memory_used_bytes": 488 * GIB,
            "gpu_memory_total_bytes": 640 * GIB,
            "gpu_temperature_celsius": 66,
            "gpu_power_watts": 4880,
            "cpu_util_percent": 37.5,
            "memory_used_bytes": 1210 * GIB,
            "memory_total_bytes": 2048 * GIB,
            "network_rx_bytes_per_second": 1.8 * GIB,
            "network_tx_bytes_per_second": 1.2 * GIB,
            "ib_rx_bytes_per_second": 42 * GIB,
            "ib_tx_bytes_per_second": 41 * GIB,
            "ib_errors_per_second": 0.05 if receive_errors else 0.0,
        },
        "series": {
            "gpu_util_percent": _series(util, 10),
            "cpu_util_percent": _series(37, 6),
            "memory_used_bytes": _series(1210 * GIB, 40 * GIB),
            "network_rx_bytes_per_second": _series(1.8 * GIB, 0.4 * GIB),
            "network_tx_bytes_per_second": _series(1.2 * GIB, 0.3 * GIB),
            "ib_rx_bytes_per_second": _series(42 * GIB, 4 * GIB),
            "ib_tx_bytes_per_second": _series(41 * GIB, 4 * GIB),
        },
        "gpus": [_gpu(i, util - i * 3) for i in range(8)] if with_gpus else [],
        "ib_ports": [
            _ib_port(f"mlx5_{i}", receive_errors if i == 2 else 0, detail=with_gpus)
            for i in range(4)
        ],
    }


def _payload(nodes: Optional[List[Dict[str, Any]]] = None, **overrides: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "cluster_id": "pc_123",
        "cluster_display_name": "research-h100",
        "window": "1h",
        "step_seconds": 60,
        "as_of": "2026-11-26T19:30:00Z",
        "available": True,
        "unavailable_reason": None,
        "unavailable_code": None,
        "nodes": nodes
        if nodes is not None
        else [_node("gpu-01"), _node("gpu-02", util=41.0, state="cordoned")],
    }
    payload.update(overrides)
    return payload


def _status(*clusters: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "as_of": "2026-11-26T19:30:00Z",
        "clusters": [{"allocations": [], "sources": [], **c} for c in clusters],
    }


class _StubConfig:
    def __init__(self, team_id: Optional[str]) -> None:
        self.team_id = team_id


class _DummyAPIClient:
    """Serves `/factory/status` from ``status`` and metrics from ``responses`` in order."""

    def __init__(self, responses: List[Any], status: Optional[Any] = None) -> None:
        self._responses = responses
        self._status = status
        self.calls: List[Dict[str, Any]] = []

    def get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None, timeout: Any = None
    ) -> Dict[str, Any]:
        self.calls.append({"endpoint": endpoint, "params": params})
        if endpoint == STATUS_ENDPOINT:
            if self._status is None:
                raise NotFoundError("HTTP 404: Not Found")
            if isinstance(self._status, Exception):
                raise self._status
            return self._status
        served = sum(1 for c in self.calls if c["endpoint"] != STATUS_ENDPOINT)
        response = self._responses[min(served, len(self._responses)) - 1]
        if isinstance(response, Exception):
            raise response
        return response

    @property
    def metrics_calls(self) -> List[Dict[str, Any]]:
        return [c for c in self.calls if c["endpoint"] != STATUS_ENDPOINT]


@pytest.fixture(autouse=True)
def _console_width(monkeypatch: pytest.MonkeyPatch) -> None:
    # Rich reads COLUMNS once, when the module-level console is created.
    monkeypatch.setattr(factory_clusters.console, "width", 240)


def _install(
    monkeypatch: pytest.MonkeyPatch,
    *responses: Any,
    status: Optional[Any] = None,
    team_id: Optional[str] = None,
) -> _DummyAPIClient:
    dummy = _DummyAPIClient(list(responses), status)
    monkeypatch.setattr("prime_cli.commands.factory_clusters.APIClient", lambda: dummy)
    monkeypatch.setattr("prime_cli.commands.factory_clusters.Config", lambda: _StubConfig(team_id))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    return dummy


def _run(*args: str):
    result = runner.invoke(app, ["factory", "metrics", *args], env=TEST_ENV)
    return result, strip_ansi(result.output)


def test_client_sends_window_and_node() -> None:
    dummy = _DummyAPIClient([_payload(window="24h")])
    client = FactoryMetricsClient(dummy)  # type: ignore[arg-type]

    client.get_node_metrics("pc/123", window="24h")
    metrics = client.get_node_metrics("pc_123", node="gpu-01")

    assert dummy.calls == [
        {"endpoint": "/model-factory/clusters/pc%2F123/metrics/nodes", "params": {"window": "24h"}},
        {"endpoint": METRICS_ENDPOINT, "params": {"window": "1h", "node": "gpu-01"}},
    ]
    assert metrics.nodes[1].state == "cordoned"


def test_client_sends_current_only() -> None:
    """The server skips the range queries on this flag, so it has to reach it."""
    dummy = _DummyAPIClient([_payload(), _payload()])
    client = FactoryMetricsClient(dummy)  # type: ignore[arg-type]

    client.get_node_metrics("pc_123", current_only=True)
    client.get_node_metrics("pc_123")

    assert dummy.calls[0]["params"] == {"window": "1h", "current_only": "true"}
    assert "current_only" not in dummy.calls[1]["params"]


def test_client_accepts_null_points_and_missing_series() -> None:
    node = _node("gpu-01")
    node["series"] = {"gpu_util_percent": [{"timestamp": 1, "value": None}]}
    dummy = _DummyAPIClient([_payload(nodes=[node])])

    metrics = FactoryMetricsClient(dummy).get_node_metrics("pc_123")  # type: ignore[arg-type]

    assert metrics.nodes[0].series["gpu_util_percent"][0].value is None
    assert "cpu_util_percent" not in metrics.nodes[0].series


def test_client_rejects_shape_drift_and_unknown_schema() -> None:
    with pytest.raises(APIError, match="response shape"):
        FactoryMetricsClient(_DummyAPIClient([{"schema_version": 1}])).get_node_metrics("c")  # type: ignore[arg-type]
    with pytest.raises(APIError, match="schema version"):
        FactoryMetricsClient(_DummyAPIClient([_payload(schema_version=2)])).get_node_metrics("c")  # type: ignore[arg-type]


def test_table_shows_every_node_with_state_and_bars(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _payload())

    result, output = _run("pc_123")

    assert result.exit_code == 0, result.output
    assert "research-h100 · 2 nodes, 16 GPUs · Last hour, 1min steps" in output
    for header in (
        "NODE",
        "STATE",
        "GPU UTIL",
        "GPU MEM",
        "TEMP",
        "CPU",
        "NET IN/OUT",
        "IB IN/OUT",
    ):
        assert header in output
    assert "XID" not in output
    assert "ready" in output and "under inspection" in output
    assert "under inspection = pulled from scheduling by Prime" in output
    assert "cordoned" not in output
    assert "█" in output and "░" in output
    assert " 92%" in output and " 41%" in output
    assert "488/640 GiB" in output
    assert "1.8/1.2 GiB/s" in output
    assert "42/41 GiB/s" in output
    assert any(c in output for c in "▁▂▃▄▅▆▇")


def test_plain_mode_drops_bars_and_sparklines(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _payload())

    result, output = _run("pc_123", "--plain")

    assert result.exit_code == 0, result.output
    assert "gpu-01" in output and "92%" in output
    assert "488/640 GiB" in output and "42/41 GiB/s" in output
    assert not any(c in output for c in "█░▁▂▃▄▅▆▇")


def test_narrow_terminal_drops_bars_but_keeps_node_sparklines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _payload(), _payload(nodes=[_node("gpu-01", with_gpus=True)]))
    monkeypatch.setattr(factory_clusters.console, "width", 80)

    result, output = _run("pc_123")
    assert result.exit_code == 0, result.output
    assert " 92%" in output and "█" not in output and "…" not in output
    assert max(len(line) for line in output.splitlines()) <= 80
    assert " 76%" in output

    result, output = _run("pc_123", "--node", "gpu-01")
    assert result.exit_code == 0, result.output
    assert any(c in output for c in "▁▂▃▄▅▆▇") and "…" not in output
    assert max(len(line) for line in output.splitlines()) <= 80


def test_node_asks_the_server_and_shows_per_gpu_breakdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy = _install(monkeypatch, _payload(nodes=[_node("gpu-02", util=41.0, with_gpus=True)]))

    result, output = _run("pc_123", "--node", "gpu-02")

    assert result.exit_code == 0, result.output
    assert dummy.metrics_calls[0]["params"] == {"window": "1h", "node": "gpu-02"}
    assert "gpu-02 8× H100_80GB" in output
    assert "UTIL TREND" in output and "TEMP TREND" in output
    assert "61/80 GiB" in output and "1.2/2.0 TiB" in output and "488/640 GiB" in output
    assert "IB IN" in output and "42.0 GiB/s" in output
    gpu_rows = [line for line in output.splitlines() if line.strip().split(" ")[0] in "01234567"]
    assert len([r for r in gpu_rows if "°C" in r]) == 8


def test_node_lists_ib_ports_and_highlights_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(
        monkeypatch,
        _payload(nodes=[_node("gpu-02", with_gpus=True, receive_errors=3)]),
    )

    result, output = _run("pc_123", "--node", "gpu-02")

    assert result.exit_code == 0, result.output
    assert "IB PORT" in output and "ERRORS (WINDOW)" in output
    port_rows = [line for line in output.splitlines() if line.startswith("mlx5_")]
    assert len(port_rows) == 4
    assert all("active link up" in r and "400 Gb/s" in r and "10.5 GiB/s" in r for r in port_rows)
    assert "receive 3" in port_rows[2]
    assert "IB ERRORS" in output and "0.05/s" in output


def test_ib_errors_are_styled_red(monkeypatch: pytest.MonkeyPatch) -> None:
    from prime_cli.api.factory_metrics import IbPortMetrics

    clean = IbPortMetrics(device="mlx5_0", port="1", symbol_errors=0, receive_errors=0)
    bad = IbPortMetrics(device="mlx5_0", port="1", link_downed=2, receive_errors=1)
    assert factory_clusters._ib_errors(clean) == "[dim]0[/dim]"
    assert factory_clusters._ib_errors(bad) == "[bold red]link down 2, receive 1[/bold red]"


def test_unknown_keys_are_tolerated_and_not_rendered(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node("gpu-01", with_gpus=True)
    node["current"]["load_1m"] = 12.5
    node["series"]["disk_read_bytes_per_second"] = _series(5, 1)
    node["nfs_mounts"] = [{"path": "/data"}]
    node["gpus"][0]["sm_clock_hz"] = 1.9e9
    node["ib_ports"][0]["lid"] = 7
    payload = _payload(nodes=[node], storage={"nfs": True})
    _install(monkeypatch, payload)

    result, output = _run("pc_123", "--node", "gpu-01")

    assert result.exit_code == 0, result.output
    assert "load" not in output.lower() and "disk" not in output.lower()


def test_unknown_node_is_a_friendly_404(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, NotFoundError("HTTP 404: Not Found"))

    result, output = _run("pc_123", "--node", "gpu-99")

    assert result.exit_code == 3
    assert "Cluster 'pc_123' or its node 'gpu-99' was not found" in output


def test_json_is_raw_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _payload()
    payload["future_field"] = {"kept": True}
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "metrics", "pc_123", "--json"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == payload


def _without_series(item: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in item.items() if k != "series"}
    for key in ("gpus", "ib_ports"):
        if key in out:
            out[key] = [_without_series(child) for child in out[key]]
    return out


def test_json_current_only_drops_every_series(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _payload(nodes=[_node("gpu-01", with_gpus=True), _node("gpu-02")])
    payload["nodes"][0]["future_field"] = 1
    _install(monkeypatch, payload)

    result = runner.invoke(
        app, ["factory", "metrics", "pc_123", "--json", "--current-only"], env=TEST_ENV
    )

    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert '"series"' not in result.stdout
    assert data == {**payload, "nodes": [_without_series(n) for n in payload["nodes"]]}
    assert data["nodes"][0]["future_field"] == 1
    assert data["nodes"][0]["gpus"][0]["current"]["gpu_util_percent"] == 92.0
    assert "series" in payload["nodes"][0]


def test_table_current_only_hides_trends(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _payload(), _payload(nodes=[_node("gpu-01", with_gpus=True)]))

    result, output = _run("pc_123", "--current-only")
    assert result.exit_code == 0, result.output
    assert "TREND" not in output and not any(c in output for c in "▁▂▃▄▅▆▇")
    assert "488/640 GiB" in output and "█" in output

    result, output = _run("pc_123", "--node", "gpu-01", "--current-only")
    assert result.exit_code == 0, result.output
    assert "TREND" not in output and not any(c in output for c in "▁▂▃▄▅▆▇")
    assert "61/80 GiB" in output and "IB PORT" in output


@pytest.mark.parametrize(
    ("error", "code", "exit_code"),
    [
        (NotFoundError("HTTP 404: Not Found"), "not_found", 3),
        (UnauthorizedError("API key unauthorized."), "unauthorized", 4),
        (ServerError("HTTP 503: unavailable"), "metrics_unavailable", 5),
        (ServerError("HTTP 500: boom"), "server_error", 1),
    ],
)
def test_json_errors_are_structured_on_stderr(
    monkeypatch: pytest.MonkeyPatch, error: APIError, code: str, exit_code: int
) -> None:
    _install(monkeypatch, error)

    result = runner.invoke(app, ["factory", "metrics", "pc_123", "--json"], env=TEST_ENV)

    assert result.exit_code == exit_code
    assert result.stdout == ""
    body = json.loads(result.stderr)
    assert list(body) == ["error"] and body["error"]["code"] == code
    assert body["error"]["message"] and "[" not in body["error"]["message"]


def test_json_usage_errors_are_structured(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = _install(monkeypatch, _payload())

    result = runner.invoke(
        app,
        ["factory", "metrics", "pc_123", "-o", "json", "--window", "2h"],
        env=TEST_ENV,
    )

    assert result.exit_code == 2 and result.stdout == ""
    error = json.loads(result.stderr)["error"]
    assert error == {
        "code": "invalid_argument",
        "message": "Invalid --window '2h'. Choose one of: 1h, 6h, 24h, 7d.",
    }
    assert dummy.calls == []


@pytest.mark.parametrize(
    ("error", "exit_code"),
    [(NotFoundError("HTTP 404: Not Found"), 3), (ServerError("HTTP 503: unavailable"), 1)],
)
@pytest.mark.parametrize("selector", ["research h100", "2"])
def test_failed_status_lookup_does_not_send_a_name_as_id(
    monkeypatch: pytest.MonkeyPatch, selector: str, error: APIError, exit_code: int
) -> None:
    dummy = _install(monkeypatch, _payload(), status=error, team_id="team-1")

    result = runner.invoke(app, ["factory", "metrics", selector, "--json"], env=TEST_ENV)

    assert result.exit_code == exit_code
    error_body = json.loads(result.stderr)["error"]
    assert f"Could not resolve cluster '{selector}'" in error_body["message"]
    assert "Pass the cluster id instead" in error_body["message"]
    assert dummy.metrics_calls == []


def test_help_documents_current_only_errors_and_exit_codes() -> None:
    result = runner.invoke(app, ["factory", "metrics", "--help"], env=TEST_ENV)

    output = strip_ansi(result.output)
    assert result.exit_code == 0
    assert "--current-only: series is dropped" in output
    assert "metrics_unavailable" in output and "ambiguous_cluster" in output
    assert "5 metrics store unavailable (HTTP 503)" in output


def test_failed_status_lookup_still_passes_an_id_through(monkeypatch: pytest.MonkeyPatch) -> None:
    status = ServerError("HTTP 503: unavailable")
    dummy = _install(monkeypatch, _payload(), status=status, team_id="team-1")

    result, _ = _run("pc_123")

    assert result.exit_code == 0, result.output
    assert dummy.metrics_calls[0]["endpoint"] == METRICS_ENDPOINT


@pytest.mark.parametrize(
    ("reason", "code", "expected"),
    [
        ("No GPU exporter data yet.", "no_data", "Metrics are unavailable: No GPU exporter data"),
        (None, "no_cluster_label", "has not reported its metrics label yet"),
    ],
)
def test_unavailable_metrics_say_why(
    monkeypatch: pytest.MonkeyPatch, reason: Optional[str], code: str, expected: str
) -> None:
    _install(
        monkeypatch,
        _payload(nodes=[], available=False, unavailable_reason=reason, unavailable_code=code),
    )

    result, output = _run("pc_123")

    assert result.exit_code == 0, result.output
    assert expected in output


@pytest.mark.parametrize(
    ("error", "expected", "exit_code"),
    [
        (NotFoundError("HTTP 404: Not Found"), "was not found, or you are not a member", 3),
        (ServerError("HTTP 503: unavailable"), "the metrics store can't be reached", 5),
        (UnauthorizedError("API key unauthorized."), "need the factory read scope", 4),
        (ServerError("HTTP 500: boom"), "HTTP 500: boom", 1),
    ],
)
def test_api_errors_are_friendly(
    monkeypatch: pytest.MonkeyPatch, error: APIError, expected: str, exit_code: int
) -> None:
    _install(monkeypatch, error, team_id="team-1")

    result, output = _run("pc_123")

    assert result.exit_code == exit_code
    assert result.stdout.startswith("Error:")
    assert expected in output
    assert "Traceback" not in output


def test_not_found_without_a_team_points_at_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A teamless 404 is usually an unresolved display name, not a missing cluster."""
    _install(monkeypatch, NotFoundError("HTTP 404: Not Found"))

    result, output = _run("dell-h100-nj")

    assert result.exit_code == 3
    assert "No team is selected" in output
    assert "prime switch" in output


@pytest.mark.parametrize("selector", ["research-h100", "2", "pc_123"])
def test_cluster_resolves_by_name_index_or_id(
    monkeypatch: pytest.MonkeyPatch, selector: str
) -> None:
    status = _status(
        {"id": "pc_other", "display_name": "other"},
        {"id": "pc_123", "display_name": "research-h100"},
    )
    dummy = _install(monkeypatch, _payload(), status=status, team_id="team-1")

    result, _ = _run(selector)

    assert result.exit_code == 0, result.output
    assert dummy.metrics_calls[0]["endpoint"] == METRICS_ENDPOINT


def test_a_cluster_id_skips_the_status_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolving costs a `factory status` call; an id needs no resolving."""
    cluster_id = "o9g4nhwfh9rfr84rnua0yotj"
    dummy = _install(monkeypatch, _payload(), status=_status(), team_id="team-1")

    result, _ = _run(cluster_id)

    assert result.exit_code == 0, result.output
    assert [c["endpoint"] for c in dummy.calls] == [
        f"/model-factory/clusters/{cluster_id}/metrics/nodes"
    ]


def test_an_unknown_id_falls_back_to_the_status_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shape heuristic can be wrong; the 404 has to recover, not surface."""
    status = _status({"id": "pc_123", "display_name": "abcdefghij0123456789"})
    dummy = _install(
        monkeypatch,
        NotFoundError("HTTP 404: Not Found"),
        _payload(),
        status=status,
        team_id="team-1",
    )

    result, _ = _run("abcdefghij0123456789")

    assert result.exit_code == 0, result.output
    assert STATUS_ENDPOINT in [c["endpoint"] for c in dummy.calls]
    assert dummy.metrics_calls[-1]["endpoint"] == METRICS_ENDPOINT


def test_a_resolved_name_does_not_look_up_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    """A name already cost a status call; a 404 after it is the answer, not a retry."""
    status = _status({"id": "pc_123", "display_name": "research-h100"})
    dummy = _install(
        monkeypatch, NotFoundError("HTTP 404: Not Found"), status=status, team_id="team-1"
    )

    result, output = _run("research-h100", "--node", "gpu-99")

    assert result.exit_code == 3
    assert len([c for c in dummy.calls if c["endpoint"] == STATUS_ENDPOINT]) == 1
    assert "node 'gpu-99'" in output


def test_a_node_404_without_a_team_still_points_at_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cause is the same with --node: nothing can resolve a name without a team."""
    _install(monkeypatch, NotFoundError("HTTP 404: Not Found"))

    result, output = _run("dell-h100-nj", "--node", "gpu-01")

    assert result.exit_code == 3
    assert "No team is selected" in output
    assert "gpu-01" in output


def test_unmatched_selector_or_no_team_is_used_as_id(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = _install(monkeypatch, _payload())

    result, _ = _run("pc_123")

    assert result.exit_code == 0, result.output
    assert [c["endpoint"] for c in dummy.calls] == [METRICS_ENDPOINT]


def test_ambiguous_or_id_less_name_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    status = _status({"id": "a", "display_name": "dup"}, {"id": "b", "display_name": "dup"})
    _install(monkeypatch, _payload(), status=status, team_id="team-1")
    result, output = _run("dup")
    assert result.exit_code == 2 and "matches multiple clusters" in output
    result = runner.invoke(app, ["factory", "metrics", "dup", "--json"], env=TEST_ENV)
    assert result.exit_code == 2
    assert json.loads(result.stderr)["error"]["code"] == "ambiguous_cluster"

    _install(monkeypatch, _payload(), status=_status({"display_name": "old"}), team_id="team-1")
    result, output = _run("old")
    assert result.exit_code == 2 and "does not report cluster ids" in output


def test_invalid_window_and_watch_json_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = _install(monkeypatch, _payload())

    result, output = _run("pc_123", "--window", "2h")
    assert result.exit_code == 2 and "Invalid --window '2h'" in output

    result, output = _run("pc_123", "--watch", "--json")
    assert result.exit_code == 2 and "--watch cannot be combined" in output
    assert dummy.calls == []


def _stop_after(monkeypatch: pytest.MonkeyPatch, count: int) -> List[int]:
    sleeps: List[int] = []

    def fake_sleep(seconds: int) -> None:
        sleeps.append(seconds)
        if len(sleeps) == count:
            raise KeyboardInterrupt

    monkeypatch.setattr(factory_clusters.time, "sleep", fake_sleep)
    return sleeps


def test_watch_refreshes_and_survives_a_failed_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = _install(
        monkeypatch,
        _payload(),
        ServerError("HTTP 503: unavailable"),
        _payload(nodes=[_node("gpu-01", util=12.0)]),
    )
    sleeps = _stop_after(monkeypatch, 3)

    result, output = _run("pc_123", "--watch", "--interval", "5", "--plain")

    assert result.exit_code == 0, result.output
    assert len(dummy.calls) == 3
    assert sleeps == [5, 5, 5]
    assert "Refresh failed: Node metrics are unavailable right now" in output
    assert "refreshing every 5s" in output
    assert "12%" in output


def test_watch_live_mode_renders_latest_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = _install(monkeypatch, _payload(), _payload(nodes=[_node("gpu-01", util=12.0)]))
    _stop_after(monkeypatch, 2)

    result, output = _run("pc_123", "--watch")

    assert result.exit_code == 0, result.output
    assert len(dummy.calls) == 2
    assert " 12%" in output and "refreshing every 10s" in output


def test_watch_keeps_node_view_when_node_disappears(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(
        monkeypatch,
        _payload(nodes=[_node("gpu-02", with_gpus=True)]),
        _payload(nodes=[_node("gpu-01")]),
    )
    _stop_after(monkeypatch, 2)

    result, output = _run("pc_123", "--watch", "--node", "gpu-02", "--plain")

    assert result.exit_code == 0, result.output
    assert "Node 'gpu-02' is not reporting metrics." in output
    assert "NODE" not in output
    assert output.count("research-h100 · gpu-02") == 2
    assert "1 node" not in output


def _points(*values: Optional[float]) -> List[MetricPoint]:
    return [MetricPoint(timestamp=i, value=v) for i, v in enumerate(values)]


def test_sparkline_scales_downsamples_and_keeps_gaps() -> None:
    assert sparkline(_points(0, 50, 100), lo=0, hi=100) == "▁▅█"
    assert sparkline(_points(5, 5, 5)) == "▁▁▁"
    assert len(sparkline(_points(*range(200)), width=20)) == 20
    assert sparkline([]) == ""
    assert sparkline(_points(None, None)) == ""
    assert sparkline(_points(100, None, 0), lo=0, hi=100) == "█ ▁"
    assert sparkline(_points(100, 100, None, None), width=2, lo=0, hi=100) == "█ "


def test_sparkline_uses_the_step_grid_for_missing_timestamps() -> None:
    points = [MetricPoint(timestamp=t, value=100) for t in (0, 300, 1200)]
    assert sparkline(points, lo=0, hi=100, step_seconds=300) == "██  █"
    assert sparkline(points, lo=0, hi=100) == "███"
