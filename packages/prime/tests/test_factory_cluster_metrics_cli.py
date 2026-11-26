"""Tests for `prime factory clusters metrics` (per-node cluster metrics)."""

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
            "gpu_temperature_celsius": 58 + index,
            "gpu_power_watts": 610,
        },
        "series": {
            "gpu_util_percent": _series(util, 8),
            "gpu_temperature_celsius": _series(58 + index, 3),
        },
    }


def _node(
    name: str, util: float = 92.0, state: str = "ready", with_gpus: bool = False
) -> Dict[str, Any]:
    return {
        "name": name,
        "gpu_type": "H100_80GB",
        "gpu_count": 8,
        "state": state,
        "current": {
            "gpu_util_percent": util,
            "gpu_memory_used_bytes": 488 * GIB,
            "gpu_temperature_celsius": 66,
            "gpu_power_watts": 4880,
            "cpu_util_percent": 37.5,
            "memory_used_bytes": 1210 * GIB,
            "memory_total_bytes": 2048 * GIB,
            "network_rx_bytes_per_second": 1.8 * GIB,
            "network_tx_bytes_per_second": 1.2 * GIB,
        },
        "series": {
            "gpu_util_percent": _series(util, 10),
            "cpu_util_percent": _series(37, 6),
            "memory_used_bytes": _series(1210 * GIB, 40 * GIB),
            "network_rx_bytes_per_second": _series(1.8 * GIB, 0.4 * GIB),
            "network_tx_bytes_per_second": _series(1.2 * GIB, 0.3 * GIB),
        },
        "gpus": [_gpu(i, util - i * 3) for i in range(8)] if with_gpus else [],
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

    def __init__(self, responses: List[Any], status: Optional[Dict[str, Any]] = None) -> None:
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
    status: Optional[Dict[str, Any]] = None,
    team_id: Optional[str] = None,
) -> _DummyAPIClient:
    dummy = _DummyAPIClient(list(responses), status)
    monkeypatch.setattr("prime_cli.commands.factory_clusters.APIClient", lambda: dummy)
    monkeypatch.setattr("prime_cli.commands.factory_clusters.Config", lambda: _StubConfig(team_id))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    return dummy


def _run(*args: str):
    result = runner.invoke(app, ["factory", "clusters", "metrics", *args], env=TEST_ENV)
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
    assert "research-h100 · 2 nodes, 16 GPUs · window 1h" in output
    for header in ("NODE", "STATE", "GPU UTIL", "GPU MEM", "TEMP", "POWER", "CPU", "NET RX"):
        assert header in output
    assert "XID" not in output
    assert "ready" in output and "cordoned" in output
    assert "█" in output and "░" in output
    assert " 92%" in output and " 41%" in output
    assert "488 GiB" in output
    assert "1.8 GiB/s" in output
    assert any(c in output for c in "▁▂▃▄▅▆▇")


def test_plain_mode_drops_bars_and_sparklines(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _payload())

    result, output = _run("pc_123", "--plain")

    assert result.exit_code == 0, result.output
    assert "gpu-01" in output and "92%" in output
    assert not any(c in output for c in "█░▁▂▃▄▅▆▇")


def test_narrow_terminal_drops_bars_but_keeps_node_sparklines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _payload(), _payload(nodes=[_node("gpu-01", with_gpus=True)]))
    monkeypatch.setattr(factory_clusters.console, "width", 80)

    result, output = _run("pc_123")
    assert result.exit_code == 0, result.output
    assert " 92%" in output and "█" not in output and "…" not in output

    result, output = _run("pc_123", "--node", "gpu-01")
    assert result.exit_code == 0, result.output
    assert any(c in output for c in "▁▂▃▄▅▆▇") and "…" not in output


def test_node_asks_the_server_and_shows_per_gpu_breakdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy = _install(monkeypatch, _payload(nodes=[_node("gpu-02", util=41.0, with_gpus=True)]))

    result, output = _run("pc_123", "--node", "gpu-02")

    assert result.exit_code == 0, result.output
    assert dummy.metrics_calls[0]["params"] == {"window": "1h", "node": "gpu-02"}
    assert "gpu-02 8× H100_80GB" in output
    assert "UTIL 1h" in output and "TEMP 1h" in output
    assert "61.0 GiB" in output and "1.2/2.0 TiB" in output
    gpu_rows = [line for line in output.splitlines() if line.strip().split(" ")[0] in "01234567"]
    assert len([r for r in gpu_rows if "°C" in r]) == 8


def test_unknown_node_is_a_friendly_404(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, NotFoundError("HTTP 404: Not Found"))

    result, output = _run("pc_123", "--node", "gpu-99")

    assert result.exit_code == 1
    assert "Cluster 'pc_123' or its node 'gpu-99' was not found" in output


def test_json_is_raw_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _payload()
    payload["future_field"] = {"kept": True}
    _install(monkeypatch, payload)

    result = runner.invoke(
        app, ["factory", "clusters", "metrics", "pc_123", "--json"], env=TEST_ENV
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == payload


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
    ("error", "expected"),
    [
        (NotFoundError("HTTP 404: Not Found"), "was not found, or you are not a member"),
        (ServerError("HTTP 503: unavailable"), "the metrics store can't be reached"),
        (UnauthorizedError("API key unauthorized."), "need the factory read scope"),
    ],
)
def test_api_errors_are_friendly(
    monkeypatch: pytest.MonkeyPatch, error: APIError, expected: str
) -> None:
    _install(monkeypatch, error)

    result, output = _run("pc_123")

    assert result.exit_code == 1
    assert expected in output
    assert "Traceback" not in output


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


def test_unmatched_selector_or_no_team_is_used_as_id(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = _install(monkeypatch, _payload())

    result, _ = _run("pc_123")

    assert result.exit_code == 0, result.output
    assert [c["endpoint"] for c in dummy.calls] == [METRICS_ENDPOINT]


def test_ambiguous_or_id_less_name_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    status = _status({"id": "a", "display_name": "dup"}, {"id": "b", "display_name": "dup"})
    _install(monkeypatch, _payload(), status=status, team_id="team-1")
    result, output = _run("dup")
    assert result.exit_code == 1 and "matches multiple clusters" in output

    _install(monkeypatch, _payload(), status=_status({"display_name": "old"}), team_id="team-1")
    result, output = _run("old")
    assert result.exit_code == 1 and "does not report cluster ids" in output


def test_invalid_window_and_watch_json_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = _install(monkeypatch, _payload())

    result, output = _run("pc_123", "--window", "2h")
    assert result.exit_code == 1 and "Invalid --window '2h'" in output

    result, output = _run("pc_123", "--watch", "--json")
    assert result.exit_code == 1 and "--watch cannot be combined" in output
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
