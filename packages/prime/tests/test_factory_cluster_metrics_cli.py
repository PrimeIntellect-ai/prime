"""Tests for `prime factory clusters metrics` (per-node cluster metrics)."""

import json
import math
from typing import Any, Dict, List, Optional

import pytest
from prime_cli.api.factory_metrics import FactoryMetricsClient, MetricPoint
from prime_cli.client import APIError
from prime_cli.commands import factory_clusters
from prime_cli.commands.factory_clusters import sparkline
from prime_cli.core.client import NotFoundError
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


def _series(base: float, amp: float, n: int = 60) -> List[Dict[str, float]]:
    return [
        {"timestamp": 1_764_000_000 + i * 60, "value": base + amp * math.sin(i / 6)}
        for i in range(n)
    ]


def _gpu(index: int, util: float, xid: int = 0) -> Dict[str, Any]:
    return {
        "index": index,
        "uuid": f"GPU-{index:04d}",
        "current": {
            "gpu_util_percent": util,
            "gpu_memory_used_bytes": 61 * GIB,
            "gpu_memory_total_bytes": 80 * GIB,
            "gpu_temperature_celsius": 58 + index,
            "gpu_power_watts": 610,
            "xid_errors": xid,
        },
        "series": {
            "gpu_util_percent": _series(util, 8),
            "gpu_temperature_celsius": _series(58 + index, 3),
        },
    }


def _node(name: str, util: float = 92.0, xid: int = 0, **overrides: Any) -> Dict[str, Any]:
    node: Dict[str, Any] = {
        "name": name,
        "gpu_type": "H100_80GB",
        "gpu_count": 8,
        "current": {
            "gpu_util_percent": util,
            "gpu_memory_used_bytes": 488 * GIB,
            "gpu_memory_total_bytes": 640 * GIB,
            "gpu_temperature_celsius": 66,
            "gpu_power_watts": 4880,
            "xid_errors": xid,
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
        "gpus": [_gpu(i, util - i * 3, xid if i == 3 else 0) for i in range(8)],
    }
    node.update(overrides)
    return node


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
        "nodes": nodes
        if nodes is not None
        else [_node("gpu-01"), _node("gpu-02", util=41.0, xid=2)],
    }
    payload.update(overrides)
    return payload


class _DummyAPIClient:
    def __init__(self, responses: List[Any]) -> None:
        self._responses = responses
        self.calls: List[Dict[str, Any]] = []

    def get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None, timeout: Any = None
    ) -> Dict[str, Any]:
        self.calls.append({"endpoint": endpoint, "params": params})
        response = self._responses[min(len(self.calls), len(self._responses)) - 1]
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture(autouse=True)
def _console_width(monkeypatch: pytest.MonkeyPatch) -> None:
    # Rich reads COLUMNS once, when the module-level console is created.
    monkeypatch.setattr(factory_clusters.console, "width", 240)


def _install(monkeypatch: pytest.MonkeyPatch, *responses: Any) -> _DummyAPIClient:
    dummy = _DummyAPIClient(list(responses))
    monkeypatch.setattr("prime_cli.commands.factory_clusters.APIClient", lambda: dummy)
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    return dummy


def _run(*args: str):
    result = runner.invoke(app, ["factory", "clusters", "metrics", *args], env=TEST_ENV)
    return result, strip_ansi(result.output)


def test_client_calls_endpoint_with_window() -> None:
    dummy = _DummyAPIClient([_payload(window="24h")])
    metrics = FactoryMetricsClient(dummy).get_node_metrics("pc/123", window="24h")  # type: ignore[arg-type]

    assert dummy.calls == [
        {"endpoint": "/model-factory/clusters/pc%2F123/metrics/nodes", "params": {"window": "24h"}}
    ]
    assert metrics.nodes[1].current.xid_errors == 2
    assert metrics.nodes[0].gpus[0].series["gpu_util_percent"][0].value == pytest.approx(92.0)


def test_client_rejects_shape_drift_and_unknown_schema() -> None:
    with pytest.raises(APIError, match="response shape"):
        FactoryMetricsClient(_DummyAPIClient([{"schema_version": 1}])).get_node_metrics("c")  # type: ignore[arg-type]
    with pytest.raises(APIError, match="schema version"):
        FactoryMetricsClient(_DummyAPIClient([_payload(schema_version=2)])).get_node_metrics("c")  # type: ignore[arg-type]


def test_table_shows_every_node_with_bars_and_xid(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _payload())

    result, output = _run("pc_123")

    assert result.exit_code == 0, result.output
    assert "research-h100 · 2 nodes, 16 GPUs · window 1h" in output
    for header in ("NODE", "GPU UTIL", "GPU MEM", "TEMP", "POWER", "XID", "CPU", "NET RX"):
        assert header in output
    assert "gpu-01" in output and "gpu-02" in output
    assert "█" in output and "░" in output
    assert " 92%" in output and " 41%" in output
    assert "1.8 GiB/s" in output
    assert " 2 " in output
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
    _install(monkeypatch, _payload())
    monkeypatch.setattr(factory_clusters.console, "width", 80)

    result, output = _run("pc_123")
    assert result.exit_code == 0, result.output
    assert " 92%" in output and "█" not in output and "…" not in output

    result, output = _run("pc_123", "--node", "gpu-01")
    assert result.exit_code == 0, result.output
    assert any(c in output for c in "▁▂▃▄▅▆▇") and "…" not in output


def test_node_shows_per_gpu_breakdown(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _payload())

    result, output = _run("pc_123", "--node", "gpu-02")

    assert result.exit_code == 0, result.output
    assert "gpu-02 8× H100_80GB" in output
    assert "UTIL 1h" in output and "TEMP 1h" in output
    assert "61/80 GiB" in output
    assert "1.2/2.0 TiB" in output
    gpu_rows = [line for line in output.splitlines() if line.strip().split(" ")[0] in "01234567"]
    assert len([r for r in gpu_rows if "°C" in r]) == 8


def test_unknown_node_lists_available_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _payload())

    result, output = _run("pc_123", "--node", "gpu-99")

    assert result.exit_code == 1
    assert "No node named 'gpu-99'" in output
    assert "gpu-01, gpu-02" in output


def test_json_is_raw_passthrough_and_node_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _payload()
    payload["future_field"] = {"kept": True}
    _install(monkeypatch, payload)

    result = runner.invoke(
        app, ["factory", "clusters", "metrics", "pc_123", "--json"], env=TEST_ENV
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == payload

    result = runner.invoke(
        app,
        ["factory", "clusters", "metrics", "pc_123", "--json", "--node", "gpu-01"],
        env=TEST_ENV,
    )
    assert [n["name"] for n in json.loads(result.stdout)["nodes"]] == ["gpu-01"]


def test_unavailable_metrics_say_why(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(
        monkeypatch,
        _payload(nodes=[], available=False, unavailable_reason="no GPU exporter data yet"),
    )

    result, output = _run("pc_123")

    assert result.exit_code == 0, result.output
    assert "Metrics are unavailable: no GPU exporter data yet" in output


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (NotFoundError("HTTP 404: Not Found"), "was not found, or you don't have access"),
        (APIError("HTTP 403: Forbidden"), "You don't have access to metrics"),
        (APIError("HTTP 503: upstream down"), "temporarily unavailable"),
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


def test_invalid_window_and_watch_json_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = _install(monkeypatch, _payload())

    result, output = _run("pc_123", "--window", "2h")
    assert result.exit_code == 1 and "Invalid --window '2h'" in output

    result, output = _run("pc_123", "--watch", "--json")
    assert result.exit_code == 1 and "--watch cannot be combined" in output
    assert dummy.calls == []


def test_watch_refreshes_and_survives_a_failed_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = _install(
        monkeypatch,
        _payload(),
        APIError("HTTP 503: upstream down"),
        _payload(nodes=[_node("gpu-01", util=12.0)]),
    )
    sleeps: List[int] = []

    def fake_sleep(seconds: int) -> None:
        sleeps.append(seconds)
        if len(sleeps) == 3:
            raise KeyboardInterrupt

    monkeypatch.setattr(factory_clusters.time, "sleep", fake_sleep)

    result, output = _run("pc_123", "--watch", "--interval", "5", "--plain")

    assert result.exit_code == 0, result.output
    assert len(dummy.calls) == 3
    assert sleeps == [5, 5, 5]
    assert "Refresh failed: Node metrics are temporarily unavailable" in output
    assert "refreshing every 5s" in output
    assert "12%" in output


def test_watch_live_mode_renders_latest_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = _install(monkeypatch, _payload(), _payload(nodes=[_node("gpu-01", util=12.0)]))
    sleeps: List[int] = []

    def fake_sleep(seconds: int) -> None:
        sleeps.append(seconds)
        if len(sleeps) == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(factory_clusters.time, "sleep", fake_sleep)

    result, output = _run("pc_123", "--watch")

    assert result.exit_code == 0, result.output
    assert len(dummy.calls) == 2
    assert " 12%" in output and "refreshing every 10s" in output


def test_sparkline_scales_and_downsamples() -> None:
    points = [MetricPoint(timestamp=i, value=v) for i, v in enumerate([0, 50, 100])]
    assert sparkline(points, lo=0, hi=100) == "▁▅█"
    assert sparkline([MetricPoint(timestamp=0, value=5)] * 3) == "▁▁▁"
    assert len(sparkline([MetricPoint(timestamp=i, value=i) for i in range(200)], width=20)) == 20
    assert sparkline([]) == ""
