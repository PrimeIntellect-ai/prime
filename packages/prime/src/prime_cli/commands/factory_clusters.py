"""`prime factory clusters` — per-node metrics for dedicated clusters."""

import json
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, NoReturn, Optional, Sequence, Tuple

import typer
from rich.console import Group, RenderableType
from rich.live import Live
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from ..api.factory import FactoryClient
from ..api.factory_metrics import (
    METRICS_WINDOWS,
    ClusterNodeMetrics,
    FactoryMetricsClient,
    IbPortMetrics,
    MetricPoint,
    NodeMetrics,
)
from ..client import APIClient, APIError, ServerError
from ..core import Config
from ..core.client import NotFoundError, UnauthorizedError
from ..utils import get_console, is_plain_mode, json_output_help, output_data_as_json

# `cluster_metrics` is registered on the factory app itself (factory.py), not
# on a sub-app of its own: it sits alongside `status`, `nodes` and `workloads`
# as one more view of the same clusters.
console = get_console()

METRICS_JSON_HELP = json_output_help(
    ". = {schema_version, cluster_id, cluster_display_name, window, step_seconds,",
    "     as_of, available, unavailable_reason, unavailable_code, nodes[]}",
    ".nodes[] = {name, gpu_type, gpu_count, state, current, series, gpus[], ib_ports[]}",
    ".gpus[] = {index, current, series} — filled only with --node",
    ".ib_ports[] = {device, port, state, physical_state, rate_bytes_per_second,",
    "               symbol_errors, link_downed, receive_errors, current, series}",
    "              (current/series only with --node)",
    ".current = {gpu_util_percent, gpu_memory_used_bytes, gpu_memory_total_bytes,",
    "            gpu_temperature_celsius, gpu_power_watts, cpu_util_percent,",
    "            memory_used_bytes, memory_total_bytes, network_rx_bytes_per_second,",
    "            network_tx_bytes_per_second, ib_rx_bytes_per_second,",
    "            ib_tx_bytes_per_second, ib_errors_per_second}",
    ".series = {<metric>: [{timestamp, value}]} keyed by .current field names, on the",
    "          step_seconds grid; a key is absent without data, value is null for a gap",
    "Newer servers may add keys; they are passed through here and not rendered.",
    "--node: the server returns only that node in .nodes[]",
    "--current-only: series is dropped from nodes[], gpus[] and ib_ports[]",
    "Errors go to stderr as {error: {code, message}}; code: invalid_argument,",
    "  ambiguous_cluster, not_found, unauthorized, metrics_unavailable, server_error",
    "Exit codes (all output modes): 0 ok, 1 other error, 2 invalid usage,",
    "  3 cluster/node not found or no access, 4 unauthorized,",
    "  5 metrics store unavailable (HTTP 503)",
)

ERROR_EXIT_CODES = {
    "invalid_argument": 2,
    "ambiguous_cluster": 2,
    "not_found": 3,
    "unauthorized": 4,
    "metrics_unavailable": 5,
    "server_error": 1,
}
# A selector outside this shape (spaces, punctuation) or all digits can't be a cluster id.
ID_LIKE = re.compile(r"[A-Za-z0-9_-]+")
# A cuid: lowercase alphanumeric, no separators. Display names in the fleet all
# carry hyphens, so this tells the two apart well enough to skip a lookup —
# and a wrong guess only costs the 404 that `metrics` already handles.
CLUSTER_ID = re.compile(r"[a-z0-9]{20,32}")

SPARK_CHARS = "▁▂▃▄▅▆▇█"
BAR_WIDTH = 8
WIDE_COLUMNS = 120
MISSING = "[dim]—[/dim]"
# Same styles and "under inspection" wording as `prime factory nodes`.
STATE_STYLES = {
    "ready": "green",
    "not_ready": "red",
    "offline": "red",
    "cordoned": "yellow",
    "draining": "dim",
    "unknown": "dim",
}
IB_STATE_STYLES = {
    "active": "green",
    "active_defer": "green",
    "armed": "yellow",
    "init": "yellow",
    "down": "red",
}
# Same labels as the dashboard's time range picker.
WINDOW_LABELS = {
    "1h": "Last hour",
    "6h": "Last 6 hours",
    "24h": "Last 24 hours",
    "7d": "Last 7 days",
}
UNAVAILABLE_FALLBACK = {
    "no_cluster_label": "the cluster has not reported its metrics label yet",
    "no_data": "no metrics were reported for this cluster in this window",
}


def _wide() -> bool:
    """Bars and the fleet trend column need room; narrow terminals get numbers only."""
    return not is_plain_mode() and console.width >= WIDE_COLUMNS


def _spark_width() -> int:
    return 20 if _wide() else 12


def _pct(used: Optional[float], total: Optional[float]) -> Optional[float]:
    if used is None or not total:
        return None
    return 100.0 * used / total


def _level_style(percent: float) -> str:
    if percent >= 90:
        return "red"
    if percent >= 70:
        return "yellow"
    return "green"


def _bar(percent: Optional[float], label: Optional[str] = None) -> str:
    """`████░░░░  52%`, colored by level; just the number when not `_wide()`."""
    if percent is None:
        return MISSING
    clamped = max(0.0, min(100.0, percent))
    text = label if label is not None else f"{clamped:3.0f}%"
    style = _level_style(clamped)
    if is_plain_mode():
        return text
    if not _wide():
        return f"[{style}]{text}[/{style}]"
    filled = round(clamped / 100 * BAR_WIDTH)
    return f"[{style}]{'█' * filled}[/{style}][dim]{'░' * (BAR_WIDTH - filled)}[/dim] {text}"


def _on_grid(points: Sequence[MetricPoint], step_seconds: Optional[int]) -> List[Optional[float]]:
    """Lay points on the ``step_seconds`` grid so missing timestamps become gaps."""
    if not points or not step_seconds or step_seconds <= 0:
        return [p.value for p in points]
    start = min(p.timestamp for p in points)
    values: List[Optional[float]] = [None] * (
        (max(p.timestamp for p in points) - start) // step_seconds + 1
    )
    for p in points:
        values[min(len(values) - 1, round((p.timestamp - start) / step_seconds))] = p.value
    return values


def sparkline(
    points: Sequence[MetricPoint],
    width: int = 20,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
    step_seconds: Optional[int] = None,
) -> str:
    """Render a series as unicode blocks, bucket-averaged down to ``width``.

    ``lo``/``hi`` pin the scale (0–100 for percentages); otherwise the
    series' own range is used so small swings stay visible. Gaps (``None``
    values, timestamps missing from the ``step_seconds`` grid, or buckets
    with no values) render as spaces, never as zero.
    """
    values = _on_grid(points, step_seconds)
    if len(values) > width:
        size = len(values) / width
        buckets = (values[int(i * size) : int((i + 1) * size)] for i in range(width))
        values = [
            sum(known) / len(known) if known else None
            for known in ([v for v in chunk if v is not None] for chunk in buckets)
        ]
    known = [v for v in values if v is not None]
    if not known:
        return ""
    low = min(known) if lo is None else lo
    high = max(known) if hi is None else hi
    span = high - low
    top = len(SPARK_CHARS) - 1
    return "".join(
        " "
        if v is None
        else SPARK_CHARS[0 if span <= 0 else max(0, min(top, round((v - low) / span * top)))]
        for v in values
    )


_BINARY_UNITS = ((4, "TiB"), (3, "GiB"), (2, "MiB"), (1, "KiB"))


def _bytes(value: float) -> str:
    for power, unit in _BINARY_UNITS:
        if abs(value) >= 1024**power:
            scaled = value / 1024**power
            return f"{scaled:.0f} {unit}" if scaled >= 100 else f"{scaled:.1f} {unit}"
    return f"{value:.0f} B"


def _pair(first: Optional[float], second: Optional[float], suffix: str = "") -> str:
    """`61/80 GiB`, `1.8/1.2 GiB/s` — both values in the unit that suits the larger."""
    known = [v for v in (first, second) if v is not None]
    if not known:
        return MISSING
    ref = max(known)
    scale, unit = next(((1024**p, u) for p, u in _BINARY_UNITS if ref >= 1024**p), (1, "B"))

    def fmt(v: Optional[float]) -> str:
        if v is None:
            return "—"
        return f"{v / scale:.1f}" if ref / scale < 10 else f"{v / scale:.0f}"

    return f"{fmt(first)}/{fmt(second)} {unit}{suffix}"


def _amount(value: Optional[float]) -> str:
    return MISSING if value is None else _bytes(value)


def _rate(value: Optional[float]) -> str:
    return MISSING if value is None else f"{_bytes(value)}/s"


def _num(value: Optional[float], unit: str) -> str:
    return MISSING if value is None else f"{value:.0f}{unit}"


def _state(state: str) -> str:
    style = STATE_STYLES.get(state)
    label = "under inspection" if state == "cordoned" else escape(state.replace("_", " "))
    return f"[{style}]{label}[/{style}]" if style else label


def _used_of_total(used: Optional[float], total: Optional[float], bar: bool = True) -> str:
    """Used/total with a level bar; used alone when the total is unknown."""
    percent = _pct(used, total)
    if percent is None:
        return _amount(used)
    label = _pair(used, total)
    if not bar:
        style = _level_style(percent)
        return label if is_plain_mode() else f"[{style}]{label}[/{style}]"
    return _bar(percent, label)


def _errors(value: Optional[float], label: str) -> str:
    if value is None:
        return MISSING
    if value == 0:
        return f"[dim]0{label}[/dim]"
    shown = f"{value:g}{label}"
    return f"[bold red]{shown}[/bold red]"


def _trend(series: Dict[str, List[MetricPoint]], key: str, step: int, percent: bool = True) -> str:
    line = sparkline(
        series.get(key, []),
        width=_spark_width(),
        lo=0 if percent else None,
        hi=100 if percent else None,
        step_seconds=step,
    )
    return f"[cyan]{line}[/cyan]" if line else MISSING


def _nodes_table(metrics: ClusterNodeMetrics, trends: bool) -> Table:
    show_trend = trends and _wide()
    table = Table(
        show_header=True,
        header_style="bold",
        box=None,
        pad_edge=False,
        collapse_padding=not show_trend,
    )
    table.add_column("NODE", style="bold", no_wrap=True)
    table.add_column("STATE", min_width=10)
    table.add_column("GPU UTIL", no_wrap=True)
    table.add_column("GPU MEM", justify="right", no_wrap=True)
    table.add_column("TEMP", justify="right", no_wrap=True)
    table.add_column("POWER", justify="right", no_wrap=True)
    table.add_column("CPU", no_wrap=True)
    table.add_column("MEM", no_wrap=True)
    table.add_column("NET IN/OUT", justify="right")
    table.add_column("IB IN/OUT", justify="right")
    if show_trend:
        table.add_column("GPU UTIL TREND", no_wrap=True)
    for node in metrics.nodes:
        cur = node.current
        row = [
            escape(node.name),
            _state(node.state),
            _bar(cur.gpu_util_percent),
            _used_of_total(cur.gpu_memory_used_bytes, cur.gpu_memory_total_bytes, bar=False)
            if _wide() or is_plain_mode()
            else _bar(_pct(cur.gpu_memory_used_bytes, cur.gpu_memory_total_bytes)),
            _num(cur.gpu_temperature_celsius, "°C"),
            _num(cur.gpu_power_watts, " W"),
            _bar(cur.cpu_util_percent),
            _bar(_pct(cur.memory_used_bytes, cur.memory_total_bytes)),
            _pair(cur.network_rx_bytes_per_second, cur.network_tx_bytes_per_second, "/s"),
            _pair(cur.ib_rx_bytes_per_second, cur.ib_tx_bytes_per_second, "/s"),
        ]
        if show_trend:
            row.append(_trend(node.series, "gpu_util_percent", metrics.step_seconds))
        table.add_row(*row)
    return table


def _node_summary(node: NodeMetrics, step: int, trends: bool) -> Table:
    show_trend = trends and not is_plain_mode()
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="bold")
    table.add_column(no_wrap=True)
    if show_trend:
        table.add_column(no_wrap=True)
    cur = node.current
    rows = [
        ("STATE", _state(node.state), None, False),
        ("CPU", _bar(cur.cpu_util_percent), "cpu_util_percent", True),
        (
            "MEMORY",
            _used_of_total(cur.memory_used_bytes, cur.memory_total_bytes),
            "memory_used_bytes",
            False,
        ),
        (
            "GPU MEMORY",
            _used_of_total(cur.gpu_memory_used_bytes, cur.gpu_memory_total_bytes),
            "gpu_memory_used_bytes",
            False,
        ),
        ("NET IN", _rate(cur.network_rx_bytes_per_second), "network_rx_bytes_per_second", False),
        ("NET OUT", _rate(cur.network_tx_bytes_per_second), "network_tx_bytes_per_second", False),
    ]
    if node.ib_ports or cur.ib_rx_bytes_per_second is not None:
        rows += [
            ("IB IN", _rate(cur.ib_rx_bytes_per_second), "ib_rx_bytes_per_second", False),
            ("IB OUT", _rate(cur.ib_tx_bytes_per_second), "ib_tx_bytes_per_second", False),
            ("IB ERRORS", _errors(cur.ib_errors_per_second, "/s"), "ib_errors_per_second", False),
        ]
    for label, value, key, percent in rows:
        cells = [label, value]
        if show_trend:
            cells.append(_trend(node.series, key, step, percent) if key else "")
        table.add_row(*cells)
    return table


def _gpus_table(node: NodeMetrics, step: int, trends: bool) -> Table:
    show_trend = trends and not is_plain_mode()
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("GPU", justify="right")
    table.add_column("UTIL", no_wrap=True)
    table.add_column("MEMORY", no_wrap=True)
    table.add_column("TEMP", justify="right")
    table.add_column("POWER", justify="right")
    if show_trend:
        table.add_column("UTIL TREND", no_wrap=True)
        table.add_column("TEMP TREND", no_wrap=True)
    for gpu in sorted(node.gpus, key=lambda g: g.index):
        cur = gpu.current
        row = [
            str(gpu.index),
            _bar(cur.gpu_util_percent),
            _used_of_total(cur.gpu_memory_used_bytes, cur.gpu_memory_total_bytes),
            _num(cur.gpu_temperature_celsius, "°C"),
            _num(cur.gpu_power_watts, " W"),
        ]
        if show_trend:
            row.append(_trend(gpu.series, "gpu_util_percent", step))
            row.append(_trend(gpu.series, "gpu_temperature_celsius", step, percent=False))
        table.add_row(*row)
    return table


def _ib_state(port: IbPortMetrics) -> str:
    if port.state is None:
        return MISSING
    style = IB_STATE_STYLES.get(port.state)
    label = escape(port.state.replace("_", " "))
    text = f"[{style}]{label}[/{style}]" if style else label
    if port.physical_state:
        text += f" [dim]{escape(port.physical_state.replace('_', ' '))}[/dim]"
    return text


def _link_rate(value: Optional[float]) -> str:
    return MISSING if value is None else f"{value * 8 / 1e9:.0f} Gb/s"


def _ib_errors(port: IbPortMetrics) -> str:
    counts = [
        ("symbol", port.symbol_errors),
        ("link down", port.link_downed),
        ("receive", port.receive_errors),
    ]
    if all(count is None for _, count in counts):
        return MISSING
    raised = [f"{name} {count}" for name, count in counts if count]
    if not raised:
        return "[dim]0[/dim]"
    return f"[bold red]{', '.join(raised)}[/bold red]"


def _ib_ports_table(node: NodeMetrics) -> Table:
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("IB PORT", no_wrap=True)
    table.add_column("STATE", no_wrap=True)
    table.add_column("RATE", justify="right", no_wrap=True)
    table.add_column("IN", justify="right", no_wrap=True)
    table.add_column("OUT", justify="right", no_wrap=True)
    table.add_column("ERRORS (WINDOW)")
    for port in node.ib_ports:
        table.add_row(
            escape(f"{port.device}/{port.port}"),
            _ib_state(port),
            _link_rate(port.rate_bytes_per_second),
            _rate(port.current.ib_rx_bytes_per_second),
            _rate(port.current.ib_tx_bytes_per_second),
            _ib_errors(port),
        )
    return table


def _duration(seconds: int) -> str:
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}min"
    return f"{seconds}s"


def _header(
    metrics: ClusterNodeMetrics, node: Optional[NodeMetrics], node_name: Optional[str]
) -> str:
    name = escape(metrics.cluster_display_name)
    parts = [f"[bold]{name}[/bold]"]
    if node is not None:
        gpus = f"{node.gpu_count}× {node.gpu_type}" if node.gpu_count and node.gpu_type else ""
        parts.append(f"[bold]{escape(node.name)}[/bold]" + (f" {escape(gpus)}" if gpus else ""))
    elif node_name is not None:
        parts.append(f"[bold]{escape(node_name)}[/bold]")
    elif metrics.nodes:
        count = len(metrics.nodes)
        gpus = sum(n.gpu_count or 0 for n in metrics.nodes)
        summary = f"{count} node{'s' if count != 1 else ''}"
        parts.append(f"{summary}, {gpus} GPUs" if gpus else summary)
    window = WINDOW_LABELS.get(metrics.window, escape(metrics.window))
    parts.append(f"{window}, {_duration(metrics.step_seconds)} steps")
    return " · ".join(parts)


def _inspection_legend() -> Text:
    from .factory import UNDER_INSPECTION_LEGEND

    return Text.from_markup(f"[dim]{UNDER_INSPECTION_LEGEND}[/dim]")


def render_metrics(
    metrics: ClusterNodeMetrics, node_name: Optional[str], trends: bool = True
) -> List[RenderableType]:
    """Everything one refresh prints, as a list so plain mode can print it piecewise."""
    node = next((n for n in metrics.nodes if n.name == node_name), None) if node_name else None
    out: List[RenderableType] = [Text.from_markup(_header(metrics, node, node_name))]
    if not metrics.available:
        reason = escape(
            metrics.unavailable_reason
            or UNAVAILABLE_FALLBACK.get(metrics.unavailable_code or "", "no metrics to show")
        )
        out.append(Text.from_markup(f"[yellow]Metrics are unavailable: {reason}[/yellow]"))
        return out
    if not metrics.nodes:
        out.append(Text.from_markup("[dim]No node metrics reported in this window.[/dim]"))
        return out
    if node_name is None:
        out.append(_nodes_table(metrics, trends))
        if any(n.state == "cordoned" for n in metrics.nodes):
            out.append(_inspection_legend())
        return out
    if node is None:
        missing = escape(node_name)
        out.append(Text.from_markup(f"[yellow]Node '{missing}' is not reporting metrics.[/yellow]"))
        return out
    out.append(_node_summary(node, metrics.step_seconds, trends))
    if node.state == "cordoned":
        out.append(_inspection_legend())
    if node.gpus:
        out.append(Text())
        out.append(_gpus_table(node, metrics.step_seconds, trends))
    else:
        out.append(Text.from_markup("[dim]No per-GPU breakdown reported for this node.[/dim]"))
    if node.ib_ports:
        out.append(Text())
        out.append(_ib_ports_table(node))
    return out


def _fail(json_mode: bool, code: str, message: str) -> NoReturn:
    """JSON mode: one ``{"error": {code, message}}`` object on stderr; else the human line."""
    if json_mode:
        typer.echo(json.dumps({"error": {"code": code, "message": message}}), err=True)
    else:
        console.print(f"[red]Error:[/red] {escape(message)}")
    raise typer.Exit(ERROR_EXIT_CODES[code])


def _error_code(error: APIError) -> str:
    if isinstance(error, UnauthorizedError):
        return "unauthorized"
    if isinstance(error, NotFoundError):
        return "not_found"
    if isinstance(error, ServerError) and str(error).startswith("HTTP 503"):
        return "metrics_unavailable"
    return "server_error"


def _error_message(
    error: APIError, cluster: str, node: Optional[str], has_team: bool = True
) -> Tuple[str, str]:
    """``(code, message)`` for a failed metrics request; the message is plain text."""
    code = _error_code(error)
    if code == "unauthorized":
        return code, f"{error} Cluster metrics need the factory read scope."
    if code == "not_found":
        if node is not None:
            return code, (
                f"Cluster '{cluster}' or its node '{node}' was not found, "
                "or you are not a member of the cluster's team."
            )
        if not has_team:
            # Without a team the selector went to the server as an id, because
            # display names and indexes only exist in `prime factory status`.
            # Say that, rather than leaving it reading as a permissions problem.
            return code, (
                f"Cluster '{cluster}' was not found. No team is selected, so a display "
                "name or index can't be resolved — run `prime switch` to select a team, "
                "or pass --team <team_id>."
            )
        return code, (
            f"Cluster '{cluster}' was not found, or you are not a member of its team. "
            "Run `prime factory status` to see your clusters."
        )
    if code == "metrics_unavailable":
        return code, "Node metrics are unavailable right now: the metrics store can't be reached."
    return code, str(error)


def _resolve_cluster_id(selector: str, team: Optional[str], json_mode: bool) -> str:
    """Map a display name or 1-based index from `prime factory status` to a cluster id.

    Anything that doesn't match (or no team context) is passed through as an id.
    If the status lookup fails, only a selector that could be an id is passed through.
    """
    team_id = team or Config().team_id
    if not team_id:
        return selector
    try:
        clusters = FactoryClient(APIClient()).get_status(team_id).clusters
    except APIError as e:
        if ID_LIKE.fullmatch(selector) and not selector.isdigit():
            return selector
        code = _error_code(e)
        _fail(
            json_mode,
            "server_error" if code == "metrics_unavailable" else code,
            f"Could not resolve cluster '{selector}' by name or index: "
            f"`prime factory status` failed ({e}). Pass the cluster id instead.",
        )
    if any(c.id == selector for c in clusters):
        return selector
    matches = [c for c in clusters if c.display_name == selector]
    if not matches and selector.isascii() and selector.isdigit():
        if 1 <= int(selector) <= len(clusters):
            matches = [clusters[int(selector) - 1]]
    if len(matches) > 1:
        _fail(
            json_mode,
            "ambiguous_cluster",
            f"'{selector}' matches multiple clusters. "
            "Use its 1-based index from `prime factory status` or its id instead.",
        )
    if not matches:
        return selector
    if matches[0].id is None:
        _fail(
            json_mode,
            "invalid_argument",
            "This server does not report cluster ids yet; pass the cluster id instead of its name.",
        )
    return matches[0].id


def _strip_series(raw: Dict[str, Any]) -> Dict[str, Any]:
    """The raw response without ``series`` on nodes, GPUs and IB ports."""

    def drop(item: Dict[str, Any]) -> Dict[str, Any]:
        out = {k: v for k, v in item.items() if k != "series"}
        for key in ("gpus", "ib_ports"):
            if key in out:
                out[key] = [drop(child) for child in out[key]]
        return out

    return {**raw, "nodes": [drop(node) for node in raw["nodes"]]}


def _watch_footer(interval: int, error: Optional[str]) -> str:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    footer = f"[dim]Updated {stamp} · refreshing every {interval}s · Ctrl+C to exit[/dim]"
    if error:
        footer = f"[yellow]Refresh failed: {escape(error)}[/yellow]\n" + footer
    return footer


def _refresh(fetch: Any, previous: ClusterNodeMetrics, cluster: str, node: Optional[str]):
    """Keep the last good frame on a failed refresh instead of exiting."""
    try:
        return fetch(), None
    except APIError as e:
        return previous, _error_message(e, cluster, node)[1]


def cluster_metrics(
    cluster: str = typer.Argument(
        ..., help="Cluster id, or display name / 1-based index from `prime factory status`"
    ),
    window: str = typer.Option(
        "1h",
        "--window",
        help="Time window: "
        + ", ".join(f"{w} ({WINDOW_LABELS[w].lower()})" for w in METRICS_WINDOWS),
    ),
    node: Optional[str] = typer.Option(
        None, "--node", help="Show the per-GPU breakdown for one node"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Print the API response as JSON (same as --output json)"
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
    current_only: bool = typer.Option(
        False,
        "--current-only",
        help="Latest values only: drop series from JSON and trend columns from tables",
    ),
    watch: bool = typer.Option(False, "--watch", help="Refresh until interrupted"),
    interval: int = typer.Option(
        10, "--interval", "-n", min=2, help="Seconds between refreshes with --watch"
    ),
    team: Optional[str] = typer.Option(
        None,
        "--team",
        "-t",
        hidden=True,
        help="Team ID override for name/index lookup (defaults to the selected account context)",
    ),
) -> None:
    """Show per-node GPU, CPU, memory and network metrics for a dedicated cluster."""
    if json_output:
        output = "json"
    if output not in ("table", "json"):
        _fail(False, "invalid_argument", f"Invalid output format: {output}")
    json_mode = output == "json"
    if window not in METRICS_WINDOWS:
        _fail(
            json_mode,
            "invalid_argument",
            f"Invalid --window '{window}'. Choose one of: {', '.join(METRICS_WINDOWS)}.",
        )
    if watch and json_mode:
        _fail(json_mode, "invalid_argument", "--watch cannot be combined with JSON output.")

    has_team = bool(team or Config().team_id)
    client = FactoryMetricsClient(APIClient())
    trends = not current_only

    # Resolving a name or index costs a `prime factory status` call (1.5s warm,
    # 13s cold) on top of the metrics call. A cluster id needs no lookup at all,
    # so spend it only on a selector that cannot already be one. The 404 below
    # is the safety net if the guess is wrong, and it is cheap: the server
    # rejects an unknown id before it queries anything.
    cluster_id = (
        cluster if CLUSTER_ID.fullmatch(cluster) else _resolve_cluster_id(cluster, team, json_mode)
    )

    def fetch() -> ClusterNodeMetrics:
        return client.get_node_metrics(cluster_id, window, node, current_only)

    try:
        metrics = fetch()
    except NotFoundError as e:
        resolved = _resolve_cluster_id(cluster, team, json_mode)
        if resolved == cluster_id:
            _fail(json_mode, *_error_message(e, cluster, node, has_team))
        cluster_id = resolved
        try:
            metrics = fetch()
        except APIError as retry:
            _fail(json_mode, *_error_message(retry, cluster, node, has_team))
    except APIError as e:
        _fail(json_mode, *_error_message(e, cluster, node, has_team))

    if json_mode:
        raw = metrics.raw_response
        output_data_as_json(_strip_series(raw) if current_only else raw, console)
        return

    if not watch:
        for item in render_metrics(metrics, node, trends):
            console.print(item)
        return

    error: Optional[str] = None
    try:
        if is_plain_mode():
            while True:
                for item in render_metrics(metrics, node, trends):
                    console.print(item)
                console.print(_watch_footer(interval, error))
                time.sleep(interval)
                metrics, error = _refresh(fetch, metrics, cluster, node)
        else:
            with Live(console=console, auto_refresh=False) as live:
                while True:
                    frame = [
                        *render_metrics(metrics, node, trends),
                        Text(),
                        Text.from_markup(_watch_footer(interval, error)),
                    ]
                    live.update(Group(*frame), refresh=True)
                    time.sleep(interval)
                    metrics, error = _refresh(fetch, metrics, cluster, node)
    except KeyboardInterrupt:
        return
