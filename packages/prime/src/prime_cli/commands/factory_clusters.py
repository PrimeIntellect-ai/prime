"""`prime factory clusters` — per-node metrics for dedicated clusters."""

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

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
    MetricPoint,
    NodeMetrics,
)
from ..client import APIClient, APIError, ServerError
from ..core import Config
from ..core.client import NotFoundError, UnauthorizedError
from ..utils import PlainTyper, get_console, is_plain_mode, json_output_help, output_data_as_json

app = PlainTyper(help="Dedicated Model Factory clusters", no_args_is_help=True)
console = get_console()

METRICS_JSON_HELP = json_output_help(
    ". = {schema_version, cluster_id, cluster_display_name, window, step_seconds,",
    "     as_of, available, unavailable_reason, unavailable_code, nodes[]}",
    ".nodes[] = {name, gpu_type, gpu_count, state, current, series, gpus[]}",
    ".gpus[] = {index, current, series} — filled only with --node",
    ".current = {gpu_util_percent, gpu_memory_used_bytes, gpu_temperature_celsius,",
    "            gpu_power_watts, cpu_util_percent, memory_used_bytes, memory_total_bytes,",
    "            network_rx_bytes_per_second, network_tx_bytes_per_second}",
    ".series = {<metric>: [{timestamp, value}]} keyed by .current field names;",
    "          a key is absent without data, value is null for a gap",
    "--node: the server returns only that node in .nodes[]",
)

SPARK_CHARS = "▁▂▃▄▅▆▇█"
BAR_WIDTH = 8
WIDE_COLUMNS = 120
MISSING = "[dim]—[/dim]"
STATE_STYLES = {"ready": "green", "not_ready": "red", "cordoned": "yellow"}
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


def sparkline(
    points: Sequence[MetricPoint],
    width: int = 20,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
) -> str:
    """Render a series as unicode blocks, bucket-averaged down to ``width``.

    ``lo``/``hi`` pin the scale (0–100 for percentages); otherwise the
    series' own range is used so small swings stay visible. Gaps (``None``
    values, or buckets with no values) render as spaces, never as zero.
    """
    values: List[Optional[float]] = [p.value for p in points]
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


def _bytes(value: Optional[float]) -> str:
    if value is None:
        return "—"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" or value >= 100 else f"{value:.1f} {unit}"
        value /= 1024
    return "—"


def _used_of_total(used: Optional[float], total: Optional[float]) -> str:
    """`61/80 GiB` — both values in the unit that suits the total."""
    if used is None or not total:
        return "—"
    scale, unit = next(
        (1024**p, u) for p, u in ((4, "TiB"), (3, "GiB"), (2, "MiB"), (0, "B")) if total >= 1024**p
    )

    def fmt(v: float) -> str:
        return f"{v / scale:.1f}" if total / scale < 10 else f"{v / scale:.0f}"

    return f"{fmt(used)}/{fmt(total)} {unit}"


def _amount(value: Optional[float]) -> str:
    return MISSING if value is None else _bytes(value)


def _rate(value: Optional[float]) -> str:
    return MISSING if value is None else f"{_bytes(value)}/s"


def _num(value: Optional[float], unit: str) -> str:
    return MISSING if value is None else f"{value:.0f}{unit}"


def _state(state: str) -> str:
    style = STATE_STYLES.get(state)
    label = escape(state.replace("_", " "))
    return f"[{style}]{label}[/{style}]" if style else label


def _used_bar(used: Optional[float], total: Optional[float]) -> str:
    percent = _pct(used, total)
    if percent is None:
        return MISSING
    return _bar(percent, _used_of_total(used, total))


def _trend(series: Dict[str, List[MetricPoint]], key: str, percent: bool = True) -> str:
    line = sparkline(
        series.get(key, []),
        width=_spark_width(),
        lo=0 if percent else None,
        hi=100 if percent else None,
    )
    return f"[cyan]{line}[/cyan]" if line else MISSING


def _nodes_table(metrics: ClusterNodeMetrics) -> Table:
    show_trend = _wide()
    table = Table(
        show_header=True,
        header_style="bold",
        box=None,
        pad_edge=False,
        collapse_padding=not show_trend,
    )
    table.add_column("NODE", style="bold", no_wrap=True)
    table.add_column("STATE", no_wrap=True)
    table.add_column("GPU UTIL", no_wrap=True)
    table.add_column("GPU MEM", justify="right", no_wrap=True)
    table.add_column("TEMP", justify="right")
    table.add_column("POWER", justify="right")
    table.add_column("CPU", no_wrap=True)
    table.add_column("MEM", no_wrap=True)
    table.add_column("NET RX", justify="right", no_wrap=True)
    table.add_column("NET TX", justify="right", no_wrap=True)
    if show_trend:
        table.add_column(f"GPU UTIL {metrics.window}", no_wrap=True)
    for node in metrics.nodes:
        cur = node.current
        row = [
            escape(node.name),
            _state(node.state),
            _bar(cur.gpu_util_percent),
            _amount(cur.gpu_memory_used_bytes),
            _num(cur.gpu_temperature_celsius, "°C"),
            _num(cur.gpu_power_watts, " W"),
            _bar(cur.cpu_util_percent),
            _bar(_pct(cur.memory_used_bytes, cur.memory_total_bytes)),
            _rate(cur.network_rx_bytes_per_second),
            _rate(cur.network_tx_bytes_per_second),
        ]
        if show_trend:
            row.append(_trend(node.series, "gpu_util_percent"))
        table.add_row(*row)
    return table


def _node_summary(node: NodeMetrics) -> Table:
    show_trend = not is_plain_mode()
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
            _used_bar(cur.memory_used_bytes, cur.memory_total_bytes),
            "memory_used_bytes",
            False,
        ),
        ("NET RX", _rate(cur.network_rx_bytes_per_second), "network_rx_bytes_per_second", False),
        ("NET TX", _rate(cur.network_tx_bytes_per_second), "network_tx_bytes_per_second", False),
    ]
    for label, value, key, percent in rows:
        cells = [label, value]
        if show_trend:
            cells.append(_trend(node.series, key, percent) if key else "")
        table.add_row(*cells)
    return table


def _gpus_table(node: NodeMetrics, window: str) -> Table:
    show_trend = not is_plain_mode()
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("GPU", justify="right")
    table.add_column("UTIL", no_wrap=True)
    table.add_column("MEMORY", justify="right", no_wrap=True)
    table.add_column("TEMP", justify="right")
    table.add_column("POWER", justify="right")
    if show_trend:
        table.add_column(f"UTIL {window}", no_wrap=True)
        table.add_column(f"TEMP {window}", no_wrap=True)
    for gpu in sorted(node.gpus, key=lambda g: g.index):
        cur = gpu.current
        row = [
            str(gpu.index),
            _bar(cur.gpu_util_percent),
            _amount(cur.gpu_memory_used_bytes),
            _num(cur.gpu_temperature_celsius, "°C"),
            _num(cur.gpu_power_watts, " W"),
        ]
        if show_trend:
            row.append(_trend(gpu.series, "gpu_util_percent"))
            row.append(_trend(gpu.series, "gpu_temperature_celsius", percent=False))
        table.add_row(*row)
    return table


def _header(metrics: ClusterNodeMetrics, node: Optional[NodeMetrics]) -> str:
    name = escape(metrics.cluster_display_name)
    parts = [f"[bold]{name}[/bold]"]
    if node is not None:
        gpus = f"{node.gpu_count}× {node.gpu_type}" if node.gpu_count and node.gpu_type else ""
        parts.append(f"[bold]{escape(node.name)}[/bold]" + (f" {escape(gpus)}" if gpus else ""))
    elif metrics.nodes:
        count = len(metrics.nodes)
        gpus = sum(n.gpu_count or 0 for n in metrics.nodes)
        summary = f"{count} node{'s' if count != 1 else ''}"
        parts.append(f"{summary}, {gpus} GPUs" if gpus else summary)
    parts.append(f"window {metrics.window}")
    return " · ".join(parts)


def render_metrics(metrics: ClusterNodeMetrics, node_name: Optional[str]) -> List[RenderableType]:
    """Everything one refresh prints, as a list so plain mode can print it piecewise."""
    node = next((n for n in metrics.nodes if n.name == node_name), None) if node_name else None
    out: List[RenderableType] = [Text.from_markup(_header(metrics, node))]
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
        out.append(_nodes_table(metrics))
        return out
    if node is None:
        missing = escape(node_name)
        out.append(Text.from_markup(f"[yellow]Node '{missing}' is not reporting metrics.[/yellow]"))
        return out
    out.append(_node_summary(node))
    if node.gpus:
        out.append(Text())
        out.append(_gpus_table(node, metrics.window))
    else:
        out.append(Text.from_markup("[dim]No per-GPU breakdown reported for this node.[/dim]"))
    return out


def _error_message(error: APIError, cluster: str, node: Optional[str]) -> str:
    detail = str(error)
    if isinstance(error, UnauthorizedError):
        return f"{escape(detail)} Cluster metrics need the factory read scope."
    if isinstance(error, NotFoundError):
        if node is not None:
            return (
                f"Cluster '{escape(cluster)}' or its node '{escape(node)}' was not found, "
                "or you are not a member of the cluster's team."
            )
        return (
            f"Cluster '{escape(cluster)}' was not found, or you are not a member of its team. "
            "Run `prime factory status` to see your clusters."
        )
    if isinstance(error, ServerError) and detail.startswith("HTTP 503"):
        return "Node metrics are unavailable right now: the metrics store can't be reached."
    return escape(detail)


def _resolve_cluster_id(selector: str, team: Optional[str], err_console: Any) -> str:
    """Map a display name or 1-based index from `prime factory status` to a cluster id.

    Anything that doesn't match (or no team context) is passed through as an id.
    """
    team_id = team or Config().team_id
    if not team_id:
        return selector
    try:
        clusters = FactoryClient(APIClient()).get_status(team_id).clusters
    except APIError:
        return selector
    if any(c.id == selector for c in clusters):
        return selector
    matches = [c for c in clusters if c.display_name == selector]
    if not matches and selector.isascii() and selector.isdigit():
        if 1 <= int(selector) <= len(clusters):
            matches = [clusters[int(selector) - 1]]
    if len(matches) > 1:
        err_console.print(
            f"[red]Error:[/red] '{escape(selector)}' matches multiple clusters. "
            "Use its 1-based index from `prime factory status` or its id instead."
        )
        raise typer.Exit(1)
    if not matches:
        return selector
    if matches[0].id is None:
        err_console.print(
            "[red]Error:[/red] This server does not report cluster ids yet; "
            "pass the cluster id instead of its name."
        )
        raise typer.Exit(1)
    return matches[0].id


def _watch_footer(interval: int, error: Optional[str]) -> str:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    footer = f"[dim]Updated {stamp} · refreshing every {interval}s · Ctrl+C to exit[/dim]"
    if error:
        footer = f"[yellow]Refresh failed: {error}[/yellow]\n" + footer
    return footer


def _refresh(fetch: Any, previous: ClusterNodeMetrics, cluster: str, node: Optional[str]):
    """Keep the last good frame on a failed refresh instead of exiting."""
    try:
        return fetch(), None
    except APIError as e:
        return previous, _error_message(e, cluster, node)


@app.command(name="metrics", no_args_is_help=True, epilog=METRICS_JSON_HELP)
def cluster_metrics(
    cluster: str = typer.Argument(
        ..., help="Cluster id, or display name / 1-based index from `prime factory status`"
    ),
    window: str = typer.Option("1h", "--window", help=f"Time window: {', '.join(METRICS_WINDOWS)}"),
    node: Optional[str] = typer.Option(
        None, "--node", help="Show the per-GPU breakdown for one node"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Print the API response as JSON (same as --output json)"
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
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
        console.print(f"[red]Error:[/red] Invalid output format: {escape(output)}")
        raise typer.Exit(1)
    err_console = get_console(stderr=True) if output == "json" else console
    if window not in METRICS_WINDOWS:
        err_console.print(
            f"[red]Error:[/red] Invalid --window '{escape(window)}'. "
            f"Choose one of: {', '.join(METRICS_WINDOWS)}."
        )
        raise typer.Exit(1)
    if watch and output == "json":
        err_console.print("[red]Error:[/red] --watch cannot be combined with JSON output.")
        raise typer.Exit(1)

    cluster_id = _resolve_cluster_id(cluster, team, err_console)
    client = FactoryMetricsClient(APIClient())

    def fetch() -> ClusterNodeMetrics:
        return client.get_node_metrics(cluster_id, window, node)

    try:
        metrics = fetch()
    except APIError as e:
        err_console.print(f"[red]Error:[/red] {_error_message(e, cluster, node)}")
        raise typer.Exit(1)

    if output == "json":
        output_data_as_json(metrics.raw_response, console)
        return

    if not watch:
        for item in render_metrics(metrics, node):
            console.print(item)
        return

    error: Optional[str] = None
    try:
        if is_plain_mode():
            while True:
                for item in render_metrics(metrics, node):
                    console.print(item)
                console.print(_watch_footer(interval, error))
                time.sleep(interval)
                metrics, error = _refresh(fetch, metrics, cluster, node)
        else:
            with Live(console=console, auto_refresh=False) as live:
                while True:
                    frame = [
                        *render_metrics(metrics, node),
                        Text(),
                        Text.from_markup(_watch_footer(interval, error)),
                    ]
                    live.update(Group(*frame), refresh=True)
                    time.sleep(interval)
                    metrics, error = _refresh(fetch, metrics, cluster, node)
    except KeyboardInterrupt:
        return
