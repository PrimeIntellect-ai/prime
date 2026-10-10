"""`prime factory` — Model Factory fleet status."""

from datetime import datetime
from typing import Any, List, Optional

import typer
from rich.markup import escape as rich_escape
from rich.table import Table

from ..api.factory import FactoryClient, FactoryCluster, FactoryPool, FactorySource
from ..client import APIClient, APIError
from ..core import Config
from ..utils import (
    PlainTyper,
    get_console,
    human_age,
    json_output_help,
    output_data_as_json,
    validate_output_format,
)

app = PlainTyper(
    help="View your team's dedicated Model Factory clusters and GPU allocations.",
    no_args_is_help=True,
)
console = get_console()

FACTORY_STATUS_JSON_HELP = json_output_help(
    ". = {schema_version, as_of, clusters[]}",
    ".clusters[] = {display_name, gpu_type, total_gpus, status,",
    "                 unassigned_gpus, unknown_gpus, pools[], sources[]}",
    ".pools[] = {type, reserved_gpus, in_use_gpus, idle_inside_gpus, unknown_gpus}",
    ".sources[] = {kind, status, observed_at}",
)

IN_USE_NOTE = "in-use = GPUs held by running jobs (not GPU-activity measurements)"

# Plain-language names for source kinds shown to users. The internal enum
# values (e.g. "capacity") never appear in table output.
FRIENDLY_DATA_NAMES = {
    "capacity": "node data",
    "training": "training data",
    "inference": "inference data",
    "slurm": "scheduler data",
}


def _friendly_data_name(kind: str) -> str:
    return FRIENDLY_DATA_NAMES.get(kind, f"{rich_escape(kind)} data")


def _last_seen_phrase(sources: List[FactorySource]) -> str:
    """Render `<data> last seen <age> ago` for degraded sources, plain words only."""
    names: List[str] = []
    oldest: Optional[datetime] = None
    for source in sources:
        name = _friendly_data_name(source.kind)
        if name not in names:
            names.append(name)
        if source.observed_at and (oldest is None or source.observed_at < oldest):
            oldest = source.observed_at
    phrase = " and ".join(names) if names else "data"
    if oldest is not None:
        return f"{phrase} last seen {human_age(oldest)} ago"
    return f"{phrase} unavailable"


def _pool_is_available(pool: FactoryPool, source_status: dict) -> bool:
    """A pool row renders only with complete evidence behind a fresh source."""
    if source_status.get(pool.type, "ok") != "ok":
        return False
    return (
        pool.reserved_gpus is not None
        and pool.in_use_gpus is not None
        and pool.idle_inside_gpus is not None
    )


def _styled_status(status: Optional[str]) -> str:
    if status == "online":
        return "[green]online[/green]"
    if status == "offline":
        return "[red]offline[/red]"
    return rich_escape(status or "unknown")


class _ClusterState:
    """Per-cluster rendering facts shared by the CLUSTERS and WORKLOADS sections."""

    def __init__(self, cluster: FactoryCluster) -> None:
        source_status = {s.kind: s.status for s in cluster.sources}
        self.capacity_ok = source_status.get("capacity", "ok") == "ok"
        degraded = [s for s in cluster.sources if s.status != "ok"]

        # Split pools by index so duplicate payload rows cannot be
        # misclassified by model equality.
        self.renderable: List[FactoryPool] = []
        suppressed_idx: List[int] = []
        for i, pool in enumerate(cluster.pools):
            if self.capacity_ok and _pool_is_available(pool, source_status):
                self.renderable.append(pool)
            else:
                suppressed_idx.append(i)
        self.suppressed = [cluster.pools[i] for i in suppressed_idx]

        # Sources involved in the suppressed breakdown; anything else
        # degraded still surfaces as a freshness phrase in the CLUSTERS line.
        involved: set = {p.type for p in self.suppressed}
        if not self.capacity_ok:
            involved.add("capacity")
        self.remaining_degraded = [s for s in degraded if s.kind not in involved]
        self.involved_sources = [s for s in degraded if s.kind in involved]


def _cluster_label(cluster: FactoryCluster, index: int, multi: bool) -> str:
    label = rich_escape(cluster.display_name)
    if multi:
        label = f"[cyan]\\[{index}][/cyan] {label}"
    return f"[bold]{label}[/bold]"


def _cluster_header(
    cluster: FactoryCluster, index: int, multi: bool, freshness: Optional[str]
) -> str:
    parts: List[str] = [_cluster_label(cluster, index, multi)]

    gpu_bits = [str(cluster.total_gpus)] if cluster.total_gpus is not None else []
    if cluster.gpu_type:
        gpu_bits.append(rich_escape(cluster.gpu_type))
    if gpu_bits:
        parts.append(" ".join(gpu_bits) + " GPUs")
    parts.append(_styled_status(cluster.status))
    if freshness:
        parts.append(freshness)
    return " · ".join(parts)


def _render_pool_table(pools: List[FactoryPool]) -> Table:
    table = Table(show_header=True, header_style="bold", show_lines=False)
    table.add_column("POOL", style="cyan")
    table.add_column("RESERVED", style="white", justify="right")
    table.add_column("IN USE", style="green", justify="right")
    table.add_column("IDLE INSIDE", style="blue", justify="right")
    table.add_column("UNKNOWN", style="yellow", justify="right")

    for pool in pools:
        table.add_row(
            rich_escape(pool.type),
            str(pool.reserved_gpus),
            str(pool.in_use_gpus),
            str(pool.idle_inside_gpus),
            str(pool.unknown_gpus) if pool.unknown_gpus is not None else "-",
        )
    return table


def _render_workloads_section(
    clusters: List[FactoryCluster], states: List["_ClusterState"], multi: bool
) -> None:
    footnote = False
    for index, (cluster, state) in enumerate(zip(clusters, states), start=1):
        if index > 1:
            console.print()
        console.print(_cluster_label(cluster, index, multi))

        if state.suppressed:
            names = (
                "pool"
                if not state.renderable and len(state.suppressed) == len(cluster.pools)
                else ", ".join(rich_escape(p.type) for p in state.suppressed)
            )
            line = f"{names} breakdown unavailable"
            if state.involved_sources:
                line += f" — {_last_seen_phrase(state.involved_sources)}"
            console.print(f"[yellow]{line}[/yellow]")

        if state.renderable:
            console.print(_render_pool_table(state.renderable))
            footnote = True

        # Unassigned and unknown GPUs are cluster-level facts, not pool rows.
        if state.capacity_ok and cluster.unassigned_gpus is not None:
            console.print(f"unassigned: {cluster.unassigned_gpus} GPUs")
        if state.capacity_ok and cluster.unknown_gpus is not None:
            console.print(f"unknown: {cluster.unknown_gpus} GPUs")

        if (
            not state.renderable
            and not state.suppressed
            and not (
                state.capacity_ok
                and (cluster.unassigned_gpus is not None or cluster.unknown_gpus is not None)
            )
        ):
            console.print("[dim]no pools reported[/dim]")

    if footnote:
        console.print()
        console.print(f"[dim]{IN_USE_NOTE}[/dim]")


def _select_cluster_indices(
    clusters: List[FactoryCluster], selector: str, err_console: Any
) -> List[int]:
    """Select clusters by display name (exact) or 1-based index.

    Returns the selected indices so both the parsed models (table mode) and
    the raw API response clusters (JSON mode) can be filtered consistently.
    Errors go to ``err_console`` so the JSON stream on stdout stays clean.
    """
    matches = [i for i, c in enumerate(clusters) if c.display_name == selector]
    if len(matches) > 1:
        err_console.print(
            f"[red]Error:[/red] '{rich_escape(selector)}' matches multiple clusters. "
            "Use its 1-based index from `prime factory status` instead."
        )
        raise typer.Exit(1)
    if matches:
        return matches

    # Index path: ASCII digits only. str.isdigit() is True for Unicode
    # digits (e.g. '²', '①') that int() cannot parse, so a plain isdigit()
    # guard would raise ValueError instead of the clean miss error below.
    if selector.isascii() and selector.isdigit():
        index = int(selector)
        if 1 <= index <= len(clusters):
            return [index - 1]

    err_console.print(f"[red]Error:[/red] No cluster matched '{rich_escape(selector)}'.")
    if clusters:
        names = ", ".join(
            f"[{i + 1}] {rich_escape(c.display_name)}" for i, c in enumerate(clusters)
        )
        err_console.print(f"[dim]Available clusters: {names}[/dim]")
    raise typer.Exit(1)


@app.command(name="status", epilog=FACTORY_STATUS_JSON_HELP)
def factory_status(
    team: Optional[str] = typer.Option(
        None, "--team", "-t", help="Team ID override (defaults to the selected account context)"
    ),
    cluster: Optional[str] = typer.Option(
        None,
        "--cluster",
        help="Show only this cluster, by display name or 1-based index",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Print the API response as JSON (same as --output json)"
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Show GPU allocation across your team's dedicated factory clusters.

    Example:

        prime factory status

        prime factory status --cluster research-b300

        prime factory status --json
    """
    if json_output:
        output = "json"
    validate_output_format(output, console)

    # JSON mode keeps stdout strictly data: every diagnostic goes to stderr.
    err_console = get_console(stderr=True) if output == "json" else console

    team_id = team or Config().team_id
    if not team_id:
        err_console.print(
            "No team selected in the current account context. "
            "`prime factory status` shows your team's dedicated clusters."
        )
        err_console.print(
            "[dim]Run `prime switch` to select a team, or pass --team <team_id>.[/dim]"
        )
        return

    try:
        api_client = APIClient()
        status = FactoryClient(api_client).get_status(team_id)
    except APIError as e:
        # Escape upstream error text: raw brackets (e.g. pydantic
        # "[type=...]" metadata) would crash Rich markup rendering.
        err_console.print(f"[red]Error:[/red] {rich_escape(str(e))}")
        raise typer.Exit(1)

    clusters = status.clusters
    selected: Optional[List[int]] = None
    if cluster is not None:
        selected = _select_cluster_indices(clusters, cluster, err_console)
        clusters = [clusters[i] for i in selected]

    if output == "json":
        payload = status.raw_response
        if selected is not None:
            # Filter the raw response objects, not re-serialized models, so
            # --json stays an exact passthrough of the API payload.
            raw_clusters = status.raw_response.get("clusters", [])
            payload = {**status.raw_response, "clusters": [raw_clusters[i] for i in selected]}
        output_data_as_json(payload, console)
        return

    if not clusters:
        console.print("No factory clusters allocated.")
        return

    states = [_ClusterState(c) for c in clusters]
    multi = len(clusters) > 1

    # CLUSTERS answers "what do I have": one simple inventory line each.
    console.print("[bold]CLUSTERS[/bold]")
    for index, (c, state) in enumerate(zip(clusters, states), start=1):
        freshness = (
            _last_seen_phrase(state.remaining_degraded) if state.remaining_degraded else None
        )
        console.print(_cluster_header(c, index, multi, freshness))

    # WORKLOADS answers "what is running": pool-level holding view per cluster.
    console.print()
    console.print("[bold]WORKLOADS[/bold]")
    _render_workloads_section(clusters, states, multi)
