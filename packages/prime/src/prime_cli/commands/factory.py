"""`prime factory` — Model Factory fleet status and workloads."""

from datetime import datetime
from typing import Any, List, Optional

import typer
from rich.markup import escape as rich_escape
from rich.table import Table

from ..api.factory import (
    FactoryClient,
    FactoryCluster,
    FactoryPool,
    FactorySource,
    FactoryWorkload,
    FactoryWorkloads,
)
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
    help="View your team's dedicated Model Factory clusters, GPU allocations, and workloads.",
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


FACTORY_WORKLOADS_JSON_HELP = json_output_help(
    ". = {schema_version, as_of, workloads[], sources[]}",
    ".workloads[] = {id, type, cluster_display_name, name, state, native_state,",
    "                 owner{kind, display_name}, requested_gpus, allocated_gpus,",
    "                 created_at, started_at, source{kind, status, observed_at}}",
    ".sources[] = {kind, status, observed_at}",
)

# Server-side filter values, mirrored from the public contract.
WORKLOAD_TYPE_FILTERS = ("training", "inference", "slurm")
WORKLOAD_STATE_FILTERS = ("running", "queued")

# Plain-language row-group names for degraded sources; the internal source
# enum values never appear in their own right.
FRIENDLY_JOB_NAMES = {
    "training": "training jobs",
    "inference": "inference jobs",
    "slurm": "slurm jobs",
}


def _workload_kind(workload: FactoryWorkload) -> str:
    """Source kind that produced the row; falls back to the workload type."""
    return workload.source.kind or workload.type


def _envelope_source_status(workloads: FactoryWorkloads) -> dict:
    """kind -> status map from the envelope-level source entries."""
    return {s.kind: s.status for s in workloads.sources}


def _row_source_status(workload: FactoryWorkload, envelope_status: dict) -> Optional[str]:
    """Effective status of the source behind one row.

    The envelope entries are authoritative; a row whose kind has no envelope
    entry falls back to its own coarse source status.
    """
    kind = _workload_kind(workload)
    if kind in envelope_status:
        return envelope_status[kind]
    return workload.source.status


def _degraded_kinds(
    workloads: FactoryWorkloads, rows: List[FactoryWorkload]
) -> List[FactorySource]:
    """Deduplicated source entries for every degraded kind, stable by kind.

    A degraded kind with zero rows still produces an entry (failure must not
    masquerade as an empty fleet), taking the envelope source or the newest
    row-level source as its evidence.
    """
    envelope_by_kind = {s.kind: s for s in workloads.sources}
    degraded: List[FactorySource] = []
    seen: set = set()
    for source in workloads.sources:
        if source.status != "ok" and source.kind not in seen:
            seen.add(source.kind)
            degraded.append(source)
    for row in rows:
        kind = _workload_kind(row)
        if kind not in envelope_by_kind and row.source.status != "ok" and kind not in seen:
            seen.add(kind)
            degraded.append(row.source)
    return degraded


def _jobs_unavailable_line(kind: str, source: Optional[FactorySource]) -> str:
    """`<jobs> unavailable — <data> last seen <age> ago`, plain words only."""
    jobs = FRIENDLY_JOB_NAMES.get(kind, f"{rich_escape(kind)} jobs")
    return f"{jobs} unavailable — {_last_seen_phrase([source] if source else [])}"


def _workload_owner_cell(workload: FactoryWorkload) -> str:
    owner = workload.owner
    if owner.display_name is None:
        return "[dim]-[/dim]"
    return rich_escape(owner.display_name)


def _workload_age_cell(workload: FactoryWorkload) -> str:
    """Run age for started work, wait age for queued/never-started work."""
    reference = workload.started_at if workload.started_at is not None else workload.created_at
    if reference is None:
        return "[dim]-[/dim]"
    return human_age(reference)


def _workload_gpu_cell(workload: FactoryWorkload) -> str:
    """Allocated/requested GPUs; unobserved components stay `-`, never 0."""

    def _count(value: Optional[int]) -> str:
        return str(value) if value is not None else "[dim]-[/dim]"

    return f"{_count(workload.allocated_gpus)}/{_count(workload.requested_gpus)}"


def _workload_state_cell(workload: FactoryWorkload) -> str:
    state_styles = {
        "running": "green",
        "queued": "yellow",
        "stopping": "yellow",
        "failed": "red",
        "completed": "dim",
    }
    state = workload.state
    if state in state_styles:
        cell = f"[{state_styles[state]}]{rich_escape(state)}[/{state_styles[state]}]"
    else:
        cell = rich_escape(state or "unknown")
    if workload.native_state and workload.native_state.lower() != state.lower():
        cell += f" [dim]({rich_escape(workload.native_state)})[/dim]"
    return cell


def _render_workloads_table(rows: List[FactoryWorkload]) -> Table:
    table = Table(show_header=True, header_style="bold", show_lines=False)
    table.add_column("ID", style="cyan")
    table.add_column("TYPE", style="white")
    table.add_column("NAME")
    table.add_column("OWNER")
    table.add_column("STATE")
    table.add_column("GPU A/R", justify="right")
    table.add_column("AGE", justify="right")
    # The REASON column exists only when the backend supplied coarse
    # blocking/waiting labels; it is never invented client-side.
    show_reason = any(row.reason for row in rows)
    if show_reason:
        table.add_column("REASON", style="yellow")

    for row in rows:
        cells = [
            rich_escape(row.id),
            rich_escape(row.type),
            rich_escape(row.name) if row.name else "[dim]-[/dim]",
            _workload_owner_cell(row),
            _workload_state_cell(row),
            _workload_gpu_cell(row),
            _workload_age_cell(row),
        ]
        if show_reason:
            cells.append(rich_escape(row.reason) if row.reason else "[dim]-[/dim]")
        table.add_row(*cells)
    return table


def _filter_workload_rows(
    rows: List[FactoryWorkload],
    user: Optional[str],
    cluster: Optional[str],
) -> List[FactoryWorkload]:
    """Client-side narrowing by owner display name and cluster display name."""
    if user is not None:
        rows = [r for r in rows if r.owner.display_name == user]
    if cluster is not None:
        rows = [r for r in rows if r.cluster_display_name == cluster]
    return rows


def _select_workload_clusters(
    rows: List[FactoryWorkload], cluster: str, err_console: Any
) -> List[str]:
    """Validate a --cluster selector against the workloads' cluster names."""
    matches = [name for name in _distinct_workload_clusters(rows) if name == cluster]
    if not matches:
        err_console.print(f"[red]Error:[/red] No cluster matched '{rich_escape(cluster)}'.")
        names = _distinct_workload_clusters(rows)
        if names:
            listed = ", ".join(rich_escape(n) for n in names)
            err_console.print(f"[dim]Available clusters: {listed}[/dim]")
        raise typer.Exit(1)
    return matches


def _distinct_workload_clusters(rows: List[FactoryWorkload]) -> List[str]:
    names: List[str] = []
    for row in rows:
        if row.cluster_display_name and row.cluster_display_name not in names:
            names.append(row.cluster_display_name)
    return names


@app.command(name="workloads", epilog=FACTORY_WORKLOADS_JSON_HELP)
def factory_workloads(
    team: Optional[str] = typer.Option(
        None, "--team", "-t", help="Team ID override (defaults to the selected account context)"
    ),
    type: Optional[str] = typer.Option(
        None, "--type", help="Filter by workload type: training, inference, or slurm"
    ),
    state: Optional[str] = typer.Option(None, "--state", help="Filter by state: running or queued"),
    user: Optional[str] = typer.Option(None, "--user", help="Filter by owner display name"),
    cluster: Optional[str] = typer.Option(None, "--cluster", help="Filter by cluster display name"),
    json_output: bool = typer.Option(
        False, "--json", help="Print the API response as JSON (same as --output json)"
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Show your team's factory workloads (training, inference, Slurm jobs).

    Example:

        prime factory workloads

        prime factory workloads --type slurm --state queued

        prime factory workloads --user carol --json
    """
    if json_output:
        output = "json"
    validate_output_format(output, console)

    # JSON mode keeps stdout strictly data: every diagnostic goes to stderr.
    err_console = get_console(stderr=True) if output == "json" else console

    if type is not None and type not in WORKLOAD_TYPE_FILTERS:
        err_console.print(
            f"[red]Error:[/red] Invalid --type '{rich_escape(type)}'. "
            f"Choose one of: {', '.join(WORKLOAD_TYPE_FILTERS)}."
        )
        raise typer.Exit(1)
    if state is not None and state not in WORKLOAD_STATE_FILTERS:
        err_console.print(
            f"[red]Error:[/red] Invalid --state '{rich_escape(state)}'. "
            f"Choose one of: {', '.join(WORKLOAD_STATE_FILTERS)}."
        )
        raise typer.Exit(1)

    team_id = team or Config().team_id
    if not team_id:
        err_console.print(
            "No team selected in the current account context. "
            "`prime factory workloads` shows your team's factory workloads."
        )
        err_console.print(
            "[dim]Run `prime switch` to select a team, or pass --team <team_id>.[/dim]"
        )
        return

    try:
        api_client = APIClient()
        workloads = FactoryClient(api_client).get_workloads(team_id, type=type, state=state)
    except APIError as e:
        # Escape upstream error text: raw brackets (e.g. pydantic
        # "[type=...]" metadata) would crash Rich markup rendering.
        err_console.print(f"[red]Error:[/red] {rich_escape(str(e))}")
        raise typer.Exit(1)

    rows = workloads.workloads
    selected_cluster: Optional[List[str]] = None
    if cluster is not None:
        if not rows:
            err_console.print(f"[red]Error:[/red] No cluster matched '{rich_escape(cluster)}'.")
            raise typer.Exit(1)
        selected_cluster = _select_workload_clusters(rows, cluster, err_console)
    rows = _filter_workload_rows(rows, user, cluster)

    if output == "json":
        payload = workloads.raw_response
        if user is not None or selected_cluster is not None:
            keep_ids = {row.id for row in rows}
            raw_rows = workloads.raw_response.get("workloads", [])
            # Filter the raw response objects, not re-serialized models, so
            # --json stays an exact passthrough of the API payload.
            payload = {
                **workloads.raw_response,
                "workloads": [w for w in raw_rows if w.get("id") in keep_ids],
            }
        output_data_as_json(payload, console)
        return

    envelope_status = _envelope_source_status(workloads)
    available = [row for row in rows if _row_source_status(row, envelope_status) == "ok"]
    degraded = _degraded_kinds(workloads, rows)

    # Degraded sources say so before anything else: a failed read must never
    # masquerade as an empty fleet or silently vanish.
    for source in degraded:
        console.print(f"[yellow]{_jobs_unavailable_line(source.kind, source)}[/yellow]")

    if available:
        console.print(_render_workloads_table(available))
        console.print()
        console.print(f"[dim]{IN_USE_NOTE}[/dim]")
        return

    if not degraded:
        # Genuinely nothing running or queued, with fresh evidence.
        console.print("No factory workloads found.")
