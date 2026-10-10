"""`prime factory` — Model Factory fleet status and workloads."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import typer
from rich.markup import escape as rich_escape
from rich.table import Table

from ..api.factory import (
    FactoryClient,
    FactoryCluster,
    FactoryNode,
    FactoryNodesCluster,
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
    "                 unassigned_gpus, unknown_gpus, allocations[], sources[]}",
    ".allocations[] = {type, reserved_gpus, in_use_gpus, idle_inside_gpus, unknown_gpus}",
    ".sources[] = {kind, status, observed_at}",
)

FACTORY_NODES_JSON_HELP = json_output_help(
    ". = {schema_version, as_of, clusters[], sources[]}",
    ".clusters[] = {display_name, status, nodes[]}",
    ".nodes[] = {name, state, gpu_type, gpus_total, gpus_used, assigned_to}",
    ".sources[] = {kind, status, observed_at} — one capacity entry per cluster,",
    "                                     in the same order as clusters[]",
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


def _pool_is_all_zero(pool: FactoryPool) -> bool:
    """The backend's designed inactive-group signal: all-zero counts."""
    return (
        pool.reserved_gpus == 0
        and pool.in_use_gpus == 0
        and pool.idle_inside_gpus == 0
        and pool.unknown_gpus == 0
    )


def _pool_is_available(pool: FactoryPool, source_status: dict) -> bool:
    """A workload-group row renders only with complete evidence behind a fresh source.

    Fail closed on missing freshness evidence: a present allocation whose
    source entry is omitted from the envelope is unknown, never silently
    fresh. The one exception is the backend's designed inactive-group
    signal — an all-zero row with no source entry is complete evidence
    of nothing.
    """
    if pool.type not in source_status:
        if not _pool_is_all_zero(pool):
            return False
    elif source_status[pool.type] != "ok":
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
        # Fail closed: a missing capacity source entry is unknown evidence,
        # never silently fresh.
        self.capacity_ok = source_status.get("capacity") == "ok"
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


class _DisplayNamedCluster(Protocol):
    """Structural type: anything with a public display name.

    The status and nodes payloads are different envelope shapes; cluster
    selection and labeling only ever read `display_name`.
    """

    display_name: str


def _cluster_label(cluster: _DisplayNamedCluster, index: int, multi: bool) -> str:
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
    table.add_column("WORKLOAD", style="cyan")
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

        if state.suppressed or not state.capacity_ok:
            if state.suppressed:
                names = (
                    "workload"
                    if not state.renderable and len(state.suppressed) == len(cluster.pools)
                    else ", ".join(rich_escape(p.type) for p in state.suppressed)
                )
            else:
                # Degraded capacity with no allocation rows at all: still a
                # coverage failure, never a valid empty cluster.
                names = "workload"
            line = f"{names} breakdown unavailable"
            if state.involved_sources:
                line += f" — {_last_seen_phrase(state.involved_sources)}"
            console.print(f"[yellow]{line}[/yellow]")

        if state.renderable:
            console.print(_render_pool_table(state.renderable))
            footnote = True

        # Unassigned and unknown GPUs are cluster-level facts, not workload-group rows.
        if state.capacity_ok and cluster.unassigned_gpus is not None:
            console.print(f"unassigned: {cluster.unassigned_gpus} GPUs")
        if state.capacity_ok and cluster.unknown_gpus is not None:
            console.print(f"unknown: {cluster.unknown_gpus} GPUs")

        # The empty-cluster wording requires fresh evidence: with degraded
        # capacity the breakdown-unavailable line above already told the
        # truth about the missing data.
        if (
            state.capacity_ok
            and not state.renderable
            and not state.suppressed
            and not (cluster.unassigned_gpus is not None or cluster.unknown_gpus is not None)
        ):
            console.print("[dim]no workloads reported[/dim]")

    if footnote:
        console.print()
        console.print(f"[dim]{IN_USE_NOTE}[/dim]")
        console.print("[dim]drill down: prime factory nodes[/dim]")


def _cluster_aggregates_known(cluster: FactoryCluster, source_status: dict) -> bool:
    """False when any ACTIVE contributor's numbers are unknowable.

    An allocation whose source entry is stale/errored — or missing while the
    row claims GPUs — can never be summed honestly: the surviving groups'
    totals would present incomplete evidence as a complete number. The
    backend's designed inactive signal (an all-zero row without a source
    entry) is complete evidence of nothing and keeps the sums knowable.
    """
    for pool in cluster.pools:
        if pool.type not in source_status:
            if not _pool_is_all_zero(pool):
                return False
        elif source_status[pool.type] != "ok":
            return False
    return True


def _fresh_groups(cluster: FactoryCluster, source_status: dict) -> List[FactoryPool]:
    """Workload groups whose numbers may render.

    A group whose source entry is stale/errored, or missing for non-zero
    claims, is excluded: numbers from degraded evidence never render. The
    all-zero row without a source entry is the backend's designed inactive
    signal and keeps its honest zeros.
    """
    fresh: List[FactoryPool] = []
    for pool in cluster.pools:
        if pool.type not in source_status:
            if not _pool_is_all_zero(pool):
                continue
        elif source_status[pool.type] != "ok":
            continue
        fresh.append(pool)
    return fresh


def _sum_group_metric(groups: List[FactoryPool], attr: str) -> Optional[int]:
    """Sum one GPU metric over fresh groups.

    Unobserved values contribute nothing — a null must not poison the
    healthy peers' aggregate. No observed values at all renders as None
    (the caller turns it into an em-dash, never a zero).
    """
    total = 0
    observed = False
    for pool in groups:
        value = getattr(pool, attr)
        if value is None:
            continue
        total += value
        observed = True
    return total if observed else None


def _compact_metric_cell(value: Optional[int]) -> str:
    return str(value) if value is not None else "—"


def _compact_nodes_cell(
    nodes_cluster: Optional[FactoryNodesCluster], source: Optional[FactorySource]
) -> str:
    """Healthy/total nodes (plus cordoned count) for the compact status row."""
    if nodes_cluster is None:
        return "—"
    # Fail closed: the envelope-level capacity entry is the freshness
    # evidence for this cluster's node view.
    capacity_ok = source is not None and source.kind == "capacity" and source.status == "ok"
    if not capacity_ok:
        return "—"
    total = len(nodes_cluster.nodes)
    healthy = sum(1 for n in nodes_cluster.nodes if n.state == "ready")
    cell = f"{healthy}/{total}"
    cordoned = sum(1 for n in nodes_cluster.nodes if n.state == "cordoned")
    if cordoned:
        cell += f", {cordoned} cgdn"
    return cell


def _join_data_phrases(data_cell: str, phrase: str) -> str:
    if data_cell == "fresh":
        return phrase
    if phrase in data_cell:
        return data_cell
    return f"{data_cell}, {phrase}"


def _compact_data_cell(cluster: FactoryCluster) -> str:
    """Freshness of the cluster's sources, plain words, internal kinds invisible."""
    source_status = {s.kind: s.status for s in cluster.sources}
    phrases: List[str] = []
    for source in cluster.sources:
        if source.status != "ok":
            if source.observed_at is not None:
                phrases.append(
                    f"{_friendly_data_name(source.kind)} {human_age(source.observed_at)} ago"
                )
            else:
                phrases.append(f"{_friendly_data_name(source.kind)} unavailable")
    mentioned: set = set()
    for pool in cluster.pools:
        # Fail closed: an allocation claiming GPUs without a source entry
        # is unknown evidence — the DATA cell must describe the whole row.
        if pool.type not in source_status and not _pool_is_all_zero(pool):
            name = _friendly_data_name(pool.type)
            if name not in mentioned:
                mentioned.add(name)
                phrases.append(f"{name} unavailable")
    if "capacity" not in source_status:
        # Fail closed: a missing capacity entry is unknown evidence.
        phrases.append("node data unavailable")
    if not phrases:
        return "fresh"
    return ", ".join(phrases)


# The best-effort node view must never stall the status glance.
NODES_FETCH_TIMEOUT_S = 3.0


def _fetch_node_pairs(
    api_client: APIClient, team_id: str
) -> List[Tuple[FactoryNodesCluster, Optional[FactorySource]]]:
    """Best-effort node summaries for the compact status table.

    Returns positionally paired (cluster, capacity source) entries — the
    same identity the envelope sources use. Keying by display name would
    let duplicate-named clusters overwrite each other. The node view is a
    display aid, never a hard dependency: on any API error (including a
    timeout, which the client raises as APIError) the status table renders
    with an em-dash NODES column.
    """
    try:
        nodes = FactoryClient(api_client).get_nodes(team_id, timeout=NODES_FETCH_TIMEOUT_S)
        return [
            (cluster, nodes.sources[i] if i < len(nodes.sources) else None)
            for i, cluster in enumerate(nodes.clusters)
        ]
    except APIError:
        return []
    except ValueError:
        # An HTTP 200 with a non-JSON body escapes the client as a raw
        # decoding error, not an APIError. The node view is best-effort:
        # degrade to an em-dash NODES column instead of crashing the
        # status glance. (The client's error semantics for other commands
        # stay unchanged.)
        return []


def _render_status_table(
    clusters: List[FactoryCluster],
    node_pairs: List[Tuple[FactoryNodesCluster, Optional[FactorySource]]],
    selected: Optional[List[int]] = None,
    nodes_fetch_ok: bool = True,
    ambiguous_names: Optional[set] = None,
) -> None:
    """The default sinfo-style glance: one row per cluster, no prose.

    ``clusters`` may be a --cluster-selected subset; ``selected`` holds the
    original payload indices, so node summaries pair positionally with the
    full payload. ``nodes_fetch_ok`` marks whether the best-effort node view
    arrived at all; ``ambiguous_names`` are display names that appear more
    than once — equal-name replacement between the two requests is
    undetectable, so those rows render an em-dash NODES cell.
    """
    any_degraded = False
    table = Table(show_header=True, header_style="bold", show_lines=False)
    table.add_column("CLUSTER", style="cyan")
    table.add_column("GPU")
    table.add_column("STATUS")
    table.add_column("HELD", justify="right")
    table.add_column("IN USE", justify="right")
    # Idle inside the reservation — never readable as free-to-start capacity.
    table.add_column("IDLE INSIDE", justify="right")
    table.add_column("NODES", justify="right")
    table.add_column("DATA")

    for position, cluster in enumerate(clusters):
        source_status = {s.kind: s.status for s in cluster.sources}
        data_cell = _compact_data_cell(cluster)
        if not nodes_fetch_ok:
            # DATA describes the whole row: a missing node view is not
            # "fresh".
            data_cell = _join_data_phrases(data_cell, "node view unavailable")
        if data_cell != "fresh":
            any_degraded = True
        capacity_ok = source_status.get("capacity") == "ok"
        aggregates_known = capacity_ok and _cluster_aggregates_known(cluster, source_status)
        groups = _fresh_groups(cluster, source_status) if aggregates_known else []
        original_index = selected[position] if selected is not None else position
        nodes_cluster, nodes_source = (
            node_pairs[original_index] if original_index < len(node_pairs) else (None, None)
        )
        if nodes_cluster is not None and nodes_cluster.display_name != cluster.display_name:
            # The status and nodes responses are separate requests: if the
            # fleet moved between them, positional identity no longer
            # holds. Never display one cluster's node counts on another
            # cluster's row — degrade the NODES cell instead.
            nodes_cluster, nodes_source = None, None
        if (
            nodes_cluster is not None
            and ambiguous_names
            and cluster.display_name in ambiguous_names
        ):
            # Duplicate display names make the equal-name identity check
            # blind to replacement; never risk cross-wired counts.
            nodes_cluster, nodes_source = None, None
        table.add_row(
            rich_escape(cluster.display_name),
            rich_escape(cluster.gpu_type) if cluster.gpu_type else "—",
            _styled_status(cluster.status),
            _compact_metric_cell(_sum_group_metric(groups, "reserved_gpus")),
            _compact_metric_cell(_sum_group_metric(groups, "in_use_gpus")),
            _compact_metric_cell(_sum_group_metric(groups, "idle_inside_gpus")),
            _compact_nodes_cell(nodes_cluster, nodes_source),
            data_cell,
        )
    console.print(table)
    if any_degraded:
        # At most one dim line under the table, nothing else.
        console.print("[dim]degraded sources — details: prime factory status --verbose[/dim]")


def _select_cluster_indices(
    clusters: Sequence[_DisplayNamedCluster], selector: str, err_console: Any
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
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Show the detailed per-cluster allocation sections instead of the compact table",
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

        prime factory status --verbose

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

    if not verbose:
        # The default is the compact sinfo-style glance: one row per
        # cluster, at most one dim line under the table, no prose.
        all_clusters = status.clusters
        name_counts: Dict[str, int] = {}
        for status_cluster in all_clusters:
            name = status_cluster.display_name
            name_counts[name] = name_counts.get(name, 0) + 1
        ambiguous = {name for name, count in name_counts.items() if count > 1}
        node_pairs = _fetch_node_pairs(api_client, team_id)
        _render_status_table(
            clusters,
            node_pairs,
            selected,
            nodes_fetch_ok=bool(node_pairs),
            ambiguous_names=ambiguous,
        )
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

    # WORKLOADS answers "what is running": workload-group holding view per cluster.
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


def _view_degraded_sources(
    workloads: FactoryWorkloads,
    view_rows: List[FactoryWorkload],
    suppressed: List[FactoryWorkload],
    requested_type: Optional[str] = None,
) -> List[FactorySource]:
    """Warning sources for the workloads view — warnings, never row erasure.

    Rows render on their own per-row evidence, so an aggregate degraded
    source can never erase healthy peers. Envelope-degraded kinds warn when
    they have rows in the narrowed view — even when every returned row of
    the kind is fresh, because the degraded portion produced no rows and
    must not vanish silently — or when they produced no rows at all in the
    payload (a failed read must not masquerade as an empty fleet). Kinds
    narrowed out of the view entirely do not warn.
    """
    warnings: List[FactorySource] = []
    seen: set = set()
    view_kinds = {row.source.kind for row in view_rows}
    payload_kinds = {row.source.kind for row in workloads.workloads}
    for source in workloads.sources:
        if source.status != "ok" and source.kind not in seen:
            zero_rows = source.kind not in payload_kinds
            if (
                requested_type is not None
                and source.kind != requested_type
                and zero_rows
                and source.kind not in view_kinds
            ):
                # Kinds narrowed out server-side by --type: their absence
                # is expected filtering, not a failed read.
                continue
            if source.kind in view_kinds or zero_rows:
                seen.add(source.kind)
                warnings.append(source)
    newest_by_kind: Dict[str, FactorySource] = {}
    for row in suppressed:
        kind = row.source.kind
        current = newest_by_kind.get(kind)
        if current is None or _is_newer_observed_at(row.source, current):
            # Several degraded rows may carry fallback evidence for the
            # same kind; the newest observation is the honest last-seen.
            newest_by_kind[kind] = row.source
    for source in newest_by_kind.values():
        if source.kind not in seen:
            seen.add(source.kind)
            warnings.append(source)
    return warnings


def _is_newer_observed_at(candidate: FactorySource, current: FactorySource) -> bool:
    """True when candidate's observation is more recent than current's.

    A missing observed_at on the incumbent never beats an observed
    candidate; two missing observed_ats keep the first (stable order).
    """
    if candidate.observed_at is None:
        return False
    if current.observed_at is None:
        return True
    return candidate.observed_at > current.observed_at


def _jobs_unavailable_line(
    kind: str, source: Optional[FactorySource], mixed_coverage: bool = False
) -> str:
    """Plain-language degraded-aggregate warning.

    Fresh rows of the same kind render alongside ("mixed coverage"): say
    the data is partial without claiming all jobs are gone, and omit the
    age — the aggregate's observed_at then describes the healthy read and
    cannot be attributed to the degraded evidence honestly. Without fresh
    peers, the wording keeps the degraded evidence's last-seen age.
    """
    if mixed_coverage:
        return f"some {rich_escape(kind)} job data is unavailable — results may be incomplete"
    jobs = FRIENDLY_JOB_NAMES.get(kind, f"{rich_escape(kind)} jobs")
    return f"{jobs} unavailable — {_last_seen_phrase([source] if source else [])}"


def _workload_owner_cell(workload: FactoryWorkload) -> str:
    owner = workload.owner
    if owner.display_name is None:
        return "[dim]-[/dim]"
    return rich_escape(owner.display_name)


def _workload_age_cell(workload: FactoryWorkload) -> str:
    """Run age for started work, wait age for queued/never-started work."""
    if workload.state == "queued":
        # A requeued row keeps its historical started_at; the documented
        # wait age must come from created_at, not from the old run.
        reference = workload.created_at
    else:
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
    table.add_column("CLUSTER")
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
            rich_escape(row.cluster_display_name) if row.cluster_display_name else "[dim]-[/dim]",
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
        workloads = FactoryClient(api_client).get_workloads(
            team_id, workload_type=type, state=state
        )
    except APIError as e:
        # Escape upstream error text: raw brackets (e.g. pydantic
        # "[type=...]" metadata) would crash Rich markup rendering.
        err_console.print(f"[red]Error:[/red] {rich_escape(str(e))}")
        raise typer.Exit(1)

    rows = workloads.workloads

    # Identity resolution: the workloads envelope carries no cluster list,
    # only row identities, so a --cluster selector can miss because a
    # degraded source omitted a cluster's rows entirely. Surface the
    # degraded-source warnings instead of exiting with a clean miss.
    if cluster is not None and not any(row.cluster_display_name == cluster for row in rows):
        fresh_kinds = {row.source.kind for row in rows if row.source.status == "ok"}
        for source in workloads.sources:
            if source.status != "ok":
                warning = _jobs_unavailable_line(source.kind, source, source.kind in fresh_kinds)
                err_console.print(f"[yellow]{warning}[/yellow]")
        err_console.print(f"[red]Error:[/red] No cluster matched '{rich_escape(cluster)}'.")
        names = _distinct_workload_clusters(rows)
        if names:
            listed = ", ".join(rich_escape(n) for n in names)
            err_console.print(f"[dim]Available clusters: {listed}[/dim]")
        raise typer.Exit(1)
    rows = _filter_workload_rows(rows, user, cluster)

    if output == "json":
        payload = workloads.raw_response
        if user is not None or cluster is not None:
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

    # Per-row freshness governs rendering: a row renders when its OWN
    # evidence is fresh, never suppressed by an aggregate worst-source or a
    # missing sibling group. Degraded aggregates are warnings only.
    available = [row for row in rows if row.source.status == "ok"]
    suppressed = [row for row in rows if row.source.status != "ok"]
    degraded = _view_degraded_sources(workloads, rows, suppressed, requested_type=type)

    # Degraded sources say so before anything else: a failed read must never
    # masquerade as an empty fleet or silently vanish. Fresh rows of the
    # same kind mark mixed coverage — partial data, not "all unavailable".
    fresh_kinds = {row.source.kind for row in rows if row.source.status == "ok"}
    for source in degraded:
        warning = _jobs_unavailable_line(source.kind, source, source.kind in fresh_kinds)
        console.print(f"[yellow]{warning}[/yellow]")

    if available:
        console.print(_render_workloads_table(available))
        console.print()
        console.print(f"[dim]{IN_USE_NOTE}[/dim]")
        return

    if not degraded:
        # Distinguish an honestly empty fleet from filters that matched
        # nothing: --type/--state are server-side, so a filtered result of
        # zero rows is not evidence that the team has no workloads.
        if type is not None or state is not None or user is not None or cluster is not None:
            console.print("No factory workloads match the given filters.")
        else:
            # Genuinely nothing running or queued, with fresh evidence.
            console.print("No factory workloads found.")


# Coarse public node states from the frozen nodes contract; the labels are
# the only node vocabulary shown to users.
NODE_STATES = ("ready", "cordoned", "offline", "unknown")
# The backend only ever populates slurm claims in v1.5; per-node
# training/inference placement is NOT observed and must not be filterable
# until it exists.
NODE_ASSIGNEES = ("slurm",)


class _NodesState:
    """Per-cluster rendering facts for the nodes view.

    Freshness comes from the envelope-level `sources` list: the frozen
    nodes contract pairs one capacity entry with each cluster, in the same
    order as `clusters` — the cluster objects carry no sources of their own.
    """

    def __init__(self, cluster: FactoryNodesCluster, source: Optional[FactorySource]) -> None:
        # Fail closed: a missing capacity source entry is unknown evidence,
        # never silently fresh.
        sources = [source] if source is not None else []
        source_status = {s.kind: s.status for s in sources}
        self.capacity_ok = source_status.get("capacity") == "ok"
        degraded = [s for s in sources if s.status != "ok"]
        self.capacity_sources = [s for s in degraded if s.kind == "capacity"]
        self.remaining_degraded = [s for s in degraded if s.kind != "capacity"]


def _node_state_cell(node: FactoryNode) -> str:
    state_styles = {
        "ready": "green",
        "cordoned": "yellow",
        "offline": "red",
        "unknown": "dim",
    }
    state = node.state or "unknown"
    if state in state_styles:
        return f"[{state_styles[state]}]{rich_escape(state)}[/{state_styles[state]}]"
    return rich_escape(state)


def _node_gpu_cell(node: FactoryNode) -> str:
    """Used/total GPUs; unobserved components stay `-`, never 0."""

    def _count(value: Optional[int]) -> str:
        return str(value) if value is not None else "[dim]-[/dim]"

    return f"{_count(node.gpus_used)}/{_count(node.gpus_total)}"


def _nodes_cluster_header(
    cluster: FactoryNodesCluster, index: int, multi: bool, freshness: Optional[str]
) -> str:
    parts: List[str] = [_cluster_label(cluster, index, multi)]
    parts.append(_styled_status(cluster.status))
    if freshness:
        parts.append(freshness)
    return " · ".join(parts)


def _render_nodes_table(nodes: List[FactoryNode]) -> Table:
    table = Table(show_header=True, header_style="bold", show_lines=False)
    table.add_column("NODE", style="cyan")
    table.add_column("STATE", style="white")
    table.add_column("HELD", justify="right")
    table.add_column("ASSIGNED TO")

    for node in nodes:
        if node.assigned_to:
            assigned = rich_escape(node.assigned_to)
        else:
            # Null means placement is NOT observed in v1.5 — the node is
            # not proven unassigned, so never label it that way.
            assigned = "[dim]unknown[/dim]"
        table.add_row(
            rich_escape(node.name),
            _node_state_cell(node),
            _node_gpu_cell(node),
            assigned,
        )
    return table


def _filter_nodes(
    nodes: List[FactoryNode],
    state: Optional[str],
    assigned_to: Optional[str],
) -> List[FactoryNode]:
    """Client-side narrowing of node rows by state and assignee."""
    if state is not None:
        nodes = [n for n in nodes if (n.state or "unknown") == state]
    if assigned_to is not None:
        nodes = [n for n in nodes if n.assigned_to == assigned_to]
    return nodes


def _filter_raw_nodes(
    raw_nodes: List[Dict[str, Any]],
    state: Optional[str],
    assigned_to: Optional[str],
) -> List[Dict[str, Any]]:
    """Raw-payload mirror of `_filter_nodes` for exact `--json` filtering."""
    if state is not None:
        raw_nodes = [n for n in raw_nodes if (n.get("state") or "unknown") == state]
    if assigned_to is not None:
        raw_nodes = [n for n in raw_nodes if n.get("assigned_to") == assigned_to]
    return raw_nodes


@app.command(name="nodes", epilog=FACTORY_NODES_JSON_HELP)
def factory_nodes(
    team: Optional[str] = typer.Option(
        None, "--team", "-t", help="Team ID override (defaults to the selected account context)"
    ),
    cluster: Optional[str] = typer.Option(
        None, "--cluster", help="Show only this cluster, by display name or 1-based index"
    ),
    state: Optional[str] = typer.Option(
        None, "--state", help="Filter nodes by state: ready, cordoned, offline, or unknown"
    ),
    assigned_to: Optional[str] = typer.Option(
        None,
        "--assigned-to",
        help=(
            "Filter nodes by assignee: slurm (per-node training/inference "
            "placement is not observed yet)"
        ),
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Print the API response as JSON (same as --output json)"
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Show the nodes of your team's dedicated factory clusters (sinfo-like).

    Example:

        prime factory nodes

        prime factory nodes --cluster research-b300 --state cordoned

        prime factory nodes --assigned-to slurm --json
    """
    if json_output:
        output = "json"
    validate_output_format(output, console)

    # JSON mode keeps stdout strictly data: every diagnostic goes to stderr.
    err_console = get_console(stderr=True) if output == "json" else console

    if state is not None and state not in NODE_STATES:
        err_console.print(
            f"[red]Error:[/red] Invalid --state '{rich_escape(state)}'. "
            f"Choose one of: {', '.join(NODE_STATES)}."
        )
        raise typer.Exit(1)
    if assigned_to is not None and assigned_to not in NODE_ASSIGNEES:
        err_console.print(
            f"[red]Error:[/red] Invalid --assigned-to '{rich_escape(assigned_to)}'. "
            "Only slurm is observable; per-node training/inference "
            "placement is not observed yet."
        )
        raise typer.Exit(1)

    team_id = team or Config().team_id
    if not team_id:
        err_console.print(
            "No team selected in the current account context. "
            "`prime factory nodes` shows your team's factory nodes."
        )
        err_console.print(
            "[dim]Run `prime switch` to select a team, or pass --team <team_id>.[/dim]"
        )
        return

    try:
        api_client = APIClient()
        nodes_payload = FactoryClient(api_client).get_nodes(team_id)
    except APIError as e:
        # Escape upstream error text: raw brackets (e.g. pydantic
        # "[type=...]" metadata) would crash Rich markup rendering.
        err_console.print(f"[red]Error:[/red] {rich_escape(str(e))}")
        raise typer.Exit(1)

    clusters = nodes_payload.clusters
    selected: Optional[List[int]] = None
    if cluster is not None:
        selected = _select_cluster_indices(clusters, cluster, err_console)
        clusters = [clusters[i] for i in selected]

    if output == "json":
        payload = nodes_payload.raw_response
        raw_clusters = nodes_payload.raw_response.get("clusters", [])
        raw_sources = nodes_payload.raw_response.get("sources", [])
        if selected is not None:
            # Index-based selection matches table mode: with duplicate
            # display names, name-matching would return both clusters. The
            # envelope-level sources pair with clusters by order, so they
            # follow the same selection.
            raw_clusters = [raw_clusters[i] for i in selected]
            raw_sources = [raw_sources[i] for i in selected if i < len(raw_sources)]
        if state is not None or assigned_to is not None:
            raw_clusters = [
                {
                    **raw_cluster,
                    "nodes": _filter_raw_nodes(raw_cluster.get("nodes", []), state, assigned_to),
                }
                for raw_cluster in raw_clusters
            ]
        if selected is not None or state is not None or assigned_to is not None:
            # Filter the raw response objects, not re-serialized models, so
            # --json stays an exact passthrough of the API payload.
            payload = {
                **nodes_payload.raw_response,
                "clusters": raw_clusters,
                "sources": raw_sources,
            }
        output_data_as_json(payload, console)
        return

    if not clusters:
        console.print("No factory clusters allocated.")
        return

    # Envelope-level sources pair with clusters by order; a missing entry
    # fails closed inside _NodesState.
    paired_sources = [
        nodes_payload.sources[i] if i < len(nodes_payload.sources) else None
        for i in (selected if selected is not None else range(len(clusters)))
    ]
    states = [_NodesState(cluster, source) for cluster, source in zip(clusters, paired_sources)]
    multi = len(clusters) > 1

    console.print("[bold]CLUSTERS[/bold]")
    for index, (c, state_obj) in enumerate(zip(clusters, states), start=1):
        if index > 1:
            console.print()
        freshness = (
            _last_seen_phrase(state_obj.remaining_degraded)
            if state_obj.remaining_degraded
            else None
        )
        console.print(_nodes_cluster_header(c, index, multi, freshness))

        if not state_obj.capacity_ok:
            # The node inventory itself is degraded: say so instead of a
            # quiet empty or stale table.
            line = "node breakdown unavailable"
            if state_obj.capacity_sources:
                line += f" — {_last_seen_phrase(state_obj.capacity_sources)}"
            console.print(f"[yellow]{line}[/yellow]")
            continue

        rows = _filter_nodes(c.nodes, state, assigned_to)
        if rows:
            console.print(_render_nodes_table(rows))
        elif state is not None or assigned_to is not None:
            # Distinguish an honestly node-less cluster from filters that
            # matched nothing.
            console.print("[dim]no nodes match the given filters[/dim]")
        else:
            console.print("[dim]no nodes reported[/dim]")
