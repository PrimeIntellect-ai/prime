"""`prime factory` — Model Factory fleet status and workloads."""

import math
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, NamedTuple, Optional, Protocol, Sequence, Tuple

import typer
from rich.markup import escape as rich_escape
from rich.table import Table
from rich.text import Text

from ..api.factory import (
    FactoryClient,
    FactoryCluster,
    FactoryNode,
    FactoryNodes,
    FactoryNodesCluster,
    FactoryPool,
    FactorySource,
    FactoryStatus,
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
    "(legacy servers emit .clusters[].pools[] with the same fields)",
    "--team: use a different team than the selected account context",
)

FACTORY_NODES_JSON_HELP = json_output_help(
    ". = {schema_version, as_of, clusters[], sources[]}",
    ".clusters[] = {display_name, status, nodes[]}",
    ".nodes[] = {name, state, gpu_type, gpus_total, gpus_used, assigned_to}",
    ".sources[] = {kind, status, observed_at} — one capacity entry per cluster,",
    "                                     in the same order as clusters[]",
    "--team: use a different team than the selected account context",
)

IN_USE_NOTE = "in-use = GPUs held by running jobs (not GPU-activity measurements)"
SLURM_ESTIMATE_NOTE = (
    "slurm in use = estimated from running jobs (GPUs/8); CPU-only jobs not counted"
)
SCHEDULER_LEGEND = (
    "Slurm cluster: — = direct platform placement; unknown = deployment not identified."
)


def _truncated_note(limit: int) -> str:
    """The dim truncation note, derived from the effective page size.

    Raising --limit is only advice while there is headroom to raise it.
    """
    note = f"Showing first {limit} workloads — narrow the server query with --type/--state/--since"
    if limit < WORKLOAD_LIMIT_MAX:
        note += ", or raise --limit"
    return note


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


def _styled_status(status: Optional[str]) -> str:
    if status == "online":
        return "[green]online[/green]"
    if status == "offline":
        return "[red]offline[/red]"
    return rich_escape(status or "unknown")


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


class _Usage(NamedTuple):
    """Node counts for one table row: in_use + free + down == nodes."""

    nodes: int
    in_use: int
    free: int
    down: int


def _pool_usage(
    nodes: List[FactoryNode],
    slurm_jobs: List[FactoryWorkload],
    allocations: List[FactoryPool],
) -> List[Tuple[str, _Usage]]:
    """Split one cluster's nodes into pools; every node lands in exactly one.

    Slurm nodes report no per-node load, so busy Slurm nodes are estimated
    from the running jobs' GPUs, capped at the ready Slurm nodes. Ready
    non-Slurm nodes holding GPUs are attributed to training and inference
    from the allocation summary; whatever is left over is "other".
    """
    gpus_per_node = max((n.gpus_total or 0 for n in nodes), default=0)

    slurm = [n for n in nodes if n.assigned_to == "slurm"]
    slurm_down = sum(1 for n in slurm if n.state != "ready")
    slurm_ready = len(slurm) - slurm_down
    job_nodes = (
        sum(math.ceil((job.allocated_gpus or 0) / gpus_per_node) for job in slurm_jobs)
        if gpus_per_node
        else 0
    )
    slurm_busy = min(job_nodes, slurm_ready)
    names = sorted({job.scheduler_display_name for job in slurm_jobs if job.scheduler_display_name})
    slurm_label = f"slurm ({', '.join(names)})" if names else "slurm"
    pools = [(slurm_label, _Usage(len(slurm), slurm_busy, slurm_ready - slurm_busy, slurm_down))]

    direct = [n for n in nodes if n.assigned_to != "slurm"]
    ready = [n for n in direct if n.state == "ready"]
    busy = sum(1 for n in ready if (n.gpus_used or 0) > 0)
    unattributed = busy
    for kind in ("training", "inference"):
        gpus = sum(a.in_use_gpus or 0 for a in allocations if a.type == kind)
        count = min(math.ceil(gpus / gpus_per_node) if gpus_per_node else 0, unattributed)
        pools.append((kind, _Usage(count, count, 0, 0)))
        unattributed -= count
    pools.append(("other", _Usage(unattributed, unattributed, 0, 0)))
    free = len(ready) - busy
    down = len(direct) - len(ready)
    pools.append(("unassigned", _Usage(free + down, 0, free, down)))
    return pools


def _short_gpu_type(gpu_type: Optional[str]) -> str:
    """Compact GPU model for the glance: 'B300_262GB' -> 'B300'."""
    if not gpu_type:
        return "—"
    return rich_escape(gpu_type.split("_")[0])


def _paired_nodes(
    cluster: FactoryCluster, index: int, nodes: FactoryNodes, ambiguous_names: set
) -> Optional[Tuple[FactoryNodesCluster, FactorySource]]:
    """The node view of a status cluster, paired by payload position.

    The status and nodes payloads are separate requests: a position whose
    name differs, or a name that appears twice in either payload, cannot
    be paired safely — never show one cluster's nodes on another's row.
    """
    if index >= len(nodes.clusters) or cluster.display_name in ambiguous_names:
        return None
    nodes_cluster = nodes.clusters[index]
    if nodes_cluster.display_name != cluster.display_name:
        return None
    return nodes_cluster, nodes.sources[index]


def _cluster_warning(
    cluster: FactoryCluster,
    paired: Optional[Tuple[FactoryNodesCluster, FactorySource]],
    slurm_jobs: FactoryWorkloads,
) -> Optional[str]:
    """One short line when the cluster is not online or its data is degraded."""
    problems: List[str] = []
    if cluster.status != "online":
        problems.append(_styled_status(cluster.status))
    sources = list(cluster.sources)
    if paired is None:
        problems.append("node data unavailable")
    else:
        nodes_cluster, node_source = paired
        sources.append(node_source)
        if any(n.assigned_to == "slurm" for n in nodes_cluster.nodes):
            sources.extend(slurm_jobs.sources)
    degraded = [s for s in sources if s.status != "ok"]
    if degraded:
        problems.append(_last_seen_phrase(degraded))
    return " · ".join(problems) if problems else None


def _duplicated_names(names: List[str]) -> set:
    counts: Dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    return {name for name, count in counts.items() if count > 1}


def _count_cell(count: int) -> Text:
    # Zeros are dimmed so the non-zero FREE and DOWN counts stand out.
    return Text(str(count), style="dim default" if count == 0 else "")


def _render_status_table(
    status: FactoryStatus,
    nodes: FactoryNodes,
    slurm_jobs: FactoryWorkloads,
    selected: Optional[List[int]],
    verbose: bool,
) -> None:
    """One row per cluster, counted in nodes (sinfo-style); --verbose adds pool sub-rows.

    ``selected`` holds the original payload indices of a --cluster
    selection, so rows keep the 1-based index users select by and node
    views pair positionally with the full payload.
    """
    table = Table(show_header=True, header_style="bold", show_lines=False)
    table.add_column("CLUSTER", style="cyan")
    table.add_column("GPU")
    table.add_column("NODES", justify="right")
    table.add_column("IN USE", justify="right")
    table.add_column("FREE", justify="right", style="green")
    table.add_column("DOWN", justify="right", style="red")

    # Ambiguity can arise in EITHER snapshot: the two requests are
    # independent.
    ambiguous = _duplicated_names([c.display_name for c in status.clusters]) | _duplicated_names(
        [c.display_name for c in nodes.clusters]
    )
    multi = len(status.clusters) > 1
    indices = selected if selected is not None else list(range(len(status.clusters)))
    warnings: List[str] = []
    for position, index in enumerate(indices):
        if verbose and position:
            table.add_section()
        cluster = status.clusters[index]
        name_cell = rich_escape(cluster.display_name)
        if multi:
            name_cell = f"[cyan]\\[{index + 1}][/cyan] {name_cell}"
        gpu_cell = _short_gpu_type(cluster.gpu_type)
        paired = _paired_nodes(cluster, index, nodes, ambiguous)
        warning = _cluster_warning(cluster, paired, slurm_jobs)
        if warning:
            warnings.append(f"{name_cell}: {warning}")
        if paired is None:
            table.add_row(name_cell, gpu_cell, *["—"] * 4)
            continue
        nodes_cluster, _ = paired
        jobs = [j for j in slurm_jobs.workloads if j.cluster_display_name == cluster.display_name]
        pools = _pool_usage(nodes_cluster.nodes, jobs, cluster.pools)
        total = _Usage(*(sum(column) for column in zip(*(usage for _, usage in pools))))
        table.add_row(name_cell, gpu_cell, *map(_count_cell, total))
        if verbose:
            for label, usage in pools:
                if usage.nodes:
                    table.add_row(
                        f"  {rich_escape(label)}", "", *map(_count_cell, usage), style="dim"
                    )
    console.print(table)
    for warning in warnings:
        console.print(f"[yellow]{warning}[/yellow]")
    if slurm_jobs.truncated:
        console.print(
            "[yellow]running Slurm job list was truncated — busy Slurm nodes may be undercounted"
            "[/yellow]"
        )
    if verbose:
        console.print(f"[dim]{SLURM_ESTIMATE_NOTE}[/dim]", highlight=False)


def _fetch_usage_view(
    client: FactoryClient, team_id: str
) -> Tuple[FactoryStatus, FactoryNodes, FactoryWorkloads]:
    """Status, node inventory, and running Slurm jobs, fetched in parallel."""
    with ThreadPoolExecutor(max_workers=3) as executor:
        status = executor.submit(client.get_status, team_id)
        nodes = executor.submit(client.get_nodes, team_id)
        slurm_jobs = executor.submit(
            client.get_workloads, team_id, workload_type="slurm", state="running"
        )
        return status.result(), nodes.result(), slurm_jobs.result()


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
        None,
        "--team",
        "-t",
        hidden=True,
        help="Team ID override (defaults to the selected account context)",
    ),
    cluster: Optional[str] = typer.Option(
        None,
        "--cluster",
        help="Show only this cluster, by display name or 1-based index",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Break each cluster down by pool: slurm, training, inference, other, unassigned",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Print the API response as JSON (same as --output json)"
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Show node usage across your team's factory clusters."""

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
        client = FactoryClient(APIClient())
        if output == "json":
            status = client.get_status(team_id)
        else:
            status, nodes, slurm_jobs = _fetch_usage_view(client, team_id)
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

    _render_status_table(status, nodes, slurm_jobs, selected, verbose)


FACTORY_WORKLOADS_JSON_HELP = json_output_help(
    ". = {schema_version, as_of, workloads[], sources[]}",
    ".workloads[] = {id, type, cluster_display_name, name, state, native_state,",
    "                 scheduler_display_name — the slurm cluster",
    "                 (deployment) the job runs on; null = direct platform",
    "                 placement or an unidentified deployment (rendered unknown),",
    "                 owner{kind, display_name}, requested_gpus, allocated_gpus,",
    "                 created_at, started_at, ended_at (terminal rows),",
    "                 reason (when supplied),",
    "                 source{kind, status, observed_at}}",
    ".sources[] = {kind, status, observed_at}",
    "--team: use a different team than the selected account context",
)

# Server-side filter values, mirrored from the public contract.
WORKLOAD_TYPE_FILTERS = ("training", "inference", "slurm")
WORKLOAD_STATE_FILTERS = ("running", "queued", "completed", "failed", "all")
# Terminal (or mixed-history) queries: larger result sets — the DURATION
# column appears and a default --limit applies.
TERMINAL_QUERY_STATES = ("completed", "failed", "all")
WORKLOAD_TERMINAL_STATES = ("completed", "failed")
# History queries can be large; the server caps at 200 rows.
WORKLOAD_LIMIT_MAX = 200
WORKLOAD_TERMINAL_DEFAULT_LIMIT = 50

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
    scheduler_filter_active: bool = False,
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
            elif scheduler_filter_active and source.kind == requested_type:
                # The scheduler predicate removed every row of this kind
                # from the narrowed view; the degraded read may be exactly
                # the omitted portion — keep the warning.
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


def _workload_scheduler_cell(workload: FactoryWorkload) -> str:
    """Scheduler cell: deployment name, 'unknown' for failed Slurm lookups,
    a dash for direct platform placement (training/inference).

    A dash and 'unknown' mean different things: the first is direct
    placement by design, the second is a Slurm row whose deployment could
    not be identified — never render them the same way.
    """
    if workload.scheduler_display_name:
        return rich_escape(workload.scheduler_display_name)
    if workload.type == "slurm":
        return "[dim]unknown[/dim]"
    # The em dash matches SCHEDULER_LEGEND exactly — never the generic
    # missing-value hyphen, so direct placement stays distinguishable.
    return "[dim]—[/dim]"


def _workload_owner_cell(workload: FactoryWorkload) -> str:
    owner = workload.owner
    if owner.display_name is None:
        return "[dim]-[/dim]"
    return rich_escape(owner.display_name)


def _parse_since(value: str) -> datetime:
    """Parse --since: Nd (days ago), Nh (hours ago), or an ISO 8601 stamp."""
    text = value.strip()
    days = re.fullmatch(r"(\d+)d", text)
    if days:
        try:
            return datetime.now(timezone.utc) - timedelta(days=int(days.group(1)))
        except (OverflowError, OSError):
            # Out-of-range relatives are invalid input, never a crash.
            raise ValueError(f"Invalid --since '{text}'.") from None
    hours = re.fullmatch(r"(\d+)h", text)
    if hours:
        try:
            return datetime.now(timezone.utc) - timedelta(hours=int(hours.group(1)))
        except (OverflowError, OSError):
            raise ValueError(f"Invalid --since '{text}'.") from None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"Invalid --since '{text}'.") from None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_duration(started: datetime, ended: datetime) -> str:
    """Compact span humanization for the DURATION column."""
    seconds = int((ended - started).total_seconds())
    if seconds < 0:
        return "[dim]-[/dim]"
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes = remainder // 60
    if days:
        text = f"{days}d"
        if hours:
            text += f" {hours}h"
        return text
    if hours:
        text = f"{hours}h"
        if minutes:
            text += f" {minutes}m"
        return text
    if minutes:
        return f"{minutes}m"
    return f"{seconds}s"


def _workload_duration_cell(workload: FactoryWorkload) -> str:
    """Run duration for terminal rows; dashes when the span is unknown."""
    if (
        workload.started_at is None
        or workload.ended_at is None
        or workload.state not in WORKLOAD_TERMINAL_STATES
    ):
        return "[dim]-[/dim]"
    return _format_duration(workload.started_at, workload.ended_at)


def _workload_age_cell(workload: FactoryWorkload) -> str:
    """Ended-relative age for terminal rows; run/wait age for live ones."""
    if workload.state in WORKLOAD_TERMINAL_STATES:
        if workload.ended_at is None:
            return "[dim]-[/dim]"
        return f"{human_age(workload.ended_at)} ago"
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


def _render_workloads_table(rows: List[FactoryWorkload], show_duration: bool = False) -> Table:
    table = Table(show_header=True, header_style="bold", show_lines=False)
    table.add_column("ID", style="cyan")
    table.add_column("TYPE", style="white")
    table.add_column("NAME")
    table.add_column("CLUSTER")
    # Slurm rows name their deployment; training/inference rows are direct
    # placement (no scheduler application in between).
    table.add_column("SLURM CLUSTER")
    table.add_column("OWNER")
    table.add_column("STATE")
    table.add_column("GPU A/R", justify="right")
    table.add_column("AGE", justify="right")
    # The DURATION column appears only for terminal/history queries; the
    # live squeue view is unchanged.
    if show_duration:
        table.add_column("DURATION", justify="right")
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
            _workload_scheduler_cell(row),
            _workload_owner_cell(row),
            _workload_state_cell(row),
            _workload_gpu_cell(row),
            _workload_age_cell(row),
        ]
        if show_duration:
            cells.append(_workload_duration_cell(row))
        if show_reason:
            cells.append(rich_escape(row.reason) if row.reason else "[dim]-[/dim]")
        table.add_row(*cells)
    return table


def _raw_workload_matches(
    raw_workload: Dict[str, Any],
    user: Optional[str],
    cluster: Optional[str],
    scheduler: Optional[str] = None,
) -> bool:
    """Raw-payload mirror of the client-side --user/--cluster/--scheduler
    predicates."""
    owner = raw_workload.get("owner") or {}
    if user is not None and owner.get("display_name") != user:
        return False
    if cluster is not None and raw_workload.get("cluster_display_name") != cluster:
        return False
    if scheduler is not None and raw_workload.get("scheduler_display_name") != scheduler:
        return False
    return True


def _filter_workload_rows(
    rows: List[FactoryWorkload],
    user: Optional[str],
    cluster: Optional[str],
    scheduler: Optional[str] = None,
) -> List[FactoryWorkload]:
    """Client-side narrowing by owner, cluster, and scheduler deployment."""
    if user is not None:
        rows = [r for r in rows if r.owner.display_name == user]
    if cluster is not None:
        rows = [r for r in rows if r.cluster_display_name == cluster]
    if scheduler is not None:
        rows = [r for r in rows if r.scheduler_display_name == scheduler]
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
        None,
        "--team",
        "-t",
        hidden=True,
        help="Team ID override (defaults to the selected account context)",
    ),
    type: Optional[str] = typer.Option(
        None, "--type", help="Filter by workload type: training, inference, or slurm"
    ),
    state: Optional[str] = typer.Option(
        None,
        "--state",
        help="Filter by state: running, queued, completed, failed, or all",
    ),
    since: Optional[str] = typer.Option(
        None,
        "--since",
        help="Only workloads after this time: Nd, Nh, or an ISO 8601 timestamp",
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        help=(
            "Maximum rows to fetch (1-200; 50 by default for terminal queries, "
            "200 when a --scheduler filter narrows them)"
        ),
    ),
    user: Optional[str] = typer.Option(None, "--user", help="Filter by owner display name"),
    slurm_cluster: Optional[str] = typer.Option(
        None,
        "--slurm-cluster",
        help="Filter by the slurm cluster (deployment) the job runs on",
    ),
    scheduler: Optional[str] = typer.Option(
        None,
        "--scheduler",
        help="Alias of --slurm-cluster",
        hidden=True,
    ),
    cluster: Optional[str] = typer.Option(None, "--cluster", help="Filter by cluster display name"),
    json_output: bool = typer.Option(
        False, "--json", help="Print the API response as JSON (same as --output json)"
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Show your team's factory workloads (training, inference, Slurm jobs)."""

    if slurm_cluster is None:
        # --scheduler is the hidden legacy alias of --slurm-cluster.
        slurm_cluster = scheduler
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
    since_at: Optional[datetime] = None
    if since is not None:
        try:
            since_at = _parse_since(since)
        except ValueError as e:
            # Render the parse error plainly; never leak internals.
            err_console.print(f"[red]Error:[/red] {rich_escape(str(e))}")
            err_console.print("[dim]Use Nd, Nh, or an ISO 8601 timestamp.[/dim]")
            raise typer.Exit(1)
    if limit is not None and not 1 <= limit <= WORKLOAD_LIMIT_MAX:
        err_console.print(
            f"[red]Error:[/red] Invalid --limit '{limit}'. "
            f"Choose a value between 1 and {WORKLOAD_LIMIT_MAX}."
        )
        raise typer.Exit(1)
    if limit is None and state in TERMINAL_QUERY_STATES:
        # History queries can be large; fetch a bounded page by default.
        # A client-side --scheduler narrow happens AFTER the server pages,
        # so request the widest bounded page there — otherwise matching
        # history beyond the default page would be filtered into a false
        # empty result.
        if slurm_cluster is not None:
            limit = WORKLOAD_LIMIT_MAX
        else:
            limit = WORKLOAD_TERMINAL_DEFAULT_LIMIT

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

    # Scheduler deployments are Slurm: with --scheduler and no explicit
    # --type, narrow server-side to slurm so the page fills with the only
    # rows that can match — the client-side predicate then sees all of the
    # deployment's in-window history instead of a mixed, crowded page.
    server_type = type if type is not None else ("slurm" if slurm_cluster is not None else None)

    try:
        api_client = APIClient()
        workloads = FactoryClient(api_client).get_workloads(
            team_id, workload_type=server_type, state=state, since=since_at, limit=limit
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
        payload_kinds = {row.source.kind for row in rows}
        for source in workloads.sources:
            if source.status != "ok":
                if (
                    server_type is not None
                    and source.kind != server_type
                    and source.kind not in payload_kinds
                ):
                    # Kinds narrowed out server-side (--type, or the
                    # implicit slurm scope of --scheduler): their absence
                    # is expected filtering, not a failed read — do not
                    # report unrelated outages on a miss either.
                    continue
                warning = _jobs_unavailable_line(source.kind, source, source.kind in fresh_kinds)
                err_console.print(f"[yellow]{warning}[/yellow]")
        err_console.print(f"[red]Error:[/red] No cluster matched '{rich_escape(cluster)}'.")
        names = _distinct_workload_clusters(rows)
        if names:
            listed = ", ".join(rich_escape(n) for n in names)
            err_console.print(f"[dim]Available clusters: {listed}[/dim]")
        raise typer.Exit(1)
    rows = _filter_workload_rows(rows, user, cluster, scheduler=slurm_cluster)

    if output == "json":
        payload = workloads.raw_response
        if user is not None or cluster is not None or slurm_cluster is not None:
            raw_rows = workloads.raw_response.get("workloads", [])
            # Filter the raw response objects with the same predicates as
            # the parsed models, not re-serialized models, so --json stays
            # an exact passthrough — and duplicate IDs (same id, different
            # owner/cluster) do not let filtered-out twins back in.
            payload = {
                **workloads.raw_response,
                "workloads": [
                    w for w in raw_rows if _raw_workload_matches(w, user, cluster, slurm_cluster)
                ],
            }
        output_data_as_json(payload, console)
        return

    # Per-row freshness governs rendering: a row renders when its OWN
    # evidence is fresh, never suppressed by an aggregate worst-source or a
    # missing sibling group. Degraded aggregates are warnings only.
    available = [row for row in rows if row.source.status == "ok"]
    suppressed = [row for row in rows if row.source.status != "ok"]
    # With a scheduler filter and no explicit --type, only Slurm rows can
    # match: unrelated kind outages are not this view's concern.
    degraded = _view_degraded_sources(
        workloads,
        rows,
        suppressed,
        requested_type=server_type,
        scheduler_filter_active=slurm_cluster is not None,
    )

    # Degraded sources say so before anything else: a failed read must never
    # masquerade as an empty fleet or silently vanish. Fresh rows of the
    # same kind mark mixed coverage — partial data, not "all unavailable".
    fresh_kinds = {row.source.kind for row in rows if row.source.status == "ok"}
    for source in degraded:
        warning = _jobs_unavailable_line(source.kind, source, source.kind in fresh_kinds)
        console.print(f"[yellow]{warning}[/yellow]")

    if available:
        show_duration = state in TERMINAL_QUERY_STATES
        console.print(_render_workloads_table(available, show_duration=show_duration))
        console.print()
        console.print(f"[dim]{IN_USE_NOTE}[/dim]")
        console.print(f"[dim]{SCHEDULER_LEGEND}[/dim]")
        if workloads.truncated:
            # The backend caps the unbounded default window at 200.
            effective = limit if limit is not None else WORKLOAD_LIMIT_MAX
            console.print(f"[dim]{_truncated_note(effective)}[/dim]")
        return

    if not rows and not degraded:
        # Matching-but-suppressed rows are NOT a miss (the warnings above
        # carry their unavailability), and a degraded relevant source can
        # never establish a no-match either — warning-only in both cases.
        _print_empty_workloads_result(
            type,
            state,
            since,
            user,
            cluster,
            slurm_cluster,
            limit,
            bool(workloads.truncated),
            pre_filter_row_count=len(workloads.workloads),
        )
    return None


def _print_empty_workloads_result(
    type: Optional[str],
    state: Optional[str],
    since: Optional[str],
    user: Optional[str],
    cluster: Optional[str],
    slurm_cluster: Optional[str],
    limit: Optional[int],
    truncated: bool = False,
    pre_filter_row_count: int = 0,
) -> None:
    # Distinguish an honestly empty fleet from filters that matched
    # nothing: --type/--state/--since are server-side, so a filtered
    # result of zero rows is not evidence that the team has no workloads.
    filters_active = (
        type is not None
        or state is not None
        or since is not None
        or user is not None
        or cluster is not None
        or slurm_cluster is not None
    )
    # The page was cut before any CLIENT predicate when the backend
    # flagged truncation OR the response actually filled the requested
    # limit — merely SUPPLYING a limit proves nothing (a short response
    # was not cut). Server-side filters (--type/--state/--since) apply
    # before the cap: their zero-row results are definitive.
    client_filters_active = user is not None or cluster is not None or slurm_cluster is not None
    capped_window = None
    if truncated:
        capped_window = limit if limit is not None else WORKLOAD_LIMIT_MAX
    elif limit is not None and pre_filter_row_count >= limit:
        capped_window = limit
    if client_filters_active and capped_window is not None:
        # The page was cut server-side (explicit limit or a truncated
        # envelope) BEFORE any client predicate ran — a miss beyond the
        # fetched window cannot be proven absent.
        console.print(
            "No factory workloads match the given filters in the fetched "
            f"window ({capped_window} rows)."
        )
    elif filters_active:
        # A filtered view matched nothing in the fetched data — say so even
        # when a degraded-source warning also printed (both facts are
        # true; the warning already qualified the data).
        console.print("No factory workloads match the given filters.")
    else:
        # Genuinely nothing running or queued, with fresh evidence.
        console.print("No factory workloads found.")


# Coarse public node states from the frozen nodes contract; the labels are
# the only node vocabulary shown to users.
# v1.5.2: `draining` = a Slurm-claimed node carrying a drain flag
# (takes no new work; GPU counts still show what is ON it).
NODE_STATES = ("ready", "cordoned", "draining", "offline", "unknown")
UNDER_INSPECTION_LEGEND = "under inspection = pulled from scheduling by Prime"
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
        # Draining renders lowercase dim (user spec): the node takes no
        # new work but still holds its GPUs/jobs — a caution note, not
        # the red of a dead node.
        "draining": "dim",
        "offline": "red",
        "unknown": "dim",
    }
    state = node.state or "unknown"
    # Customer-facing display: "cordoned" is infra-speak — the public phrase
    # is "under inspection" (frontierCluster precedent). The contract enum
    # value never renders.
    display = "under inspection" if state == "cordoned" else state
    if state in state_styles:
        style = state_styles[state]
        return f"[{style}]{rich_escape(display)}[/{style}]"
    return rich_escape(display)


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
        None,
        "--team",
        "-t",
        hidden=True,
        help="Team ID override (defaults to the selected account context)",
    ),
    cluster: Optional[str] = typer.Option(
        None, "--cluster", help="Show only this cluster, by display name or 1-based index"
    ),
    state: Optional[str] = typer.Option(
        None,
        "--state",
        help="Filter nodes by state: ready, cordoned, draining, offline, or unknown",
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
    """Show the nodes of your team's factory clusters (sinfo-like)."""

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
            if any(n.state == "cordoned" for n in rows):
                # Footnotes only under a table, and only when the term shows.
                console.print(f"[dim]{UNDER_INSPECTION_LEGEND}[/dim]")
        elif state is not None or assigned_to is not None:
            # Distinguish an honestly node-less cluster from filters that
            # matched nothing.
            console.print("[dim]no nodes match the given filters[/dim]")
        else:
            console.print("[dim]no nodes reported[/dim]")


# The team's Slurm deployments live under the factory namespace: the
# dedicated top-level surface is superseded by this sub-app.
from .factory_slurm import app as slurm_app  # noqa: E402

app.add_typer(slurm_app, name="slurm", rich_help_panel="Model Factory")
