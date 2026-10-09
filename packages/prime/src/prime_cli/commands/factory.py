"""`prime factory` — Model Factory fleet status."""

from typing import Any, List, Optional

import typer
from rich.markup import escape as rich_escape
from rich.table import Table

from ..api.factory import FactoryClient, FactoryCluster, FactorySource
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

IN_USE_NOTE = "IN USE = allocated to leaf workloads, not measured GPU activity."


def _fmt_count(value: Optional[int]) -> str:
    """Render a GPU count; unknown values stay '?' instead of a tidy zero."""
    return "?" if value is None else str(value)


def _describe_source(source: FactorySource) -> str:
    """Render one source's freshness label: `<kind> Ns ago`, or its coarse status."""
    age = f" {human_age(source.observed_at)} ago" if source.observed_at else ""
    if source.status == "ok":
        return f"{source.kind}{age}" if age else source.kind
    return f"{source.kind} {source.status}{age}"


def _styled_status(status: Optional[str]) -> str:
    if status == "online":
        return "[green]online[/green]"
    if status == "offline":
        return "[red]offline[/red]"
    return rich_escape(status or "unknown")


def _cluster_header(cluster: FactoryCluster, index: int, multi: bool) -> str:
    parts: List[str] = []
    label = rich_escape(cluster.display_name)
    if multi:
        label = f"[cyan]\\[{index}][/cyan] {label}"
    parts.append(f"[bold]{label}[/bold]")

    gpu_bits = [str(cluster.total_gpus)] if cluster.total_gpus is not None else []
    if cluster.gpu_type:
        gpu_bits.append(cluster.gpu_type)
    if gpu_bits:
        parts.append(" ".join(gpu_bits) + " GPUs")
    parts.append(_styled_status(cluster.status))
    if cluster.sources:
        parts.append(" · ".join(_describe_source(s) for s in cluster.sources))
    return " · ".join(parts)


def _render_pool_table(cluster: FactoryCluster) -> Table:
    table = Table(show_header=True, header_style="bold", show_lines=False)
    table.add_column("POOL", style="cyan")
    table.add_column("RESERVED", style="white", justify="right")
    table.add_column("IN USE", style="green", justify="right")
    table.add_column("IDLE INSIDE", style="blue", justify="right")
    table.add_column("UNKNOWN", style="yellow", justify="right")

    for pool in cluster.pools:
        table.add_row(
            rich_escape(pool.type),
            _fmt_count(pool.reserved_gpus),
            _fmt_count(pool.in_use_gpus),
            _fmt_count(pool.idle_inside_gpus),
            _fmt_count(pool.unknown_gpus),
        )
    table.add_row("unassigned", _fmt_count(cluster.unassigned_gpus), "-", "-", "-")
    table.add_row("unknown", _fmt_count(cluster.unknown_gpus), "-", "-", "-")
    return table


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
            f"[red]Error:[/red] '{selector}' matches multiple clusters. "
            "Use its 1-based index from `prime factory status` instead."
        )
        raise typer.Exit(1)
    if matches:
        return matches

    if selector.isdigit():
        index = int(selector)
        if 1 <= index <= len(clusters):
            return [index - 1]

    err_console.print(f"[red]Error:[/red] No cluster matched '{selector}'.")
    if clusters:
        names = ", ".join(f"[{i + 1}] {c.display_name}" for i, c in enumerate(clusters))
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

    multi = len(clusters) > 1
    for idx, c in enumerate(clusters, start=1):
        if idx > 1:
            console.print()
        console.print(_cluster_header(c, idx, multi))
        console.print(_render_pool_table(c))
        degraded = [s for s in c.sources if s.status != "ok"]
        if degraded:
            console.print(
                f"[yellow]Warning:[/yellow] "
                f"{', '.join(_describe_source(s) for s in degraded)} — "
                "allocation counts may be incomplete."
            )
    console.print(f"[dim]{IN_USE_NOTE}[/dim]")
