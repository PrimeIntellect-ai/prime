"""Display utilities for table and JSON output."""

import json
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urlencode

import typer
from rich.console import Console
from rich.table import Table

from prime_cli.core import APIError, Config


def validate_output_format(output: str, console: Console) -> None:
    """Validate that output format is supported."""
    if output not in ["table", "json"]:
        console.print(
            f"[red]Error: Invalid output format '{output}'. Supported formats: table, json[/red]"
        )
        raise typer.Exit(1)


def output_data_as_json(data: Any, console: Console) -> None:
    """Output data as formatted JSON.

    `soft_wrap=True` disables Rich's terminal-width wrapping so long string
    values don't get a literal newline injected in the middle and break
    parsers (e.g. `prime train usage` run names regularly exceed 80 chars).
    """
    console.print(
        json.dumps(data, indent=2, default=str),
        markup=False,
        highlight=False,
        emoji=False,
        soft_wrap=True,
    )


def api_detail(exc: APIError) -> str:
    """The route's detail without the "HTTP 400: " / "HTTP 409: " prefix."""
    return re.sub(r"^HTTP (400|409): ", "", str(exc))


def build_table(title: str, columns: List[Tuple[str, str]], show_lines: bool = True) -> Table:
    """
    Build a Rich table with standard styling.

    Args:
        title: Table title
        columns: List of (header, style) tuples
        show_lines: Whether to show row separator lines
    """
    table = Table(title=title, show_lines=show_lines)
    for header, style in columns:
        table.add_column(header, style=style, no_wrap=(header == "ID"))
    return table


def status_color(status: str, mapping: Dict[str, str], default: str = "white") -> str:
    """Get color for status based on mapping with fallback to default."""
    return mapping.get(status, default)


def get_eval_viewer_url(evaluation_id: str) -> str:
    """Build the dashboard URL for an evaluation."""
    frontend_url = Config().frontend_url.rstrip("/")
    return f"{frontend_url}/dashboard/evaluations/{evaluation_id}"


def get_traces_viewer_url(
    run_id: str,
    *,
    episode_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    step: Optional[int] = None,
    frontend_url: Optional[str] = None,
) -> str:
    """Build the dashboard Prime Traces viewer URL for a run, or one trace in it.

    Mirrors the platform frontend's `dashboard/training/viewer/[runId]/traces`
    route and its `useRolloutUrlState` query keys. The viewer opens a trace only
    by its episode, so without `episode_id` this links the run page. The team is
    not in the URL: the page shows the team selected in the account switcher.
    """
    base = (frontend_url or Config().frontend_url).rstrip("/")
    url = f"{base}/dashboard/training/viewer/{quote(run_id, safe='')}/traces"
    if episode_id is None:
        return url
    query: Dict[str, Any] = {"episodeId": episode_id}
    if trace_id is not None:
        query["traceId"] = trace_id
    query["viewerLayout"] = "viewer"
    if step is not None:
        # The page lists one training step at a time, the latest by default.
        query["step"] = step
    return f"{url}?{urlencode(query, quote_via=quote)}"


# Common status color mappings
SANDBOX_STATUS_COLORS = {
    "PENDING": "yellow",
    "PROVISIONING": "yellow",
    "RUNNING": "green",
    "PAUSED": "blue",
    "ERROR": "red",
    "TERMINATED": "white",
    "TIMEOUT": "white",
}

POD_STATUS_COLORS = {
    "ACTIVE": "green",
    "PENDING": "yellow",
    "ERROR": "red",
    "INSTALLING": "yellow",
}

STOCK_STATUS_COLORS = {
    "High": "green",
    "Medium": "yellow",
    "Low": "red",
}

DISK_STATUS_COLORS = {
    "ACTIVE": "green",
    "PROVISIONING": "yellow",
    "PENDING": "yellow",
    "STOPPED": "blue",
    "ERROR": "red",
    "TERMINATED": "white",
}

SLURM_CLUSTER_STATUS_COLORS = {
    "PENDING": "yellow",
    "DEPLOYING": "yellow",
    "RUNNING": "green",
    "TERMINATING": "yellow",
    "COMPLETED": "white",
    "FAILED": "red",
    "STOPPED": "blue",
    "UNKNOWN": "white",
    "TOMBSTONED": "white",
}

SLURM_CLUSTER_STATUS_COLORS = {
    "PENDING": "yellow",
    "DEPLOYING": "yellow",
    "RUNNING": "green",
    "TERMINATING": "yellow",
    "COMPLETED": "white",
    "FAILED": "red",
    "STOPPED": "blue",
    "UNKNOWN": "white",
    "TOMBSTONED": "white",
}
