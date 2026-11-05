"""`prime pipelines`: run a pipeline (a Flow graph) on the platform, and fetch what it made."""

import json
import os
from pathlib import Path
from typing import Any, List, Optional

import httpx
import typer
from rich.markup import escape
from rich.table import Table

from ..client import APIClient, APIError
from ..core import Config
from ..utils import PlainTyper, get_console, output_data_as_json, validate_output_format

app = PlainTyper(
    help="Run a pipeline (a Flow graph) on the platform, and fetch what it made.",
    no_args_is_help=True,
)
console = get_console()


def _fail(error: Exception) -> None:
    console.print(f"[red]Error:[/red] {error}")
    raise typer.Exit(1)


def _summary(run: dict[str, Any]) -> None:
    stopped = " (stopped)" if run.get("stopped") else ""
    console.print(f"Run [bold]{run['id']}[/bold]: {run['status']}{stopped}")
    if run.get("pipeline_id"):
        console.print(f"Pipeline: {run['pipeline_id']}")
    console.print(f"Version: {run['version'][:12]}")
    if run.get("error") and not run.get("stopped"):
        console.print(f"[red]{escape(run['error'])}[/red]")


@app.command("run")
def run(
    graph: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="The graph file, in either form."
    ),
    team: Optional[str] = typer.Option(
        None, "--team", help="Run as this team (default: your selected team)."
    ),
    pipeline_id: Optional[str] = typer.Option(
        None,
        "--pipeline-id",
        help="Save the graph as a new version of this pipeline (default: a new pipeline).",
    ),
    sample: Optional[int] = typer.Option(
        None, "--sample", min=1, help="Preview: items each input runs on."
    ),
    secret: List[str] = typer.Option(
        [],
        "--secret",
        help="A secret a Custom node names, read from your environment by name. Repeat for more.",
    ),
    output: str = typer.Option("table", "--output", "-o", help="table or json"),
) -> None:
    """Save a graph as a pipeline and start a run of it."""
    validate_output_format(output, console)
    try:
        data = json.loads(graph.read_text())
    except ValueError as error:
        _fail(ValueError(f"{graph} is not JSON: {error}"))
    missing = [name for name in secret if name not in os.environ]
    if missing:
        _fail(ValueError(f"set {', '.join(missing)} in your environment"))
    body = {
        "graph": data,
        "team_id": team or Config().team_id,
        "pipeline_id": pipeline_id,
        "sample": sample,
        "secrets": {name: os.environ[name] for name in secret},
    }
    try:
        started = APIClient().post("/pipelines/runs", json=body, timeout=120)
    except APIError as error:
        _fail(error)
    if output == "json":
        output_data_as_json(started, console)
        return
    _summary(started)
    console.print(f"Follow it with: prime pipelines status {started['id']}")


@app.command("status")
def status(
    run_id: str = typer.Argument(..., help="The run's id."),
    output: str = typer.Option("table", "--output", "-o", help="table or json"),
) -> None:
    """A run's status: per-node counts, what waits on a person, its cost and outputs."""
    validate_output_format(output, console)
    try:
        found = APIClient().get(f"/pipelines/runs/{run_id}")
    except APIError as error:
        _fail(error)
    if output == "json":
        output_data_as_json(found, console)
        return
    _summary(found)
    state = found.get("state") or {}
    if nodes := state.get("nodes"):
        table = Table("node", "done", "running", "waiting on you", "failed", "cost")
        for node, counts in nodes.items():
            table.add_row(
                escape(node),
                str(counts.get("done", 0)),
                str(counts.get("running", 0)),
                str(counts.get("held", 0)),
                str(counts.get("failed", 0)),
                f"${counts.get('cost', 0):.2f}",
            )
        console.print(table)
    for card in state.get("cards") or []:
        where = f"{card['node']} ({card['job']})"
        console.print(
            f"[yellow]Waiting on you[/yellow] at {escape(where)}: {escape(card['question'])}"
        )
    if cost := (state.get("cost") or {}).get("usd"):
        console.print(f"Cost so far: ${cost:.2f}")
    for out in found.get("outputs") or []:
        console.print(f"Output: {out['node']}/{out['name']} ({out['bytes']} bytes)")


@app.command("outputs")
def outputs(
    run_id: str = typer.Argument(..., help="The run's id."),
    directory: Path = typer.Option(Path("."), "--dir", "-d", help="Where to save them."),
) -> None:
    """Download what a run's heads made: its report, its taskset package."""
    client = APIClient()
    try:
        found = client.get(f"/pipelines/runs/{run_id}")
    except APIError as error:
        _fail(error)
    listed = found.get("outputs") or []
    if not listed:
        console.print("This run has no outputs yet.")
        return
    directory.mkdir(parents=True, exist_ok=True)
    for out in listed:
        parts = (str(out["node"]), str(out["name"]))
        if any(not part or part in (".", "..") or "/" in part or "\\" in part for part in parts):
            _fail(APIError(f"refusing an output named {out['node']}/{out['name']}"))
        target = directory / parts[0] / parts[1]
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f".{target.name}.part")
        url = f"{client.base_url}/api/v1/pipelines/runs/{run_id}/outputs/{out['name']}"
        try:
            timeout = httpx.Timeout(300.0, connect=10.0)
            with client.client.stream("GET", url, timeout=timeout) as response:
                response.raise_for_status()
                with partial.open("wb") as file:
                    for chunk in response.iter_bytes():
                        file.write(chunk)
            # Only a complete download replaces what was there.
            partial.replace(target)
        except httpx.HTTPError as error:
            partial.unlink(missing_ok=True)
            _fail(APIError(f"downloading {out['name']}: {error}"))
        console.print(f"Saved {target}")


@app.command("stop")
def stop(run_id: str = typer.Argument(..., help="The run's id.")) -> None:
    """Stop a run. Its runner and every sandbox it started end."""
    try:
        stopped = APIClient().post(f"/pipelines/runs/{run_id}/stop")
    except APIError as error:
        _fail(error)
    _summary(stopped)
