"""`prime pipelines`: run a pipeline (a graph of agent stages) and fetch what it made."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, List, NoReturn, Optional
from urllib.parse import quote

import httpx
import typer
from rich.markup import escape
from rich.table import Table

from ..client import APIClient, APIError
from ..core import Config
from ..utils import PlainTyper, get_console, output_data_as_json, validate_output_format

app = PlainTyper(
    help="Run a pipeline (a graph of agent stages) on the platform, and fetch what it made.",
    no_args_is_help=True,
)
console = get_console()


def _fail(error: Exception) -> NoReturn:
    console.print(f"[red]Error:[/red] {escape(str(error))}")
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
        found = APIClient().get(f"/pipelines/runs/{quote(run_id, safe='')}")
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
    if state.get("cards"):
        console.print(f"Answer with: prime pipelines cards {found['id']}")
    if cost := (state.get("cost") or {}).get("usd"):
        console.print(f"Cost so far: ${cost:.2f}")
    for out in found.get("outputs") or []:
        console.print(f"Output: {escape(out['node'] + '/' + out['name'])} ({out['bytes']} bytes)")


@app.command("outputs")
def outputs(
    run_id: str = typer.Argument(..., help="The run's id."),
    directory: Path = typer.Option(Path("."), "--dir", "-d", help="Where to save them."),
) -> None:
    """Download what a run's heads made: its report, its taskset package."""
    client = APIClient()
    try:
        found = client.get(f"/pipelines/runs/{quote(run_id, safe='')}")
    except APIError as error:
        _fail(error)
    listed = found.get("outputs") or []
    if not listed:
        console.print("This run has no outputs yet.")
        return
    directory.mkdir(parents=True, exist_ok=True)
    for out in listed:
        parts = (str(out["node"]), str(out["name"]))
        if any(
            not part or part in (".", "..") or "/" in part or "\\" in part or ":" in part
            for part in parts
        ):
            _fail(APIError(f"refusing an output named {out['node']}/{out['name']}"))
        target = directory / parts[0] / parts[1]
        target.parent.mkdir(parents=True, exist_ok=True)
        partial: Optional[Path] = None
        url = (
            f"{client.base_url}/api/v1/pipelines/runs/{quote(run_id, safe='')}"
            f"/outputs/{quote(str(out['node']), safe='')}/{quote(str(out['name']), safe='')}"
        )
        try:
            timeout = httpx.Timeout(300.0, connect=10.0)
            with client.client.stream("GET", url, timeout=timeout) as response:
                response.raise_for_status()
                # A new file with a random name (O_EXCL), so nothing already there is followed.
                handle, name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
                partial = Path(name)
                with os.fdopen(handle, "wb") as file:
                    for chunk in response.iter_bytes():
                        file.write(chunk)
            # Only a complete download replaces what was there.
            partial.replace(target)
        except (httpx.HTTPError, OSError) as error:
            if partial is not None:
                partial.unlink(missing_ok=True)
            _fail(APIError(f"downloading {out['name']}: {error}"))
        console.print(f"Saved {escape(str(target))}")


@app.command("stop")
def stop(run_id: str = typer.Argument(..., help="The run's id.")) -> None:
    """Stop a run. Its runner and every sandbox it started end."""
    try:
        stopped = APIClient().post(f"/pipelines/runs/{quote(run_id, safe='')}/stop")
    except APIError as error:
        _fail(error)
    _summary(stopped)


def _run(run_id: str) -> dict[str, Any]:
    try:
        return APIClient().get(f"/pipelines/runs/{quote(run_id, safe='')}")
    except APIError as error:
        _fail(error)


def _control(run_id: str, control: dict[str, Any]) -> int:
    """Queue a control for the run; its seq."""
    try:
        queued = APIClient().post(
            f"/pipelines/runs/{quote(run_id, safe='')}/control", json={"control": control}
        )
    except APIError as error:
        _fail(error)
    return int(queued["seq"])


def _kind(card: dict[str, Any]) -> str:
    """As the canvas reads a card: options to choose from, items to keep, or a question."""
    if card.get("options"):
        return "choice"
    return "review" if card.get("items") else "question"


@app.command("cards")
def cards(
    run_id: str = typer.Argument(..., help="The run's id."),
    output: str = typer.Option("table", "--output", "-o", help="table or json"),
) -> None:
    """The cards a run is waiting on you to answer, and how to answer each."""
    validate_output_format(output, console)
    state = _run(run_id).get("state") or {}
    waiting = state.get("cards") or []
    skipped = state.get("skipped_controls") or []
    if output == "json":
        output_data_as_json({"cards": waiting, "skipped_controls": skipped}, console)
        return
    for control in skipped:
        console.print(
            f"[red]Not applied[/red] (control {control['seq']}): {escape(control['error'])}"
        )
    if not waiting:
        console.print("Nothing is waiting on you.")
        return
    for card in waiting:
        kind = _kind(card)
        console.print(
            f"[bold]{escape(card['node'])}[/bold] (job {escape(card['job'])}), {kind}: "
            f"{escape(card['question'])}"
        )
        if card.get("reason"):
            console.print(f"  {escape(card['reason'])}")
        if kind == "choice":
            console.print(f"  Options: {escape(', '.join(card['options']))}")
        for item in card.get("items") if kind == "review" else []:
            console.print(f"  - {escape(item)}")
        where = f"{run_id} --node {card['node']} --job {card['job']}"
        how = {
            "choice": "--choice OPTION",
            "review": "--keep ITEM ... | --drop ITEM ... | --keep-all",
            "question": "--note TEXT",
        }[kind]
        console.print(f"  Answer: prime pipelines reply {escape(where)} {escape(how)}")


def _card(waiting: list[dict[str, Any]], node: Optional[str], job: Optional[str]) -> dict[str, Any]:
    """The one waiting card at `node` and `job`; either may be left out while only one matches."""
    matches = [
        card
        for card in waiting
        if (node is None or card["node"] == node) and (job is None or card["job"] == job)
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        _fail(ValueError("no waiting card matches; see prime pipelines cards"))
    listed = ", ".join(f"--node {c['node']} --job {c['job']}" for c in matches)
    _fail(ValueError(f"{len(matches)} cards match; name one with {listed}"))


@app.command("reply")
def reply(
    run_id: str = typer.Argument(..., help="The run's id."),
    node: Optional[str] = typer.Option(
        None, "--node", help="The card's node (needed when several cards wait)."
    ),
    job: Optional[str] = typer.Option(
        None, "--job", help="The card's job (needed when a node has several cards)."
    ),
    choice: Optional[str] = typer.Option(None, "--choice", help="One of the card's options."),
    keep: List[str] = typer.Option(
        [], "--keep", help="On a review, an item to keep; the rest are dropped. Repeat for more."
    ),
    drop: List[str] = typer.Option(
        [], "--drop", help="On a review, an item to drop; the rest are kept. Repeat for more."
    ),
    keep_all: bool = typer.Option(False, "--keep-all", help="On a review, keep every item."),
    note: str = typer.Option(
        "", "--note", help="A note for the next step; the answer itself on a question."
    ),
) -> None:
    """Answer a card the run is waiting on, as the canvas does."""
    card = _card((_run(run_id).get("state") or {}).get("cards") or [], node, job)
    kind = _kind(card)
    answer: dict[str, Any] = {"note": note}
    if kind != "choice" and choice is not None:
        _fail(ValueError(f"this card is a {kind}; it has no options to choose from"))
    if kind != "review" and (keep or drop or keep_all):
        _fail(ValueError(f"this card is a {kind}; it has no items to keep"))
    if kind == "choice":
        if choice not in card["options"]:
            _fail(ValueError(f"choose one of: {', '.join(card['options'])}"))
        answer["choice"] = choice
    elif kind == "review":
        if sum((bool(keep), bool(drop), keep_all)) != 1:
            _fail(ValueError("say what to keep with one of --keep, --drop or --keep-all"))
        unknown = [item for item in (*keep, *drop) if item not in card["items"]]
        if unknown:
            _fail(ValueError(f"not an item of this card: {', '.join(unknown)}"))
        if keep_all:
            keep = list(card["items"])
        elif drop:
            keep = [item for item in card["items"] if item not in drop]
        answer["keep"] = [item for item in card["items"] if item in keep]
    elif not note.strip():
        _fail(ValueError("answer the question with --note"))
    seq = _control(
        run_id, {"type": "reply", "job": card["job"], "node": card["node"], "reply": answer}
    )
    kept = f", keeping {len(answer['keep'])} of {len(card['items'])}" if kind == "review" else ""
    console.print(
        f"Sent (control {seq}){escape(kept)}. The run picks it up in a few seconds; "
        f"if it can't, prime pipelines cards {escape(run_id)} says why."
    )


@app.command("refresh")
def refresh(
    run_id: str = typer.Argument(..., help="The run's id."),
    node: str = typer.Argument(..., help="A live node of the run's graph."),
) -> None:
    """Fetch a live node again: the run relaunches, rerunning it and whatever its new output
    reaches."""
    seq = _control(run_id, {"type": "refresh", "node": node})
    console.print(
        f"Sent (control {seq}). The run drains and relaunches with {escape(node)} fetched again."
    )


@app.command("drain")
def drain(run_id: str = typer.Argument(..., help="The run's id.")) -> None:
    """Wind a run down: it starts no new work, waits for what's running, then ends. Stop ends it
    at once."""
    seq = _control(run_id, {"type": "drain"})
    console.print(f"Sent (control {seq}). The run ends once what's running finishes.")
