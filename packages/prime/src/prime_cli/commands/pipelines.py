"""`prime pipelines`: run a pipeline (a graph of agent stages) and fetch what it made."""

import json
import os
import re
import shlex
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
from ..utils import (
    PlainTyper,
    get_console,
    json_output_help,
    output_data_as_json,
    validate_output_format,
)
from ..utils.prompt import confirm_or_skip

app = PlainTyper(
    help="Run a pipeline (a graph of agent stages) on the platform, and fetch what it made.",
    no_args_is_help=True,
)
console = get_console()

_RUN_FIELDS = (
    "id, pipeline_id, version, status, error?, stopped, sample?, "
    "outputs[{node, name, bytes}], network{allow[]}, created_at, started_at?, completed_at?"
)
RUN_JSON_HELP = json_output_help(f". = {{{_RUN_FIELDS}}}")
LIST_JSON_HELP = json_output_help(
    ".pipelines[] = {id, name, team_id?, version?, updated_at, runs, latest_run?}",
    f".pipelines[].latest_run = {{{_RUN_FIELDS}}}",
    ".total = number",
    ".page = number",
    ".per_page = number",
)
RUNS_JSON_HELP = json_output_help(
    f".runs[] = {{{_RUN_FIELDS}}}",
    ".total = number",
    ".page = number",
    ".per_page = number",
)
# A pipeline's or a run's id, as the platform's routes take it.
_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
GET_JSON_HELP = json_output_help(
    f". = {{{_RUN_FIELDS}, state?}}",
    ".state.nodes.<node> = {done, running, held, failed, cost}",
    ".state.cards[] = {node, job, question, reason?, options?[], items?[]} — what waits on you",
    ".state.skipped_controls[] = {seq, error} — replies the run couldn't apply",
    ".state.cost = {usd}",
)


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


def _run(run_id: str) -> dict[str, Any]:
    try:
        return APIClient().get(f"/pipelines/runs/{quote(run_id, safe='')}")
    except APIError as error:
        _fail(error)


AGREEMENTS = {"read_personal_traces": "--read-personal-traces", "run_code": "--run-code"}
"""What a team run asks its starter to agree to, as the platform's request names it, and the option
that agrees."""


def _with_options(error: APIError, agreed: set[str]) -> APIError:
    """The platform's refusal, with the option for what it asks for by its field name: an
    agreement, or which of several pipelines of one name to add the graph to."""
    if "with pipeline_id" in str(error):
        return APIError(f"{error} Pass one of these ids with --pipeline.")
    wanted = [o for name, o in AGREEMENTS.items() if name in str(error) and name not in agreed]
    return APIError(f"{error} To agree, pass {' and '.join(wanted)}.") if wanted else error


@app.command("run", epilog=RUN_JSON_HELP)
def run(
    target: str = typer.Argument(
        ...,
        metavar="GRAPH_FILE | PIPELINE",
        help="A graph file, in either form, to save and run; or a saved pipeline's id, to run "
        "its latest version.",
    ),
    team: Optional[str] = typer.Option(
        None,
        "--team",
        "-t",
        help="With a graph file: run as this team (default: your selected team). A saved "
        "pipeline runs as its own account.",
    ),
    pipeline: Optional[str] = typer.Option(
        None,
        "--pipeline",
        help="With a graph file: save it as a new version of this pipeline (default: the "
        "pipeline that has the graph's name, or a new one).",
    ),
    sample: Optional[int] = typer.Option(
        None, "--sample", min=1, help="Preview: items each input runs on."
    ),
    secret: List[str] = typer.Option(
        [],
        "--secret",
        help="A secret a Code node names, read from your environment by name. Repeat for more.",
    ),
    allow_host: List[str] = typer.Option(
        [],
        "--allow-host",
        help="A host this run's machine may reach besides the platform's, such as api.github.com. "
        "Repeat for more.",
    ),
    read_personal_traces: bool = typer.Option(
        False,
        "--read-personal-traces",
        help="On a team run: agree that Traces inputs set to your personal account read your "
        "personal traces into outputs the team can see.",
    ),
    run_code: bool = typer.Option(
        False,
        "--run-code",
        help="On a team run: agree to run the graph's Code nodes. Their source, which any member "
        "of the team may have saved, runs on the run's machine, where it can use your key and the "
        "secrets you pass. Read it first.",
    ),
    output: str = typer.Option("table", "--output", "-o", help="table or json"),
) -> None:
    """Start a run: of a graph file, saved as a version of the pipeline that has its name; or of
    a saved pipeline, by its id."""
    validate_output_format(output, console)
    missing = [name for name in secret if name not in os.environ]
    if missing:
        _fail(ValueError(f"set {', '.join(missing)} in your environment"))
    body: dict[str, Any] = {
        "sample": sample,
        "secrets": {name: os.environ[name] for name in secret},
        "network": {"allow": allow_host},
        "read_personal_traces": read_personal_traces,
        "run_code": run_code,
    }
    if Path(target).is_file():
        try:
            data = json.loads(Path(target).read_text())
        except OSError as error:
            _fail(ValueError(f"can't read {target}: {error.strerror or error}"))
        except ValueError as error:
            _fail(ValueError(f"{target} is not JSON: {error}"))
        body |= {"graph": data, "team_id": team or Config().team_id, "pipeline_id": pipeline}
    elif pipeline is not None:
        _fail(ValueError(f"{target} is not a file; --pipeline goes with a graph file"))
    elif _ID.match(target):
        # The pipeline's latest version runs, as the pipeline's own account.
        body["pipeline_id"] = target
    else:
        _fail(ValueError(f"{target} is neither a file nor a pipeline's id"))
    agreed = {name for name in AGREEMENTS if body[name]}
    try:
        started = APIClient().post("/pipelines/runs", json=body, timeout=120)
    except APIError as error:
        _fail(_with_options(error, agreed))
    if output == "json":
        output_data_as_json(started, console)
        return
    _summary(started)
    console.print(f"Follow it with: prime pipelines get {started['id']}")


def _when(value: Optional[str]) -> str:
    """A time the platform sent, to the minute, in UTC."""
    return value[:16].replace("T", " ") if value else ""


def _status(found: dict[str, Any]) -> str:
    return "STOPPED" if found.get("stopped") else str(found["status"])


def _page(page: int, num: int, total: int, shown: int, what: str) -> None:
    """The line under a list: where the next page is, or how many there are."""
    if page * num < total:
        console.print(f"[dim]{shown} of {total} {what}. Use --page {page + 1} to see more.[/dim]")
    else:
        console.print(f"[dim]Total: {total} {what}[/dim]")


def _listing(endpoint: str, params: dict[str, Any], num: int, page: int) -> dict[str, Any]:
    if num < 1 or page < 1:
        _fail(ValueError("--num and --page must be at least 1"))
    wanted = {**params, "page": page, "per_page": num}
    try:
        return APIClient().get(endpoint, params={k: v for k, v in wanted.items() if v})
    except APIError as error:
        _fail(error)


@app.command("list", epilog=LIST_JSON_HELP)
def list_pipelines(
    team: Optional[str] = typer.Option(
        None, "--team", "-t", help="This team's pipelines (default: your selected team's)."
    ),
    num: int = typer.Option(20, "--num", "-n", help="Items per page"),
    page: int = typer.Option(1, "--page", "-p", help="Page number"),
    output: str = typer.Option("table", "--output", "-o", help="table or json"),
) -> None:
    """Your pipelines, last saved first, each with its latest run."""
    validate_output_format(output, console)
    found = _listing("/pipelines", {"team_id": team or Config().team_id}, num, page)
    if output == "json":
        output_data_as_json(found, console)
        return
    if not found["pipelines"]:
        console.print("No more pipelines." if page > 1 else "No pipelines yet.")
        return
    table = Table("id", "name", "runs", "latest run", "status", "saved")
    for row in found["pipelines"]:
        latest = row.get("latest_run") or {}
        table.add_row(
            escape(row["id"]),
            escape(row["name"]),
            str(row["runs"]),
            escape(latest.get("id", "")),
            _status(latest) if latest else "",
            _when(row["updated_at"]),
        )
    console.print(table)
    _page(page, num, found["total"], len(found["pipelines"]), "pipelines")


@app.command("runs", epilog=RUNS_JSON_HELP)
def list_runs(
    pipeline: Optional[str] = typer.Argument(
        None, help="Only this pipeline's runs (default: the runs of all your pipelines)."
    ),
    team: Optional[str] = typer.Option(
        None,
        "--team",
        "-t",
        help="Without a pipeline: the runs of this team's pipelines (default: your selected "
        "team's).",
    ),
    num: int = typer.Option(20, "--num", "-n", help="Items per page"),
    page: int = typer.Option(1, "--page", "-p", help="Page number"),
    output: str = typer.Option("table", "--output", "-o", help="table or json"),
) -> None:
    """Runs, newest first: of one pipeline, or of all your pipelines."""
    validate_output_format(output, console)
    # A pipeline has its own account, so no team goes with it.
    where = {"pipeline_id": pipeline} if pipeline else {"team_id": team or Config().team_id}
    found = _listing("/pipelines/runs", where, num, page)
    if output == "json":
        output_data_as_json(found, console)
        return
    if not found["runs"]:
        console.print("No more runs." if page > 1 else "No runs yet.")
        return
    table = Table("id", "pipeline", "status", "version", "started", "outputs")
    for row in found["runs"]:
        table.add_row(
            escape(row["id"]),
            escape(row["pipeline_id"]),
            _status(row),
            escape(row["version"][:12]),
            _when(row.get("started_at") or row["created_at"]),
            str(len(row.get("outputs") or [])),
        )
    console.print(table)
    _page(page, num, found["total"], len(found["runs"]), "runs")


def _kind(card: dict[str, Any]) -> str:
    """As the canvas reads a card: options to choose from, items to keep, or a question."""
    if card.get("options"):
        return "choice"
    return "review" if card.get("items") else "question"


def _waiting(run_id: str, card: dict[str, Any]) -> None:
    """A question the run waits on, and the command that answers it."""
    kind = _kind(card)
    console.print(
        f"[yellow]Waiting on you[/yellow] at [bold]{escape(card['node'])}[/bold] "
        f"(job {escape(card['job'])}), {kind}: {escape(card['question'])}"
    )
    if card.get("reason"):
        console.print(f"  {escape(card['reason'])}")
    if kind == "choice":
        console.print(f"  Options: {escape(', '.join(card['options']))}")
    for item in card["items"] if kind == "review" else []:
        console.print(f"  - {escape(item)}")
    # Quoted for a shell: the command is there to be copied, and the names in it are the run's.
    where = shlex.join([run_id, "--node", card["node"], "--job", card["job"]])
    how = {
        "choice": "--choice OPTION",
        "review": "--keep ITEM ... | --drop ITEM ... | --keep-all",
        "question": "--note TEXT",
    }[kind]
    console.print(f"  Answer: prime pipelines reply {escape(where)} {escape(how)}")


@app.command("get", epilog=GET_JSON_HELP)
def get(
    run_id: str = typer.Argument(..., help="The run's id."),
    output: str = typer.Option("table", "--output", "-o", help="table or json"),
) -> None:
    """A run: its status, per-node counts, what waits on you and how to answer, its cost and
    outputs."""
    validate_output_format(output, console)
    found = _run(run_id)
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
    for control in state.get("skipped_controls") or []:
        console.print(
            f"[red]Not applied[/red] (control {control['seq']}): {escape(control['error'])}"
        )
    for card in state.get("cards") or []:
        _waiting(run_id, card)
    if cost := (state.get("cost") or {}).get("usd"):
        console.print(f"Cost so far: ${cost:.2f}")
    for out in found.get("outputs") or []:
        console.print(f"Output: {escape(out['node'] + '/' + out['name'])} ({out['bytes']} bytes)")


def _control(run_id: str, control: dict[str, Any]) -> int:
    """Queue a control for the run; its seq."""
    try:
        queued = APIClient().post(
            f"/pipelines/runs/{quote(run_id, safe='')}/control", json={"control": control}
        )
    except APIError as error:
        _fail(error)
    return int(queued["seq"])


def _card(
    run_id: str, waiting: list[dict[str, Any]], node: Optional[str], job: Optional[str]
) -> dict[str, Any]:
    """The one waiting question at `node` and `job`; either may be left out while only one
    matches."""
    matches = [
        card
        for card in waiting
        if (node is None or card["node"] == node) and (job is None or card["job"] == job)
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        _fail(ValueError(f"nothing waiting matches; see prime pipelines get {run_id}"))
    # Quoted for a shell, as the Answer line is: the names are the run's.
    listed = ", ".join(shlex.join(["--node", c["node"], "--job", c["job"]]) for c in matches)
    _fail(ValueError(f"{len(matches)} questions match; name one with {listed}"))


@app.command("reply")
def reply(
    run_id: str = typer.Argument(..., help="The run's id."),
    node: Optional[str] = typer.Option(
        None, "--node", help="The question's node (needed when several questions wait)."
    ),
    job: Optional[str] = typer.Option(
        None, "--job", help="The question's job (needed when a node has several)."
    ),
    choice: Optional[str] = typer.Option(None, "--choice", help="One of the question's options."),
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
    """Answer a question the run is waiting on, as the canvas does. `prime pipelines get` lists
    them, each with the command that answers it."""
    card = _card(run_id, (_run(run_id).get("state") or {}).get("cards") or [], node, job)
    kind = _kind(card)
    answer: dict[str, Any] = {"note": note}
    if kind != "choice" and choice is not None:
        _fail(ValueError(f"this is a {kind}; it has no options to choose from"))
    if kind != "review" and (keep or drop or keep_all):
        _fail(ValueError(f"this is a {kind}; it has no items to keep"))
    if kind == "choice":
        if choice not in card["options"]:
            _fail(ValueError(f"choose one of: {', '.join(card['options'])}"))
        answer["choice"] = choice
    elif kind == "review":
        if sum((bool(keep), bool(drop), keep_all)) != 1:
            _fail(ValueError("say what to keep with one of --keep, --drop or --keep-all"))
        unknown = [item for item in (*keep, *drop) if item not in card["items"]]
        if unknown:
            _fail(ValueError(f"not an item of this review: {', '.join(unknown)}"))
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
        f"if it can't, prime pipelines get {escape(shlex.quote(run_id))} says why."
    )


@app.command("outputs")
def outputs(
    run_id: str = typer.Argument(..., help="The run's id."),
    directory: Path = typer.Option(Path("."), "--dir", "-d", help="Where to save them."),
) -> None:
    """Download what a run's heads made: its report, its taskset package."""
    client = APIClient()
    listed = _run(run_id).get("outputs") or []
    if not listed:
        console.print("This run has no outputs yet.")
        return
    for out in listed:
        parts = (str(out["node"]), str(out["name"]))
        if any(
            not part or part in (".", "..") or "/" in part or "\\" in part or ":" in part
            for part in parts
        ):
            _fail(APIError(f"refusing an output named {out['node']}/{out['name']}"))
        target = directory / parts[0] / parts[1]
        partial: Optional[Path] = None
        url = (
            f"{client.base_url}/api/v1/pipelines/runs/{quote(run_id, safe='')}"
            f"/outputs/{quote(str(out['node']), safe='')}/{quote(str(out['name']), safe='')}"
        )
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
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
            partial = None
        except (httpx.HTTPError, OSError) as error:
            _fail(APIError(f"downloading {out['name']}: {error}"))
        finally:
            # Also on Ctrl-C, which is no Exception: a partial file is never left behind.
            if partial is not None:
                partial.unlink(missing_ok=True)
        console.print(f"Saved {escape(str(target))}")


@app.command("stop")
def stop(
    run_id: str = typer.Argument(..., help="The run's id."),
    drain: bool = typer.Option(
        False,
        "--drain",
        help="Let running work finish: the run starts nothing new, then ends.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation."),
) -> None:
    """Stop a run. Its runner and every sandbox it started end at once; with --drain, once what's
    running has finished."""
    question = (
        f"Wind down run {run_id}? It starts no new work and ends once what's running finishes."
        if drain
        else f"Stop run {run_id}? Its runner and every sandbox it started end now."
    )
    if not confirm_or_skip(question, yes):
        console.print("Cancelled.")
        raise typer.Exit(0)
    if drain:
        seq = _control(run_id, {"type": "drain"})
        console.print(f"Sent (control {seq}). The run ends once what's running finishes.")
        return
    try:
        stopped = APIClient().post(f"/pipelines/runs/{quote(run_id, safe='')}/stop")
    except APIError as error:
        _fail(error)
    _summary(stopped)


_LIVE_RUN = re.compile(r"Run (\S+) of this pipeline is still running")


@app.command("delete")
def delete(
    pipeline: str = typer.Argument(..., help="The pipeline's id."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation."),
) -> None:
    """Delete a pipeline with its versions, its runs and their outputs. Stop its live runs
    first."""
    client = APIClient()
    path = f"/pipelines/{quote(pipeline, safe='')}"
    try:
        found = client.get(path)
    except APIError as error:
        _fail(error)
    runs = found["runs"]
    goes = (
        "It has no runs."
        if not runs
        else "Its 1 run and that run's outputs are deleted with it."
        if runs == 1
        else f"Its {runs} runs and their outputs are deleted with it."
    )
    question = f'Delete pipeline "{found["name"]}" ({found["id"]})? {goes} This can\'t be undone.'
    if not confirm_or_skip(question, yes):
        console.print("Cancelled.")
        raise typer.Exit(0)
    try:
        # Longer than a read: every output file is deleted before the pipeline is.
        gone = client.request("DELETE", path, timeout=120)
    except APIError as error:
        if live := _LIVE_RUN.search(str(error)):
            stop = shlex.join(["prime", "pipelines", "stop", live.group(1)])
            _fail(APIError(f"{error} Stop it with: {stop}"))
        _fail(error)
    console.print(
        f"Deleted pipeline [bold]{escape(gone['name'])}[/bold] ({escape(gone['id'])}) "
        f"with {gone['runs']} run{'' if gone['runs'] == 1 else 's'}."
    )
