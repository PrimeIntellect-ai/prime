"""`prime pipelines`: run a pipeline (a package on the Environments Hub) and fetch what it made."""

import json
import os
import re
import shlex
import tempfile
import tomllib
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
    help="Run a pipeline (a package on the Environments Hub) on the platform, and fetch what "
    "it made.",
    no_args_is_help=True,
)
console = get_console()

_RUN_FIELDS = (
    "id, pipeline_id, version, package?, status, error?, stopped, "
    "outputs[{node, name, bytes}], network{allow[]}, evaluations[], trainings[], deployments[], "
    "spend_limit?, created_at, started_at?, completed_at?"
)
RUN_JSON_HELP = json_output_help(f". = {{{_RUN_FIELDS}}}")
LIST_JSON_HELP = json_output_help(
    ".pipelines[] = {id, name, team_id?, version?, package?, updated_at, runs, latest_run?}",
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
# The most items the platform's lists give in one page.
MAX_PER_PAGE = 100
# A pipeline's or a run's id, as the platform's routes take it.
_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
GET_JSON_HELP = json_output_help(
    f". = {{{_RUN_FIELDS}, spent?, state?}}",
    ".state.jobs = {<status>: number} — how many jobs are ready, waiting, held or terminal (done)",
    ".state.stages.<stage> = {<status>: number}",
    ".state.waiting_on_people[] = {job, stage, reason, by?} — what waits on you",
    ".state.failures[] = {job, stage, reason, error?} — jobs an error holds",
    ".state.skipped_controls[] = {seq, error} — replies the run couldn't apply",
    ".state.graph = the pipeline's stages and how they connect, as its package declares them",
)


def _fail(error: Exception) -> NoReturn:
    console.print(f"[red]Error:[/red] {escape(str(error))}")
    raise typer.Exit(1)


def _summary(run: dict[str, Any]) -> None:
    stopped = " (stopped)" if run.get("stopped") else ""
    console.print(f"Run [bold]{run['id']}[/bold]: {run['status']}{stopped}")
    if run.get("pipeline_id"):
        console.print(f"Pipeline: {run['pipeline_id']}")
    if run.get("package"):
        console.print(f"Package: {escape(run['package'])}")
    console.print(f"Version: {run['version'][:12]}")
    if run.get("error") and not run.get("stopped"):
        console.print(f"[red]{escape(run['error'])}[/red]")


def _run(run_id: str) -> dict[str, Any]:
    try:
        return APIClient().get(f"/pipelines/runs/{quote(run_id, safe='')}")
    except APIError as error:
        _fail(error)


# A package on the Hub: `owner/name`, or `owner/name@version` to pin a release.
_PACKAGE = re.compile(r"^[^/@\s]+/[^/@\s]+(?:@\S+)?$")
_LAUNCH_KEYS = ("config", "capabilities")


def _launch(path: Path) -> dict[str, Any]:
    """A launch file's `config` and `capabilities`: JSON if it is named `.json`, TOML otherwise."""
    try:
        # UTF-8 whatever the locale says; utf-8-sig also takes a file saved with a BOM.
        text = path.read_text(encoding="utf-8-sig")
    except OSError as error:
        _fail(ValueError(f"can't read {path}: {error.strerror or error}"))
    except ValueError as error:
        _fail(ValueError(f"{path} is not UTF-8 text: {error}"))
    try:
        data = json.loads(text) if path.suffix.lower() == ".json" else tomllib.loads(text)
    except ValueError as error:
        kind = "JSON" if path.suffix.lower() == ".json" else "TOML"
        _fail(ValueError(f"{path} is not {kind}: {error}"))
    if not isinstance(data, dict) or set(data) - set(_LAUNCH_KEYS):
        _fail(ValueError(f"{path}: a launch file has two tables, config and capabilities"))
    for key, value in data.items():
        if not isinstance(value, dict):
            _fail(ValueError(f"{path}: {key} is a table"))
    return data


def _start(body: dict[str, Any], agreed: bool, as_json: bool) -> dict[str, Any]:
    """Start the run. A team run needs its starter's agreement to run the package's code and to
    what its launch declares: the platform's refusal lists it, and this asks, once."""
    try:
        sent = {**body, "run_package": agreed}
        return APIClient().post("/pipelines/runs", json=sent, timeout=120)
    except APIError as error:
        if "with pipeline_id" in str(error):
            _fail(APIError(f"{error} Pass one of these ids with --pipeline."))
        if agreed or "run_package" not in str(error):
            _fail(error)
        asked = error
    # To stderr with --output json, so stdout stays the run alone.
    out = get_console(stderr=True) if as_json else console
    out.print(f"[yellow]{escape(str(asked).removeprefix('HTTP 422: '))}[/yellow]")
    out.print("[dim]To agree without being asked, pass --run-package.[/dim]")
    if not confirm_or_skip("Agree, and start the run?", False, err=as_json):
        out.print("Cancelled.")
        raise typer.Exit(0)
    return _start(body, True, as_json)


@app.command("run", epilog=RUN_JSON_HELP)
def run(
    target: str = typer.Argument(
        ...,
        metavar="PACKAGE | PIPELINE",
        help="A pipeline's package on the Environments Hub, to save and run: owner/name runs its "
        "latest release, owner/name@version that one. Or a saved pipeline's id, to run its latest "
        "version.",
    ),
    launch: Optional[Path] = typer.Option(
        None,
        "--launch",
        help="With a package: a TOML or JSON file with two tables. `config` is the package's own "
        "settings for this run. `capabilities` declares what the run may do besides run the "
        "package: publish, evaluate, train, metrics, traces and spend_limit.",
    ),
    team: Optional[str] = typer.Option(
        None,
        "--team",
        "-t",
        help="With a package: run as this team (default: your selected team). A saved "
        "pipeline runs as its own account.",
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        help="With a package: the pipeline's name (default: the package's name; with "
        "--pipeline, the name it has).",
    ),
    pipeline: Optional[str] = typer.Option(
        None,
        "--pipeline",
        help="With a package: save the launch as a new version of this pipeline (default: the "
        "pipeline that has the name, or a new one).",
    ),
    spend_limit: Optional[float] = typer.Option(
        None,
        "--spend-limit",
        help="With a package: dollars past which the run starts no new training, deployment or "
        "evaluation. Sets `spend_limit` over the launch file's.",
    ),
    secret: List[str] = typer.Option(
        [],
        "--secret",
        help="A secret the package reads from its environment, taken from yours by name. Repeat "
        "for more.",
    ),
    allow_host: List[str] = typer.Option(
        [],
        "--allow-host",
        help="A host this run's machine may reach besides the platform's, such as api.github.com. "
        "Repeat for more.",
    ),
    run_package: bool = typer.Option(
        False,
        "--run-package",
        help="On a team run: agree, without being asked, to run the package's code and to what "
        "its launch declares. The code, which any member of the team may have saved or pushed, "
        "runs on the run's machine, where it can use your key and the secrets you pass, and does "
        "what the launch declares as you.",
    ),
    output: str = typer.Option("table", "--output", "-o", help="table or json"),
) -> None:
    """Start a run: of a package, saved with its launch as a version of the pipeline that has the
    name; or of a saved pipeline, by its id. A team run asks you to agree to what it may do."""
    validate_output_format(output, console)
    missing = [name for name in secret if name not in os.environ]
    if missing:
        _fail(ValueError(f"set {', '.join(missing)} in your environment"))
    body: dict[str, Any] = {
        "secrets": {name: os.environ[name] for name in secret},
        "network": {"allow": allow_host},
    }
    if _PACKAGE.match(target):
        declared = _launch(launch) if launch is not None else {}
        capabilities = dict(declared.get("capabilities") or {})
        if spend_limit is not None:
            capabilities["spend_limit"] = spend_limit
        if pipeline is not None and name is None:
            # A save names the pipeline: without this it would take the package's name.
            try:
                name = APIClient().get(f"/pipelines/{quote(pipeline, safe='')}")["name"]
            except APIError as error:
                _fail(error)
        body |= {
            "package": target,
            "config": declared.get("config") or {},
            "capabilities": capabilities,
            "name": name,
            "team_id": team or Config().team_id,
            "pipeline_id": pipeline,
        }
    elif not _ID.match(target):
        _fail(
            ValueError(
                f"{target} is neither a package (owner/name, or owner/name@version) nor a "
                "saved pipeline's id"
            )
        )
    elif any(option is not None for option in (launch, name, pipeline, spend_limit)):
        _fail(
            ValueError(
                "--launch, --name, --pipeline and --spend-limit go with a package; a saved "
                "pipeline runs the launch it was saved with"
            )
        )
    else:
        # The pipeline's latest version runs, as the pipeline's own account.
        body["pipeline_id"] = target
    started = _start(body, run_package, output == "json")
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
    if not 1 <= num <= MAX_PER_PAGE or page < 1:
        _fail(ValueError(f"--num must be from 1 to {MAX_PER_PAGE}, and --page at least 1"))
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
    num: int = typer.Option(20, "--num", "-n", help=f"Items per page (at most {MAX_PER_PAGE})"),
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
    table = Table("id", "name", "package", "runs", "latest run", "status", "saved")
    for row in found["pipelines"]:
        latest = row.get("latest_run") or {}
        table.add_row(
            escape(row["id"]),
            escape(row["name"]),
            escape(row.get("package") or ""),
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
    num: int = typer.Option(20, "--num", "-n", help=f"Items per page (at most {MAX_PER_PAGE})"),
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
    table = Table("id", "pipeline", "package", "status", "version", "started", "outputs")
    for row in found["runs"]:
        table.add_row(
            escape(row["id"]),
            escape(row["pipeline_id"]),
            escape(row.get("package") or ""),
            _status(row),
            escape(row["version"][:12]),
            _when(row.get("started_at") or row["created_at"]),
            str(len(row.get("outputs") or [])),
        )
    console.print(table)
    _page(page, num, found["total"], len(found["runs"]), "runs")


# A job's statuses, in the order it goes through them; any other follows, by name.
_STATUSES = ("ready", "waiting", "held", "terminal")
# As the run's page names them.
_STATUS_LABELS = {"terminal": "done"}


def _waiting(run_id: str, held: dict[str, Any]) -> None:
    """A job the run holds for a person, and the command that answers it."""
    console.print(
        f"[yellow]Waiting on you[/yellow] at [bold]{escape(held['stage'])}[/bold] "
        f"(job {escape(held['job'])}): {escape(held.get('reason') or 'no reason given')}"
    )
    # Quoted for a shell: the command is there to be copied, and the names in it are the run's.
    where = shlex.join([run_id, "--job", held["job"]])
    console.print(f"  Answer: prime pipelines reply {escape(where)} --patch JSON")


@app.command("get", epilog=GET_JSON_HELP)
def get(
    run_id: str = typer.Argument(..., help="The run's id."),
    output: str = typer.Option("table", "--output", "-o", help="table or json"),
) -> None:
    """A run: its status, each stage's job counts, what waits on you and how to answer, what it
    started and spent, and its outputs."""
    validate_output_format(output, console)
    found = _run(run_id)
    if output == "json":
        output_data_as_json(found, console)
        return
    _summary(found)
    state = found.get("state") or {}
    if stages := state.get("stages"):
        seen = {status for counts in stages.values() for status in counts}
        statuses = [s for s in _STATUSES if s in seen] + sorted(seen - set(_STATUSES))
        table = Table("stage", *(_STATUS_LABELS.get(status, status) for status in statuses))
        for stage, counts in stages.items():
            table.add_row(escape(stage), *(str(counts.get(status, 0)) for status in statuses))
        console.print(table)
    for control in state.get("skipped_controls") or []:
        console.print(
            f"[red]Not applied[/red] (control {control['seq']}): {escape(control['error'])}"
        )
    for failure in state.get("failures") or []:
        why = " ".join(str(part) for part in (failure.get("error"), failure.get("reason")) if part)
        console.print(
            f"[red]Failed[/red] at [bold]{escape(failure['stage'])}[/bold] "
            f"(job {escape(failure['job'])}): {escape(why)}"
        )
    for held in state.get("waiting_on_people") or []:
        _waiting(run_id, held)
    for started, key in (("Evaluation", "evaluation_id"), ("Training", "training_id")):
        for row in found.get(f"{started.lower()}s") or []:
            of = " with ".join(str(row[k]) for k in ("taskset", "model") if row.get(k))
            console.print(
                f"{started} {escape(row[key])} ({escape(row['node'])})"
                + (f": {escape(of)}" if of else "")
            )
    for row in found.get("deployments") or []:
        console.print(f"Deployment {escape(row['deployment_id'])} ({escape(row['node'])})")
    if (limit := found.get("spend_limit")) is not None:
        spent = found.get("spent")
        console.print(
            f"Spend limit: ${limit:,.2f}" + (f", ${spent:,.2f} spent" if spent is not None else "")
        )
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


def _held(run_id: str, waiting: list[dict[str, Any]], job: Optional[str]) -> dict[str, Any]:
    """The one job waiting on a person; `job` names it when several wait."""
    matches = [held for held in waiting if job is None or held["job"] == job]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        _fail(ValueError(f"nothing waiting matches; see prime pipelines get {run_id}"))
    # Quoted for a shell, as the Answer line is: the names are the run's.
    listed = ", ".join(shlex.join(["--job", held["job"]]) for held in matches)
    _fail(ValueError(f"{len(matches)} jobs wait on you; name one with {listed}"))


@app.command("reply")
def reply(
    run_id: str = typer.Argument(..., help="The run's id."),
    job: Optional[str] = typer.Option(
        None, "--job", help="The job to answer (needed when several wait on you)."
    ),
    patch: str = typer.Option(
        ...,
        "--patch",
        help="The answer: a JSON object merged over the job's data before its stage runs again, "
        "such as '{\"approved\": true}'. What it holds is up to the pipeline's package.",
    ),
) -> None:
    """Answer a job the run holds for a person, as the run's page does. `prime pipelines get`
    lists them, each with the command that answers it."""
    try:
        answer = json.loads(patch)
    except ValueError as error:
        _fail(ValueError(f"--patch is not JSON: {error}"))
    if not isinstance(answer, dict):
        _fail(ValueError("--patch is a JSON object, such as '{\"approved\": true}'"))
    waiting = (_run(run_id).get("state") or {}).get("waiting_on_people") or []
    held = _held(run_id, waiting, job)
    seq = _control(run_id, {"type": "answer", "job": held["job"], "patch": answer})
    console.print(
        f"Sent (control {seq}). The run picks it up in a few seconds; "
        f"if it can't, prime pipelines get {escape(shlex.quote(run_id))} says why."
    )


@app.command("outputs")
def outputs(
    run_id: str = typer.Argument(..., help="The run's id."),
    directory: Path = typer.Option(Path("."), "--dir", "-d", help="Where to save them."),
) -> None:
    """Download the files a run wrote to its outputs, each into the folder it was written to."""
    client = APIClient()
    listed = _run(run_id).get("outputs") or []
    if not listed:
        console.print("This run has no outputs yet.")
        return
    for out in listed:
        parts = (str(out["node"]), str(out["name"]))
        # Nothing but dots and spaces is `.` or `..` to Windows, which drops trailing ones.
        if any(
            not part.rstrip(". ") or "/" in part or "\\" in part or ":" in part for part in parts
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
