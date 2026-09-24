import json
import shlex
import tarfile
import tempfile
import time
import tomllib
import uuid
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Optional

import typer
from prime_evals import EvalsAPIError, EvalsClient
from prime_sandboxes import APIClient as SandboxAPIClient
from prime_sandboxes import CreateSandboxRequest, SandboxClient
from rich.live import Live
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from ..client import APIClient, APIError
from ..core import Config
from ..utils import (
    DefaultCommandGroup,
    PlainTyper,
    get_console,
    json_output_help,
    output_data_as_json,
)
from .teams import fetch_team_members

console = get_console()

LIST_EVALS_JSON_HELP = json_output_help(
    ".evaluations[] = {evaluation_id|id, environment_names[], model_name, status, user_id, "
    "metadata}",
    ".total = number",
)

EVAL_INFO_JSON_HELP = json_output_help(
    ". = evaluation object from Prime Evals",
    "Common keys: .evaluation_id? | .id, .environment_names[]?, .model_name?, .status?, .metrics?",
    ".samples = the requested page of samples (.samples[]?, .total?, .page?, .limit?)",
)

EVAL_TABLE_MAX_TEXT_WIDTH = 30

PRIME_RL_REPO = "https://github.com/PrimeIntellect-ai/prime-rl.git"
EVAL_SANDBOX_IMAGE = "python:3.12-slim"
EVAL_SANDBOX_WORKDIR = "/workspace"
EVAL_SETUP_TIMEOUT_SECONDS = 45 * 60
# The sandbox has no lifetime; the eval decides when it ends.
EVAL_NO_DEADLINE_SECONDS = 10**9
EVAL_LOCAL_ENV_ARCHIVE_SKIP = {".git", ".venv", "__pycache__", "outputs", "dist", ".prime"}
EVAL_LINK_TIMEOUT_SECONDS = 180
# Where uploaded `@ file.toml` configs land inside the prime-rl checkout.
EVAL_CONFIG_DIR = "launch"
EVAL_POLL_SECONDS = 3
# The monitor logs one such line per source once its platform evaluation exists.
EVAL_LINK_GREP = (
    "grep -ho 'evaluation - https://[^ ]*' outputs/*/logs/latest/eval.log 2>/dev/null"
    " | cut -d' ' -f3 | sort -u"
)
# Runs inside the sandbox after the eval so nothing is left behind when the launcher
# has already exited; the API key is in the job's environment.
EVAL_SELF_DELETE = (
    "python3 -c 'import os, urllib.request as u; "
    'u.urlopen(u.Request("{base}/api/v1/sandbox/{sandbox_id}", method="DELETE", '
    'headers={{"Authorization": "Bearer " + os.environ["PRIME_API_KEY"]}}))\''
)
# Sandboxes only run Docker Hub images, so start from python:3.12-slim and add git
# and uv. Submodules are pinned to SSH URLs; the sandbox has no GitHub key, so route
# them over HTTPS. Exported (not `git config`) so `git submodule--helper clone` sees it.
EVAL_SETUP_SCRIPT = """
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
mkdir -p {workdir} && cd {workdir}
apt-get update -qq && apt-get install -y -qq --no-install-recommends git > /dev/null
pip install --quiet uv
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0="url.https://github.com/.insteadOf"
export GIT_CONFIG_VALUE_0="git@github.com:"
git clone --quiet {repo} prime-rl
cd prime-rl
git checkout --quiet {ref}
git submodule update --init --recursive --depth 1 --quiet
uv sync --quiet
mkdir -p {config_dir}
"""


class DefaultGroup(DefaultCommandGroup):
    def format_usage(self, ctx, formatter):
        formatter.write_usage(
            ctx.command_path,
            "[OPTIONS] ENVIRONMENT [ARGS]... | COMMAND [ARGS]...",
        )


subcommands_app = PlainTyper()


def handle_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except typer.Exit:
            raise
        except EvalsAPIError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Unexpected error:[/red] {e}")
            raise typer.Exit(1)

    return wrapper


def _team_user_names(client: APIClient, config: Config) -> dict[str, str]:
    """Map user ids to display names: team members in a team context, else just you."""
    if config.team_id:
        try:
            members = fetch_team_members(client, config.team_id)
        except APIError:
            return {}
        return {
            str(m.get("userId")): m.get("userName") or m.get("userEmail") or str(m.get("userId"))
            for m in members
        }
    if config.user_id:
        return {config.user_id: config.user_name or "you"}
    return {}


@subcommands_app.command("list", epilog=LIST_EVALS_JSON_HELP)
@handle_errors
def list_evals(
    num: int = typer.Option(20, "--num", "-n", help="Items per page"),
    page: int = typer.Option(1, "--page", "-p", help="Page number"),
    env: Optional[str] = typer.Option(
        None,
        "--env",
        "--env-name",
        "-e",
        help="Filter by environment (e.g., 'gsm8k' or 'owner/gsm8k')",
    ),
    as_json: bool = typer.Option(False, "--json", help="Output JSON instead of a table"),
) -> None:
    """List hosted evaluations"""

    if num < 1 or page < 1:
        console.print("[red]Error:[/red] --num and --page must be at least 1")
        raise typer.Exit(1)

    try:
        api_client = APIClient()
        config = Config()
        client = EvalsClient(api_client)

        skip = (page - 1) * num
        data = client.list_evaluations(
            env_name=env,
            team_id=config.team_id,
            skip=skip,
            limit=num,
        )

        if as_json:
            output_data_as_json(data, console)
            return

        evals = data.get("evaluations", [])

        if not evals:
            if page > 1:
                console.print("[yellow]No more results.[/yellow]")
            else:
                console.print("[yellow]No evaluations found.[/yellow]")
            return

        user_names = _team_user_names(api_client, config)
        env_slugs = _environment_slugs(api_client, evals)

        table = Table(expand=True)
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Environment", style="blue", no_wrap=True, overflow="ellipsis", ratio=1)
        table.add_column("Model", style="magenta", no_wrap=True, overflow="ellipsis", ratio=1)
        table.add_column("Status", style="yellow", no_wrap=True)
        table.add_column("Type", style="green", justify="center", no_wrap=True)
        table.add_column("User", style="dim", no_wrap=True, overflow="ellipsis")

        for e in evals:
            eval_id = str(e.get("evaluation_id", e.get("id", "")))
            user_id = str(e.get("user_id") or "")
            user = user_names.get(user_id, user_id or "-")

            env_name = "-"
            environment_names = e.get("environment_names", [])
            if environment_names and len(environment_names) > 0:
                env_name = environment_names[0]
                env_ids = e.get("environment_ids") or []
                if env_ids:
                    env_name = env_slugs.get(env_ids[0], env_name)

            table.add_row(
                eval_id if eval_id else "",
                str(env_name),
                str(e.get("model_name", "")),
                str(e.get("status", "")),
                "HOSTED" if e.get("is_hosted") else "LOCAL",
                user,
            )

        console.print(table)
        total = data.get("total", 0)
        if total > page * num:
            console.print(f"\n[dim]Page {page} - use --page {page + 1} for the next[/dim]")
        else:
            console.print(f"\n[dim]Total: {total} evaluation(s)[/dim]")

    except EvalsAPIError as e:
        console.print(f"[red]API Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print(
            "[yellow]Response may contain invalid data. Try --json to see raw response.[/yellow]"
        )
        raise typer.Exit(1)


@subcommands_app.command("info", epilog=EVAL_INFO_JSON_HELP)
@handle_errors
def info_eval(
    eval_id: str = typer.Argument(..., help="Evaluation ID (from `prime eval list`)"),
    page: int = typer.Option(1, "--page", "-p", help="Samples page number"),
    num: int = typer.Option(20, "--num", "-n", help="Samples per page"),
    as_json: bool = typer.Option(False, "--json", help="Output JSON instead of a table"),
) -> None:
    """Show a hosted evaluation and its samples"""
    client = EvalsClient(APIClient())
    evaluation = client.get_evaluation(eval_id)
    samples = client.get_samples(eval_id, page=page, limit=num)
    if as_json:
        output_data_as_json({**evaluation, "samples": samples}, console)
        return
    console.print("[bold cyan]Evaluation[/bold cyan]")
    console.print(Syntax(json.dumps(evaluation, indent=2), "json", theme="monokai"))
    console.print(f"\n[bold cyan]Samples[/bold cyan] [dim](page {page}, {num} per page)[/dim]")
    console.print(Syntax(json.dumps(samples, indent=2), "json", theme="monokai"))


app = PlainTyper(
    cls=DefaultGroup,
    help=(
        "Manage hosted evaluations (run, list, info)\n\n"
        "By default, 'prime eval <environment>' runs 'prime eval run <environment>'."
    ),
    no_args_is_help=True,
)

app.add_typer(subcommands_app, name="")


app = PlainTyper(
    cls=DefaultGroup,
    help=(
        "Manage hosted evaluations (run, list, info)\n\n"
        "By default, 'prime eval <environment>' runs 'prime eval run <environment>'."
    ),
    no_args_is_help=True,
)

app.add_typer(subcommands_app, name="")


class _StepFailed(Exception):
    """A step's command exited non-zero; the message is what to show the user."""


class _StepView:
    """Spinner + label + running timer, redrawn by `Live`."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.started = time.monotonic()
        self._spinner = Spinner("dots", style="blue")

    def elapsed(self) -> str:
        return f"{time.monotonic() - self.started:.0f}s"

    def __rich_console__(self, console, options):
        self._spinner.update(text=Text(f"{self.label} ({self.elapsed()})", style="bold blue"))
        yield self._spinner


@contextmanager
def _step(label: str):
    """Show `label` with a spinner and timer, then replace it with ✓ or ✗ and the time."""
    view = _StepView(label)
    live = Live(view, console=console, transient=True, refresh_per_second=8)
    live.start()
    try:
        yield view
    except _StepFailed as exc:
        live.stop()
        console.print(f"[red]✗[/red] {view.label} [dim]({view.elapsed()})[/dim]")
        console.print(str(exc), markup=False)
        raise typer.Exit(1) from exc
    except BaseException:
        live.stop()
        console.print(f"[red]✗[/red] {view.label} [dim]({view.elapsed()})[/dim]")
        raise
    live.stop()
    console.print(f"[green]✓[/green] {view.label} [dim]({view.elapsed()})[/dim]")


def _check(job, what: str) -> None:
    if job.exit_code == 0:
        return
    if not hasattr(job, "stdout"):
        raise _StepFailed(f"{what} failed (exit {job.exit_code})")
    output = "\n".join(part.strip() for part in (job.stdout, job.stderr) if part and part.strip())
    tail = "\n".join(output.splitlines()[-15:])
    raise _StepFailed(f"{what} failed (exit {job.exit_code})\n{tail}")


def _hub_env_install_command(slug: str) -> tuple[str, str]:
    """Resolve an `owner/name` Hub slug to its install command and taskset id.

    `--no-config` keeps prime-rl's `[tool.uv]` settings (an `exclude-newer` cooldown
    that rejects Hub wheels, which carry no upload date) out of the install."""
    owner, name = slug.split("/", 1)
    try:
        response = APIClient(require_auth=False).get(f"/environmentshub/{owner}/{name}/@latest")
    except APIError as exc:
        console.print(f"[red]Error:[/red] could not resolve {slug} on the Environments Hub: {exc}")
        raise typer.Exit(1) from exc
    details = response.get("data", response)
    index_url = details.get("install_index_url") or details.get("simple_index_url")
    if not index_url:
        console.print(f"[red]Error:[/red] {slug} has no install index (private environment?)")
        console.print(f"[yellow]Pull it with `prime env pull {slug}` and pass --env-path.[/yellow]")
        raise typer.Exit(1)
    package = name.replace("-", "_").lower()
    return f"uv pip install --no-config {package} --extra-index-url {index_url}", name


def _upload_local_env(sandboxes: SandboxClient, sandbox_id: str, env_path: Path) -> None:
    """Ship a local environment package into the sandbox at /workspace/envs/<name>."""
    env_path = env_path.resolve()

    def skip(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(info.name).parts
        return None if any(part in EVAL_LOCAL_ENV_ARCHIVE_SKIP for part in parts) else info

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as handle:
        archive = Path(handle.name)
    try:
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(env_path, arcname=env_path.name, filter=skip)
        remote_archive = f"{EVAL_SANDBOX_WORKDIR}/{env_path.name}.tar.gz"
        sandboxes.upload_file(sandbox_id, remote_archive, str(archive))
    finally:
        archive.unlink()
    result = sandboxes.execute_command(
        sandbox_id,
        f"mkdir -p envs && tar -xzf {shlex.quote(remote_archive)} -C envs",
        working_dir=EVAL_SANDBOX_WORKDIR,
    )
    if result.exit_code != 0:
        console.print(f"[red]Error:[/red] unpacking {env_path.name} failed: {result.stderr}")
        raise typer.Exit(1)


def _wait_for_links(sandboxes: SandboxClient, sandbox_id: str, job, workdir: str) -> list[str]:
    """Poll the eval log until the monitor has logged every source's platform link."""
    deadline = time.monotonic() + EVAL_LINK_TIMEOUT_SECONDS
    links: list[str] = []
    while time.monotonic() < deadline:
        status = sandboxes.get_background_job_status(sandbox_id, job)
        if status.completed:
            _check(sandboxes.get_background_job(sandbox_id, job), "Eval")
            break
        found = sandboxes.execute_command(sandbox_id, EVAL_LINK_GREP, working_dir=workdir)
        links = [line.strip() for line in (found.stdout or "").splitlines() if line.strip()]
        if links:
            break
        time.sleep(EVAL_POLL_SECONDS)
    return links


def _extract_config_files(eval_args: list[str]) -> list[Path]:
    """Every `@ path.toml` in the command (leading or after a `--section` flag) is a local
    file: collect it and point the argument at its sandbox copy."""
    files: list[Path] = []
    for i, arg in enumerate(eval_args[:-1]):
        if arg != "@":
            continue
        path = Path(eval_args[i + 1])
        if not path.is_file():
            console.print(f"[red]Error:[/red] config not found: {path}")
            raise typer.Exit(1)
        if any(f.name == path.name and f.resolve() != path.resolve() for f in files):
            console.print(f"[red]Error:[/red] two configs share the name {path.name}")
            raise typer.Exit(1)
        files.append(path)
        eval_args[i + 1] = f"{EVAL_CONFIG_DIR}/{path.name}"
    return files


def _validate_eval_config(config_files: list[Path], installed_names: set[str]) -> None:
    """Parse the TOMLs and make sure every source's taskset will be installed."""
    tasksets: list[str] = []
    for config_file in config_files:
        try:
            with open(config_file, "rb") as handle:
                data = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            console.print(f"[red]Error:[/red] {config_file} is not valid TOML: {exc}")
            raise typer.Exit(1) from exc
        for block in [data, *data.get("source", [])]:
            taskset = ((block.get("env") or {}).get("taskset") or {}).get("id")
            if taskset:
                tasksets.append(taskset)
    if not tasksets:
        console.print("[red]Error:[/red] the config names no env.taskset.id")
        raise typer.Exit(1)
    missing = sorted(t for t in tasksets if t.replace("_", "-") not in installed_names)
    if missing:
        console.print(
            "[red]Error:[/red] no environment installed for "
            + ", ".join(f"`{t}`" for t in missing)
            + " - pass --install owner/<name> for each (or --env-path for a local package)"
        )
        raise typer.Exit(1)


def _environment_slugs(client: APIClient, evals: list[dict[str, Any]]) -> dict[str, str]:
    """Map environment ids to `owner/name`. Evaluations carry ids and bare names; the
    Hub has no lookup by id, so search by name and match the id."""
    slugs: dict[str, str] = {}
    for e in evals:
        for env_id, name in zip(e.get("environment_ids") or [], e.get("environment_names") or []):
            if env_id in slugs:
                continue
            response = client.get("/environmentshub/", params={"search": name, "limit": 50})
            for entry in response.get("data", []):
                owner = (entry.get("owner") or {}).get("name")
                if entry.get("id") == env_id and owner:
                    slugs[env_id] = f"{owner}/{entry.get('name', name)}"
    return slugs


@app.command(
    "run",
    help="Run a hosted evaluation",
    no_args_is_help=True,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def run_eval_cmd(
    ctx: typer.Context,
    environment: str = typer.Argument(
        ...,
        help=(
            "Environments Hub slug (owner/name), a taskset id with --env-path, "
            "or `@ eval.toml`; the rest is passed to `uv run eval`"
        ),
    ),
    env_path: Optional[Path] = typer.Option(
        None, "--env-path", help="Local environment package to upload and install first"
    ),
    install: Optional[list[str]] = typer.Option(
        None, "--install", help="Extra Hub environments (owner/name) to install (repeatable)"
    ),
    ref: str = typer.Option("main", "--ref", help="prime-rl git ref to check out"),
    image: str = typer.Option(EVAL_SANDBOX_IMAGE, "--image", help="Sandbox docker image"),
    cpu_cores: float = typer.Option(4.0, "--cpu", help="Sandbox CPU cores"),
    memory_gb: float = typer.Option(8.0, "--memory", help="Sandbox memory in GB"),
    disk_size_gb: float = typer.Option(20.0, "--disk", help="Sandbox disk in GB"),
    env_var: Optional[list[str]] = typer.Option(
        None, "--env-var", help="Extra KEY=VALUE for the eval process (repeatable)"
    ),
    wait: bool = typer.Option(False, "--wait", help="Stay attached until the eval exits"),
    keep: bool = typer.Option(False, "--keep", help="Keep the sandbox after the eval exits"),
) -> None:
    """Every argument `prime eval run` does not own is proxied verbatim to `uv run eval`
    (see `uv run eval -h` in prime-rl). `--monitors.prime` is added unless given, so each
    finished source lands as an evaluation on the platform. The command is a launcher:
    it returns once the platform links exist, and the sandbox deletes itself when the
    eval ends (unless --wait keeps the CLI attached, or --keep)."""
    eval_args = [environment, *ctx.args]
    install_commands = [_hub_env_install_command(slug)[0] for slug in install or []]
    if env_path is not None:
        if not (env_path / "pyproject.toml").is_file():
            console.print(f"[red]Error:[/red] {env_path} has no pyproject.toml")
            raise typer.Exit(1)
        install_commands.append(
            f"uv pip install --no-config -e {EVAL_SANDBOX_WORKDIR}/envs/{env_path.resolve().name}"
        )
    elif "/" in environment:
        install_command, environment = _hub_env_install_command(environment)
        install_commands.append(install_command)
        eval_args[0] = environment
    elif environment != "@" or not install_commands:
        console.print(
            "[red]Error:[/red] prime-rl ships no environments; pass an Environments Hub slug "
            "(owner/name), --env-path for a local package, or `@ eval.toml` with --install"
        )
        raise typer.Exit(1)
    config_files = _extract_config_files(eval_args)
    if environment == "@":
        installed_names = {slug.split("/", 1)[1] for slug in install or []}
        if env_path is not None:
            installed_names.add(env_path.resolve().name.replace("_", "-"))
        _validate_eval_config(config_files, installed_names)
    if not any(arg.startswith("--monitors.prime") for arg in eval_args):
        eval_args.append("--monitors.prime")

    config = Config()
    # PRIME_RUNS_IS_HOSTED marks the evaluations prime-rl's monitor creates as hosted.
    env_vars = {"PRIME_API_KEY": config.api_key, "PRIME_RUNS_IS_HOSTED": "1"}
    if config.team_id:
        env_vars["PRIME_TEAM_ID"] = config.team_id
    for pair in env_var or []:
        if "=" not in pair:
            console.print(f"[red]Error:[/red] --env-var expects KEY=VALUE, got {pair!r}")
            raise typer.Exit(1)
        key, value = pair.split("=", 1)
        env_vars[key] = value

    sandboxes = SandboxClient(SandboxAPIClient())
    with _step("Booting sandbox") as step:
        sandbox = sandboxes.create(
            CreateSandboxRequest(
                name=f"prime-eval-{uuid.uuid4().hex[:8]}",
                docker_image=image,
                cpu_cores=cpu_cores,
                memory_gb=memory_gb,
                disk_size_gb=disk_size_gb,
                timeout_minutes=-1,
                labels=["prime-eval"],
            )
        )
        step.label = f"Booting sandbox {sandbox.id} ({image})"
        sandboxes.wait_for_creation(sandbox.id)
    try:
        with _step(f"Installing prime-rl@{ref}"):
            setup_script = EVAL_SETUP_SCRIPT.format(
                workdir=EVAL_SANDBOX_WORKDIR,
                repo=PRIME_RL_REPO,
                ref=shlex.quote(ref),
                config_dir=EVAL_CONFIG_DIR,
            )
            # Background jobs run under `sh`; the script needs bash for `pipefail`.
            setup = sandboxes.run_background_job(
                sandbox.id,
                f"bash -c {shlex.quote(setup_script)}",
                timeout=EVAL_SETUP_TIMEOUT_SECONDS,
            )
            _check(setup, "Installing prime-rl")

        with _step("Installing environments"):
            if env_path is not None:
                _upload_local_env(sandboxes, sandbox.id, env_path)
            installed = sandboxes.run_background_job(
                sandbox.id,
                " && ".join(install_commands),
                timeout=EVAL_SETUP_TIMEOUT_SECONDS,
                working_dir=f"{EVAL_SANDBOX_WORKDIR}/prime-rl",
            )
            _check(installed, "Installing environments")
            for config_file in config_files:
                sandboxes.upload_file(
                    sandbox.id,
                    f"{EVAL_SANDBOX_WORKDIR}/prime-rl/{EVAL_CONFIG_DIR}/{config_file.name}",
                    str(config_file),
                )

        command = shlex.join(["uv", "run", "eval", *eval_args])
        workdir = f"{EVAL_SANDBOX_WORKDIR}/prime-rl"
        with _step("Validating config"):
            dry_run = sandboxes.run_background_job(
                sandbox.id,
                f"{command} --dry-run",
                timeout=EVAL_SETUP_TIMEOUT_SECONDS,
                working_dir=workdir,
                env=env_vars,
            )
            _check(dry_run, "Config validation")
        if wait or keep:
            job_script = command
        else:
            cleanup = EVAL_SELF_DELETE.format(
                base=config.base_url.rstrip("/"), sandbox_id=sandbox.id
            )
            job_script = f"{command}; code=$?; {cleanup}; exit $code"
        with _step(f"Launching {command}"):
            job = sandboxes.start_background_job(
                sandbox.id, job_script, working_dir=workdir, env=env_vars
            )
            links = _wait_for_links(sandboxes, sandbox.id, job, workdir)
        for link in links:
            console.print(f"Evaluation: [link={link}]{link}[/link]")
        if not wait:
            if keep:
                console.print(f"[dim]Sandbox {sandbox.id} keeps running after the eval[/dim]")
            return
        with _step("Running eval"):
            while True:
                status = sandboxes.get_background_job_status(sandbox.id, job)
                if status.completed:
                    _check(sandboxes.get_background_job(sandbox.id, job), "Eval")
                    break
                time.sleep(EVAL_POLL_SECONDS)
    except BaseException:
        # Setup failed or we were interrupted before handing the sandbox to the eval.
        if not keep:
            sandboxes.delete(sandbox.id)
        raise
    if wait and not keep:
        sandboxes.delete(sandbox.id)
