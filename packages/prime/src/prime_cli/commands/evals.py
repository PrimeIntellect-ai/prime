import json
import shlex
import tarfile
import tempfile
import time
import uuid
from functools import wraps
from pathlib import Path
from typing import Optional

import typer
from prime_evals import EvalsAPIError, EvalsClient
from prime_sandboxes import APIClient as SandboxAPIClient
from prime_sandboxes import CreateSandboxRequest, SandboxClient
from rich.syntax import Syntax
from rich.table import Table

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
# Sandboxes only run Docker Hub images, so start from python:3.12-slim and add git
# and uv. Submodules are pinned to SSH URLs; the sandbox has no GitHub key, so route
# them over HTTPS. Exported (not `git config`) so `git submodule--helper clone` sees it.
EVAL_SETUP_SCRIPT = """
set -euo pipefail
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
uv sync --all-packages --quiet
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

        table = Table(expand=True)
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Environment", style="blue", no_wrap=True, overflow="ellipsis", ratio=1)
        table.add_column("Model", style="magenta", no_wrap=True, overflow="ellipsis", ratio=1)
        table.add_column("Status", style="yellow", no_wrap=True)
        table.add_column("User", style="dim", no_wrap=True, overflow="ellipsis")

        for e in evals:
            eval_id = str(e.get("evaluation_id", e.get("id", "")))
            user_id = str(e.get("user_id") or "")
            user = user_names.get(user_id, user_id or "-")

            env_name = "-"
            environment_names = e.get("environment_names", [])
            if environment_names and len(environment_names) > 0:
                env_name = environment_names[0]

            table.add_row(
                eval_id if eval_id else "",
                str(env_name),
                str(e.get("model_name", "")),
                str(e.get("status", "")),
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


def _hub_env_install_command(slug: str) -> tuple[str, str]:
    """Resolve an `owner/name` Hub slug to its install command and taskset id."""
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
    return f"uv pip install {package} --extra-index-url {index_url}", name


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
            "Taskset id bundled with prime-rl (gsm8k), a Hub slug (owner/name), "
            "or `@ eval.toml`; the rest is passed to `uv run eval`"
        ),
    ),
    env_path: Optional[Path] = typer.Option(
        None, "--env-path", help="Local environment package to upload and install first"
    ),
    ref: str = typer.Option("main", "--ref", help="prime-rl git ref to check out"),
    image: str = typer.Option(EVAL_SANDBOX_IMAGE, "--image", help="Sandbox docker image"),
    cpu_cores: float = typer.Option(4.0, "--cpu", help="Sandbox CPU cores"),
    memory_gb: float = typer.Option(8.0, "--memory", help="Sandbox memory in GB"),
    disk_size_gb: float = typer.Option(20.0, "--disk", help="Sandbox disk in GB"),
    env_var: Optional[list[str]] = typer.Option(
        None, "--env-var", help="Extra KEY=VALUE for the eval process (repeatable)"
    ),
    keep: bool = typer.Option(False, "--keep", help="Keep the sandbox after the eval exits"),
) -> None:
    """Every argument `prime eval run` does not own is proxied verbatim to `uv run eval`
    (see `uv run eval -h` in prime-rl). `--monitors.prime` is added unless given, so each
    finished source lands as an evaluation on the platform."""
    eval_args = [environment, *ctx.args]
    install_command = None
    if env_path is not None:
        if not (env_path / "pyproject.toml").is_file():
            console.print(f"[red]Error:[/red] {env_path} has no pyproject.toml")
            raise typer.Exit(1)
        install_command = f"uv pip install -e {EVAL_SANDBOX_WORKDIR}/envs/{env_path.resolve().name}"
    elif "/" in environment and environment != "@":
        install_command, environment = _hub_env_install_command(environment)
        eval_args[0] = environment
    config_file = None
    if environment == "@" and eval_args[1:]:
        config_file = Path(eval_args[1])
        if not config_file.is_file():
            console.print(f"[red]Error:[/red] config not found: {config_file}")
            raise typer.Exit(1)
        eval_args[1] = config_file.name
    if not any(arg.startswith("--monitors.prime") for arg in eval_args):
        eval_args.append("--monitors.prime")

    config = Config()
    env_vars = {"PRIME_API_KEY": config.api_key}
    if config.team_id:
        env_vars["PRIME_TEAM_ID"] = config.team_id
    for pair in env_var or []:
        if "=" not in pair:
            console.print(f"[red]Error:[/red] --env-var expects KEY=VALUE, got {pair!r}")
            raise typer.Exit(1)
        key, value = pair.split("=", 1)
        env_vars[key] = value

    sandboxes = SandboxClient(SandboxAPIClient())
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
    console.print(f"[dim]Sandbox {sandbox.id} ({image})[/dim]")
    try:
        started = time.monotonic()
        with console.status("[bold blue]Waiting for sandbox...", spinner="dots"):
            sandboxes.wait_for_creation(sandbox.id)
        console.print(f"[dim]Sandbox ready in {time.monotonic() - started:.0f}s[/dim]")

        started = time.monotonic()
        with console.status(f"[bold blue]Installing prime-rl@{ref}...", spinner="dots"):
            setup_script = EVAL_SETUP_SCRIPT.format(
                workdir=EVAL_SANDBOX_WORKDIR, repo=PRIME_RL_REPO, ref=shlex.quote(ref)
            )
            # Background jobs run under `sh`; the script needs bash for `pipefail`.
            setup = sandboxes.run_background_job(
                sandbox.id,
                f"bash -c {shlex.quote(setup_script)}",
                timeout=EVAL_SETUP_TIMEOUT_SECONDS,
            )
        if setup.exit_code != 0:
            console.print(setup.stdout)
            console.print(f"[red]Installing prime-rl failed (exit {setup.exit_code}):[/red]")
            console.print(setup.stderr)
            raise typer.Exit(setup.exit_code or 1)
        console.print(f"[dim]prime-rl@{ref} installed in {time.monotonic() - started:.0f}s[/dim]")

        if env_path is not None:
            _upload_local_env(sandboxes, sandbox.id, env_path)
        if install_command is not None:
            with console.status("[bold blue]Installing the environment...", spinner="dots"):
                install = sandboxes.run_background_job(
                    sandbox.id,
                    install_command,
                    timeout=EVAL_SETUP_TIMEOUT_SECONDS,
                    working_dir=f"{EVAL_SANDBOX_WORKDIR}/prime-rl",
                )
            if install.exit_code != 0:
                console.print(install.stdout)
                console.print(
                    f"[red]Installing the environment failed (exit {install.exit_code}):[/red]"
                )
                console.print(install.stderr)
                raise typer.Exit(install.exit_code or 1)

        if config_file is not None:
            sandboxes.upload_file(
                sandbox.id, f"{EVAL_SANDBOX_WORKDIR}/prime-rl/{config_file.name}", str(config_file)
            )

        command = shlex.join(["uv", "run", "eval", *eval_args])
        console.print(f"[bold blue]Running:[/bold blue] {command}")
        console.print(
            f"[dim]Follow along: prime sandbox run {sandbox.id} -w {EVAL_SANDBOX_WORKDIR}/prime-rl "
            "-- bash -c 'tail -n 50 outputs/*/logs/latest/eval.log'[/dim]"
        )
        result = sandboxes.run_background_job(
            sandbox.id,
            command,
            timeout=EVAL_NO_DEADLINE_SECONDS,
            working_dir=f"{EVAL_SANDBOX_WORKDIR}/prime-rl",
            env=env_vars,
        )
        if result.stdout:
            console.print(result.stdout)
        if result.stderr:
            console.print(result.stderr)
        if result.exit_code != 0:
            console.print(f"[red]Eval exited with code {result.exit_code}[/red]")
            raise typer.Exit(result.exit_code or 1)
        console.print("[green]✓ Eval finished[/green] - results are under `prime eval list`")
    finally:
        if keep:
            console.print(f"[dim]Sandbox kept: prime sandbox get {sandbox.id}[/dim]")
        else:
            sandboxes.delete(sandbox.id)
