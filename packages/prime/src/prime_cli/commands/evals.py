import inspect
import json
import re
import shlex
import uuid
from functools import wraps
from pathlib import Path
from typing import Any, Optional

import typer
from prime_evals import EvalsAPIError, EvalsClient, InvalidEvaluationError
from prime_sandboxes import CreateSandboxRequest, SandboxClient
from rich.progress import Progress
from rich.syntax import Syntax
from rich.table import Table

from ..client import APIClient
from ..core import Config
from ..utils import (
    DefaultCommandGroup,
    PlainTyper,
    get_console,
    json_output_help,
    output_data_as_json,
)
from ..utils.display import get_eval_viewer_url
from ..utils.eval_push import load_results_jsonl

console = get_console()

LIST_EVALS_JSON_HELP = json_output_help(
    ".evaluations[] = {evaluation_id|id, environment_names[], model_name, status, metadata}",
    ".total = number",
)

EVAL_DETAIL_JSON_HELP = json_output_help(
    ". = evaluation object from Prime Evals",
    "Common keys: .evaluation_id? | .id, .environment_names[]?, .model_name?, "
    ".status?, .metadata?, .metrics?",
)

EVAL_SAMPLES_JSON_HELP = json_output_help(
    ".samples[] = sample object",
    "Common keys: .samples[].example_id?, .samples[].input?, .samples[].output?, .samples[].score?",
    ".total? = number",
    ".page? = number",
    ".limit? = number",
)

PUSH_EVAL_JSON_HELP = json_output_help(
    "Single push: .evaluation_id = string",
    "Auto-discovery batch push: .results[] = {path, status, eval_id?, error?}",
)

EVAL_TABLE_MAX_TEXT_WIDTH = 30
EVAL_HOSTED_LABEL = "HOSTED"
EVAL_LOCAL_LABEL = "LOCAL"

PRIME_RL_REPO = "https://github.com/PrimeIntellect-ai/prime-rl.git"
EVAL_SANDBOX_IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm"
EVAL_SANDBOX_WORKDIR = "/workspace"
EVAL_SETUP_TIMEOUT_SECONDS = 45 * 60
# Submodules are pinned to SSH URLs; the sandbox has no GitHub key, so route them
# over HTTPS. Exported (not `git config`) so `git submodule--helper clone` sees it.
EVAL_SETUP_SCRIPT = """
set -euo pipefail
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


def format_output(data: dict, as_json: bool) -> None:
    if as_json:
        output_data_as_json(data, console)
    else:
        syntax = Syntax(json.dumps(data, indent=2), "json", theme="monokai")
        console.print(syntax)


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
    """List evaluations."""

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

        table = Table()
        table.add_column("ID", style="cyan")
        table.add_column("Environment", style="blue")
        table.add_column("Model", style="magenta")
        table.add_column("Status", style="yellow")
        table.add_column("Type", style="green", justify="center")
        table.add_column("Examples", style="dim", justify="right")
        table.add_column("Rollouts", style="dim", justify="right")

        for e in evals:
            eval_id = str(e.get("evaluation_id", e.get("id", "")))
            metadata = e.get("metadata", {})
            num_examples = metadata.get("num_examples", "-")
            rollouts_per_example = metadata.get("rollouts_per_example", "-")

            env_name = "-"
            environment_names = e.get("environment_names", [])
            if environment_names and len(environment_names) > 0:
                env_name = environment_names[0]

            is_hosted = bool(e.get("is_hosted"))
            execution_mode = EVAL_HOSTED_LABEL if is_hosted else EVAL_LOCAL_LABEL

            table.add_row(
                eval_id if eval_id else "",
                str(env_name)[:EVAL_TABLE_MAX_TEXT_WIDTH],
                str(e.get("model_name", ""))[:EVAL_TABLE_MAX_TEXT_WIDTH],
                str(e.get("status", "")),
                execution_mode,
                str(num_examples),
                str(rollouts_per_example),
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


@subcommands_app.command("get", epilog=EVAL_DETAIL_JSON_HELP)
@handle_errors
def get_eval(
    eval_id: str = typer.Argument(..., help="The ID of the evaluation to retrieve"),
    as_json: bool = typer.Option(False, "--json", help="Output JSON instead of a table"),
) -> None:
    api_client = APIClient()
    client = EvalsClient(api_client)
    data = client.get_evaluation(eval_id)
    format_output(data, as_json)


@subcommands_app.command("samples", epilog=EVAL_SAMPLES_JSON_HELP)
@handle_errors
def get_samples(
    eval_id: str = typer.Argument(..., help="The ID of the evaluation"),
    page: int = typer.Option(1, "--page", "-p", help="Page number"),
    num: int = typer.Option(100, "--num", "-n", help="Items per page"),
    as_json: bool = typer.Option(False, "--json", help="Output JSON instead of a table"),
) -> None:
    api_client = APIClient()
    client = EvalsClient(api_client)
    data = client.get_samples(eval_id, page=page, limit=num)
    format_output(data, as_json)


def _load_eval_directory(directory: Path) -> dict:
    with open(directory / "metadata.json") as f:
        metadata = json.load(f)

    env_field = metadata.get("env_id") or metadata.get("env")
    if not env_field or "model" not in metadata:
        raise ValueError(
            f"Missing required 'env_id' or 'model' field in {directory / 'metadata.json'}"
        )

    results = load_results_jsonl(directory / "results.jsonl")

    for sample in results:
        if "id" in sample and "example_id" not in sample:
            sample["example_id"] = sample["id"]

    avg_pattern = re.compile(r"^avg_(.+)$")
    metrics = {}
    metadata_copy = {}
    for key, value in metadata.items():
        if match := avg_pattern.match(key):
            metrics[match.group(1)] = value
        else:
            metadata_copy[key] = value

    return {
        "eval_name": f"{env_field}-{metadata['model']}",
        "model_name": metadata["model"],
        "env": env_field,
        "metrics": metrics,
        "metadata": metadata_copy,
        "results": results,
    }


def _has_eval_files(directory: Path) -> bool:
    return (directory / "metadata.json").exists() and (directory / "results.jsonl").exists()


def _validate_eval_path(path_str: str) -> Path:
    """Validate and return the evaluation directory path."""
    path = Path(path_str)

    if path.is_file():
        # Auto-correct: if user passed metadata.json or results.jsonl, use parent directory
        if path.name in ("metadata.json", "results.jsonl"):
            parent = path.parent
            if _has_eval_files(parent):
                return parent
            raise ValueError(
                f"Directory '{parent}' must contain both metadata.json and results.jsonl"
            )
        raise ValueError(
            f"Expected a directory path, but got file: {path}\n"
            f"Pass a directory containing metadata.json and results.jsonl"
        )

    if path.is_dir():
        if _has_eval_files(path):
            return path

        has_metadata = (path / "metadata.json").exists()
        has_results = (path / "results.jsonl").exists()
        if has_metadata and not has_results:
            raise ValueError(f"Directory '{path}' is missing results.jsonl")
        elif has_results and not has_metadata:
            raise ValueError(f"Directory '{path}' is missing metadata.json")
        else:
            raise ValueError(f"Directory '{path}' is missing both metadata.json and results.jsonl")

    raise FileNotFoundError(f"Path not found: {path}")


def _discover_eval_outputs() -> list[Path]:
    outputs_dir = Path("outputs/evals")
    if not outputs_dir.exists():
        return []

    eval_dirs = []
    for env_dir in outputs_dir.iterdir():
        if not env_dir.is_dir():
            continue
        for run_dir in env_dir.iterdir():
            if run_dir.is_dir() and _has_eval_files(run_dir):
                eval_dirs.append(run_dir)

    return sorted(eval_dirs)


def _resolve_eval_viewer_url(evaluation_id: str, response: Optional[dict[str, Any]] = None) -> str:
    viewer_url = response.get("viewer_url") if response else None
    if viewer_url:
        return str(viewer_url)
    return get_eval_viewer_url(evaluation_id)


def _push_samples_with_progress(
    client: EvalsClient, evaluation_id: str, samples: list[dict[str, Any]]
) -> None:
    if not console.is_terminal or not _push_samples_accepts_progress_callback(client):
        client.push_samples(evaluation_id, samples)
        return

    with Progress(console=console, transient=True) as progress:
        task_id = progress.add_task("Uploading samples", total=len(samples))
        client.push_samples(
            evaluation_id,
            samples,
            progress_callback=lambda uploaded: progress.update(task_id, advance=uploaded),
        )


def _push_samples_accepts_progress_callback(client: EvalsClient) -> bool:
    try:
        parameters = inspect.signature(client.push_samples).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == "progress_callback" or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _require_published_environment_for_eval_push(env_name: str, eval_path: Path) -> None:
    console.print("[red]Error:[/red] Evaluation uploads require a pushed environment.")
    console.print(
        f"[yellow]Push '{env_name}' before uploading this evaluation:[/yellow] "
        f"prime env push {env_name}"
    )
    console.print("[dim]Then retry with an owner-qualified environment:[/dim]")
    console.print(f"[dim]  --env <owner>/{env_name}[/dim]")
    console.print(f"[dim]Example: prime eval push {eval_path} --env <owner>/{env_name}[/dim]")
    raise typer.Exit(1)


def _push_single_eval(
    config_path: str,
    env_slug: Optional[str],
    run_id: Optional[str],
    eval_id: Optional[str],
    is_public: bool = False,
    name: Optional[str] = None,
) -> str:
    path = _validate_eval_path(config_path)
    eval_data = _load_eval_directory(path)
    eval_name = name or eval_data["eval_name"]
    console.print(f"[blue]✓ Loaded eval data:[/blue] {path}")

    detected_env = eval_data.get("env_id") or eval_data.get("env")
    if not env_slug and detected_env and not run_id and not eval_id:
        env_slug = detected_env

    environments = None
    if env_slug and not run_id and not eval_id:
        if "/" not in env_slug:
            _require_published_environment_for_eval_push(env_slug, path)
        environments = [{"slug": env_slug}]

    console.print()

    api_client = APIClient()
    client = EvalsClient(api_client)

    if eval_id:
        console.print(f"[blue]Checking evaluation:[/blue] {eval_id}")
        try:
            client.get_evaluation(eval_id)
            console.print("[green]✓ Found existing evaluation[/green]")

            console.print("[blue]Updating evaluation...[/blue]")
            client.update_evaluation(
                evaluation_id=eval_id,
                name=eval_name,
                model_name=eval_data.get("model_name"),
                framework=eval_data.get("metadata", {}).get("framework", "verifiers"),
                task_type=eval_data.get("metadata", {}).get("task_type"),
                metadata=eval_data.get("metadata"),
                metrics=eval_data.get("metrics"),
                tags=eval_data.get("tags", []),
            )
            console.print(f"[green]✓ Updated evaluation:[/green] {eval_id}")
        except Exception as e:
            console.print(f"[red]Error:[/red] Could not update evaluation {eval_id}: {e}")
            raise
        console.print()
    else:
        console.print("[blue]Creating evaluation...[/blue]")
        create_response = client.create_evaluation(
            name=eval_name,
            environments=environments,
            run_id=run_id,
            model_name=eval_data.get("model_name"),
            framework=eval_data.get("metadata", {}).get("framework", "verifiers"),
            task_type=eval_data.get("metadata", {}).get("task_type"),
            metadata=eval_data.get("metadata"),
            metrics=eval_data.get("metrics"),
            tags=eval_data.get("tags", []),
            is_public=is_public,
        )

        eval_id = create_response.get("evaluation_id")
        if not eval_id:
            raise ValueError("Failed to get evaluation ID from response")

        console.print(f"[green]✓ Created evaluation:[/green] {eval_id}")
        console.print()

    results = eval_data.get("results", [])
    if results:
        console.print(f"[blue]Pushing {len(results)} samples...[/blue]")
        _push_samples_with_progress(client, eval_id, results)
        console.print("[green]✓ Samples pushed successfully[/green]")
        console.print()

    console.print("[blue]Finalizing evaluation...[/blue]")
    finalize_response = client.finalize_evaluation(eval_id, metrics=eval_data.get("metrics"))
    viewer_url = _resolve_eval_viewer_url(eval_id, finalize_response)
    console.print("[green]✓ Evaluation finalized[/green]")
    console.print()

    console.print("[green]✓ Success[/green]")
    console.print(f"[blue]Evaluation ID:[/blue] {eval_id}")
    console.print(f"[dim]View results:[/dim] {viewer_url}")
    console.print()
    console.print("[dim]Inspect evaluation data:[/dim]")
    console.print(f"  prime eval get {eval_id}")
    console.print(f"  prime eval samples {eval_id}")

    return eval_id


@subcommands_app.command("push", epilog=PUSH_EVAL_JSON_HELP)
@handle_errors
def push_eval(
    config_path: Optional[str] = typer.Argument(
        None,
        help=(
            "Path to eval directory containing metadata.json and results.jsonl. "
            "If not provided, auto-discovers from outputs/evals/"
        ),
    ),
    env_id: Optional[str] = typer.Option(
        None,
        "--env",
        "--env-id",
        "-e",
        help=(
            "Published environment slug (owner/name). "
            "Push local environments with `prime env push` first."
        ),
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        "-r",
        help="Link to existing training run id",
    ),
    eval_id: Optional[str] = typer.Option(
        None,
        "--eval",
        "--eval-id",
        help="Push to existing evaluation id",
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        help="Explicit evaluation name override",
    ),
    is_public: bool = typer.Option(
        False,
        "--public",
        help="Make the pushed evaluation public. Evaluations are private by default.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Output JSON instead of a table"),
) -> None:
    """Push evaluation data to Prime Evals.

    The directory must contain metadata.json and results.jsonl files.

    \b
    Examples:
        prime eval push                                    # Push current dir or auto-discover
        prime eval push outputs/evals/gsm8k--gpt-4/abc123  # Push specific directory
        prime eval push --env owner/gsm8k                  # Push with environment override
        prime eval push --name "gsm8k smoke test"         # Override evaluation display name
        prime eval push --public                           # Create a public evaluation
        prime eval push --eval xyz789 --name "rerun"      # Update an existing evaluation name
    """
    try:
        if eval_id and is_public:
            console.print(
                "[red]Error:[/red] The --public flag cannot be used with --eval-id. "
                "Visibility can only be set when creating a new evaluation."
            )
            raise typer.Exit(1)

        if config_path is None and eval_id:
            console.print("[red]Error:[/red] Cannot use --eval-id with auto-discovery")
            console.print()
            console.print("[yellow]Tip:[/yellow] Specify an explicit path when using --eval-id:")
            console.print("  prime eval push /path/to/eval/data --eval-id <eval-id>")
            console.print("  prime eval push outputs/evals/env--model/run-id --eval-id <eval-id>")
            raise typer.Exit(1)

        if config_path is None:
            current_dir = Path(".")
            if _has_eval_files(current_dir):
                result_eval_id = _push_single_eval(".", env_id, run_id, eval_id, is_public, name)
                if as_json:
                    console.print()
                    output_data_as_json({"evaluation_id": result_eval_id}, console)
                return

            eval_dirs = _discover_eval_outputs()
            if not eval_dirs:
                console.print("[red]Error:[/red] No evaluation outputs found")
                console.print(
                    "[yellow]Hint:[/yellow] Run from a directory with "
                    "metadata.json and results.jsonl, or from a directory containing outputs/evals/"
                )
                raise typer.Exit(1)

            console.print(f"[blue]Found {len(eval_dirs)} evaluation(s) to push:[/blue]")
            for eval_dir in eval_dirs:
                console.print(f"  - {eval_dir}")
            console.print()

            results = []
            for eval_dir in eval_dirs:
                try:
                    result_eval_id = _push_single_eval(
                        str(eval_dir), env_id, run_id, eval_id, is_public, name
                    )
                    results.append(
                        {"path": str(eval_dir), "eval_id": result_eval_id, "status": "success"}
                    )
                except Exception as e:
                    console.print(f"[red]Failed to push {eval_dir}:[/red] {e}")
                    results.append({"path": str(eval_dir), "error": str(e), "status": "failed"})
                console.print()

            success_count = sum(1 for r in results if r["status"] == "success")
            console.print(
                f"[blue]Summary:[/blue] {success_count}/{len(eval_dirs)} "
                f"evaluations pushed successfully"
            )

            if as_json:
                output_data_as_json({"results": results}, console)

            if success_count < len(eval_dirs):
                raise typer.Exit(1)

            return

        result_eval_id = _push_single_eval(config_path, env_id, run_id, eval_id, is_public, name)

        if as_json:
            console.print()
            output_data_as_json({"evaluation_id": result_eval_id}, console)

    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except json.JSONDecodeError as e:
        console.print(f"[red]Error:[/red] Invalid JSON in metadata.json: {e}")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except InvalidEvaluationError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print()
        console.print("[yellow]Tip:[/yellow] You must provide one of:")
        console.print("  --eval <eval_id>     (to update an existing evaluation)")
        console.print("  --run-id <run_id>    (to link to an existing training run)")
        console.print("  --env <env>          (published environment slug, e.g., 'owner/gsm8k')")
        console.print("  [or ensure owner/name 'env' or 'env_id' is set in metadata.json]")
        raise typer.Exit(1)
    except KeyError as e:
        console.print(f"[red]Error:[/red] Missing required field: {e}")
        console.print(
            "[yellow]Hint:[/yellow] metadata.json must contain 'env' (or 'env_id') and 'model'"
        )
        raise typer.Exit(1)
    except EvalsAPIError as e:
        console.print(f"[red]API Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


app = PlainTyper(
    cls=DefaultGroup,
    help=(
        "Manage hosted evaluations (run, list, get, samples, push)\n\n"
        "By default, 'prime eval <environment>' runs 'prime eval run <environment>'."
    ),
    no_args_is_help=True,
)

app.add_typer(subcommands_app, name="")


app = PlainTyper(
    cls=DefaultGroup,
    help=(
        "Manage hosted evaluations (run, list, get, samples, push)\n\n"
        "By default, 'prime eval <environment>' runs 'prime eval run <environment>'."
    ),
    no_args_is_help=True,
)

app.add_typer(subcommands_app, name="")


@app.command(
    "run",
    help="Run `uv run eval` from prime-rl in a sandbox and stream results to the platform",
    no_args_is_help=True,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def run_eval_cmd(
    ctx: typer.Context,
    environment: str = typer.Argument(
        ..., help="Taskset id (e.g. gsm8k) or `@ eval.toml`; the rest is passed to `uv run eval`"
    ),
    ref: str = typer.Option("main", "--ref", help="prime-rl git ref to check out"),
    image: str = typer.Option(EVAL_SANDBOX_IMAGE, "--image", help="Sandbox docker image"),
    cpu_cores: float = typer.Option(4.0, "--cpu", help="Sandbox CPU cores"),
    memory_gb: float = typer.Option(8.0, "--memory", help="Sandbox memory in GB"),
    disk_size_gb: float = typer.Option(30.0, "--disk", help="Sandbox disk in GB"),
    timeout_minutes: int = typer.Option(
        180, "--timeout-minutes", help="Sandbox lifetime; the eval is killed with it"
    ),
    env_var: Optional[list[str]] = typer.Option(
        None, "--env-var", help="Extra KEY=VALUE for the eval process (repeatable)"
    ),
    keep: bool = typer.Option(False, "--keep", help="Keep the sandbox after the eval exits"),
) -> None:
    """Every argument `prime eval run` does not own is proxied verbatim to `uv run eval`
    (see `uv run eval -h` in prime-rl). `--monitors.prime` is added unless given, so each
    finished source lands as an evaluation on the platform."""
    eval_args = [environment, *ctx.args]
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

    sandboxes = SandboxClient(APIClient())
    sandbox = sandboxes.create(
        CreateSandboxRequest(
            name=f"prime-eval-{uuid.uuid4().hex[:8]}",
            docker_image=image,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            disk_size_gb=disk_size_gb,
            timeout_minutes=timeout_minutes,
            labels=["prime-eval"],
        )
    )
    console.print(f"[dim]Sandbox {sandbox.id} ({image})[/dim]")
    try:
        with console.status("[bold blue]Waiting for sandbox...", spinner="dots"):
            sandboxes.wait_for_creation(sandbox.id)

        with console.status(f"[bold blue]Installing prime-rl@{ref}...", spinner="dots"):
            setup = sandboxes.run_background_job(
                sandbox.id,
                EVAL_SETUP_SCRIPT.format(repo=PRIME_RL_REPO, ref=shlex.quote(ref)),
                timeout=EVAL_SETUP_TIMEOUT_SECONDS,
                working_dir=EVAL_SANDBOX_WORKDIR,
            )
        if setup.exit_code != 0:
            console.print(setup.stdout)
            console.print(f"[red]Installing prime-rl failed (exit {setup.exit_code}):[/red]")
            console.print(setup.stderr)
            raise typer.Exit(setup.exit_code or 1)

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
            timeout=timeout_minutes * 60,
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
