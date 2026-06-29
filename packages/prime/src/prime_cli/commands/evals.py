from __future__ import annotations

import inspect
import json
import time
import tomllib
from functools import wraps
from pathlib import Path
from typing import Any, Optional

from prime_evals import EvalsAPIError, EvalsClient, InvalidEvaluationError
from pydantic import AliasChoices, Field, ValidationError
from pydantic_config import BaseConfig
from rich.progress import Progress
from rich.syntax import Syntax
from rich.table import Table

from ..client import APIClient, APIError
from ..core import Config
from ..utils import (
    get_console,
    json_output_help,
    output_data_as_json,
)
from ..utils.display import get_eval_viewer_url
from ..utils.env_metadata import find_environment_metadata, parse_env_id
from ..utils.hosted_eval import (
    EvalStatus,
    HostedEvalConfig,
    clean_logs,
    get_new_log_lines,
)
from ..verifiers_bridge import run_eval_view

console = get_console()

# verifiers.* must be imported inside functions (not at module top): top-level
# imports drag in huggingface_hub/datasets/pyarrow/pandas/numpy and triple
# `prime --version` startup time. Same convention as the rest of prime_cli.

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

HOSTED_LOGS_DEFAULT_TAIL_LINES = 1000
HOSTED_LOGS_DEFAULT_POLL_INTERVAL_SECONDS = 5.0
HOSTED_RUN_DEFAULT_POLL_INTERVAL_SECONDS = 10.0
HOSTED_LOGS_RATE_LIMIT_THRESHOLD = 3
HOSTED_LOGS_RATE_LIMIT_WAIT_SECONDS = 30
HOSTED_LOGS_RETRY_WAIT_SECONDS = 10
HOSTED_LOGS_STATUS_UPDATE_EVERY_POLLS = 6
EVAL_TABLE_MAX_TEXT_WIDTH = 30
EVAL_SUBMIT_EXAMPLE_COMMAND = "prime eval submit gsm8k"
EVAL_HOSTED_LABEL = "HOSTED"
EVAL_LOCAL_LABEL = "LOCAL"


def handle_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SystemExit:
            raise
        except EvalsAPIError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise SystemExit(1)
        except Exception as e:
            console.print(f"[red]Unexpected error:[/red] {e}")
            raise SystemExit(1)

    return wrapper


def _validate_output_format(output: str, allowed: list[str]) -> None:
    if output not in allowed:
        console.print(f"[red]Error:[/red] output must be one of: {', '.join(allowed)}")
        raise SystemExit(1)


def format_output(data: dict, output: str) -> None:
    if output == "json":
        output_data_as_json(data, console)
    else:
        syntax = Syntax(json.dumps(data, indent=2), "json", theme="monokai")
        console.print(syntax)


def _parse_json_object_option(raw: Optional[str], option_name: str) -> Optional[dict[str, Any]]:
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        console.print(f"[red]Error:[/red] invalid {option_name}: {exc}")
        raise SystemExit(1) from exc

    if type(parsed) is not dict:
        console.print(f"[red]Error:[/red] {option_name} must be a JSON object")
        raise SystemExit(1)

    return parsed


def _parse_string_map_option(raw: Optional[str], option_name: str) -> Optional[dict[str, str]]:
    parsed = _parse_json_object_option(raw, option_name)
    if parsed is None:
        return None

    for key, value in parsed.items():
        if type(key) is not str or type(value) is not str:
            console.print(
                f"[red]Error:[/red] {option_name} must contain only string keys and values"
            )
            raise SystemExit(1)

    return parsed


def _load_hosted_eval_configs(config_path: Path) -> list[HostedEvalConfig]:
    """Load Prime's strict hosted-eval TOML format without the Verifiers V0 parser."""
    try:
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        console.print(f"[red]Error:[/red] invalid hosted eval config: {exc}")
        raise SystemExit(1) from exc
    entries = raw.pop("eval", None)
    if entries is None:
        entries = [raw]
        defaults: dict[str, Any] = {}
    elif isinstance(entries, list) and entries:
        defaults = raw
    else:
        console.print("[red]Error:[/red] hosted config requires one or more [[eval]] tables")
        raise SystemExit(1)
    try:
        return [HostedEvalConfig.model_validate({**defaults, **entry}) for entry in entries]
    except (TypeError, ValidationError) as exc:
        console.print(f"[red]Error:[/red] invalid hosted eval config: {exc}")
        raise SystemExit(1) from exc


def _fetch_eval_status(client: APIClient, eval_id: str) -> dict[str, Any]:
    return client.get(f"/evaluations/{eval_id}")


def _fetch_logs(client: APIClient, eval_id: str) -> str:
    response = client.get(f"/hosted-evaluations/{eval_id}/logs")
    return response.get("logs") or ""


def _build_hosted_evaluation_payload(
    config: HostedEvalConfig, environment_ids: list[str]
) -> dict[str, Any]:
    eval_config: dict[str, Any] = {
        "num_examples": config.num_examples,
        "rollouts_per_example": config.rollouts_per_example,
        "allow_sandbox_access": config.allow_sandbox_access,
        "allow_instances_access": config.allow_instances_access,
        "allow_tunnel_access": config.allow_tunnel_access,
    }

    if config.env_args:
        eval_config["env_args"] = config.env_args
    if config.timeout_minutes is not None:
        eval_config["timeout_minutes"] = config.timeout_minutes
    if config.custom_secrets:
        eval_config["custom_secrets"] = config.custom_secrets
    if config.sampling_args:
        eval_config["sampling_args"] = config.sampling_args
    if config.max_concurrent is not None:
        eval_config["max_concurrent"] = config.max_concurrent
    if config.max_retries is not None:
        eval_config["max_retries"] = config.max_retries
    if config.state_columns:
        eval_config["state_columns"] = config.state_columns
    if config.independent_scoring:
        eval_config["independent_scoring"] = True
    if config.verbose:
        eval_config["verbose"] = True
    if config.headers:
        eval_config["headers"] = config.headers
    if config.extra_env_kwargs:
        eval_config["extra_env_kwargs"] = config.extra_env_kwargs
    if config.api_client_type:
        eval_config["api_client_type"] = config.api_client_type
    if config.api_base_url:
        eval_config["api_base_url"] = config.api_base_url
    if config.api_key_var:
        eval_config["api_key_var"] = config.api_key_var

    payload: dict[str, Any] = {
        "environment_ids": environment_ids,
        "inference_model": config.model,
        "eval_config": eval_config,
    }
    if config.name:
        payload["name"] = config.name

    return payload


def _create_hosted_evaluations(
    config: HostedEvalConfig, environment_ids: list[str]
) -> dict[str, Any]:
    client = APIClient()
    payload = _build_hosted_evaluation_payload(config, environment_ids)

    if client.config.team_id:
        payload["team_id"] = client.config.team_id

    created = client.post("/hosted-evaluations", json=payload)
    evaluation_id = created.get("evaluation_id")
    evaluation_ids = created.get("evaluation_ids")

    if not evaluation_id and not evaluation_ids:
        raise APIError(f"Failed to get evaluation ID from response: {created}")

    return created


def _print_eval_status(eval_data: dict[str, Any]) -> None:
    status_str, status = _parse_eval_status(eval_data)
    color = status.color if status else "white"

    console.print(f"[{color}]Status: {status_str}[/{color}]")

    error_message = eval_data.get("error_message")
    if error_message:
        console.print(f"[red]Error:[/red] {error_message}")

    viewer_url = eval_data.get("viewer_url")
    if viewer_url:
        console.print(f"[dim]View: {viewer_url}[/dim]")
        return

    eval_id = eval_data.get("evaluation_id")
    if eval_id:
        console.print(f"[dim]View: {get_eval_viewer_url(eval_id)}[/dim]")


def _parse_eval_status(eval_data: dict[str, Any]) -> tuple[str, EvalStatus | None]:
    raw_status = eval_data.get("status")
    status_str = raw_status if type(raw_status) is str else "UNKNOWN"

    try:
        return status_str, EvalStatus(status_str)
    except ValueError:
        return status_str, None


def _handle_log_poll_error(eval_id: str, exc: APIError, consecutive_errors: int) -> None:
    if "404" in str(exc):
        console.print(f"\n[red]Evaluation {eval_id} not found.[/red]")
        raise SystemExit(1) from exc

    if "429" not in str(exc):
        raise exc

    if consecutive_errors >= HOSTED_LOGS_RATE_LIMIT_THRESHOLD:
        console.print(
            f"[yellow]Rate limited. Waiting {HOSTED_LOGS_RATE_LIMIT_WAIT_SECONDS:.0f}s...[/yellow]"
        )
        time.sleep(HOSTED_LOGS_RATE_LIMIT_WAIT_SECONDS)
    else:
        time.sleep(HOSTED_LOGS_RETRY_WAIT_SECONDS)


def _display_logs_follow(eval_id: str, poll_interval: float) -> None:
    console.print(f"[dim]Watching logs for evaluation {eval_id}... (Ctrl+C to stop)[/dim]\n")

    client = APIClient()
    last_logs = ""
    consecutive_errors = 0
    no_logs_polls = 0

    while True:
        try:
            eval_data = _fetch_eval_status(client, eval_id)
            status_str, status = _parse_eval_status(eval_data)

            if status and status in EvalStatus.terminal_statuses():
                final_logs = clean_logs(_fetch_logs(client, eval_id))
                if final_logs and final_logs != last_logs:
                    for line in get_new_log_lines(last_logs, final_logs):
                        console.print(line)
                    last_logs = final_logs
                console.print()
                _print_eval_status(eval_data)
                if status != EvalStatus.COMPLETED:
                    raise SystemExit(1)
                return

            raw_logs = _fetch_logs(client, eval_id)
            logs = clean_logs(raw_logs) if raw_logs else ""
            consecutive_errors = 0

            if logs and logs != last_logs:
                for line in get_new_log_lines(last_logs, logs):
                    console.print(line)
                last_logs = logs
                no_logs_polls = 0
            else:
                no_logs_polls += 1

            if no_logs_polls > 0 and no_logs_polls % HOSTED_LOGS_STATUS_UPDATE_EVERY_POLLS == 0:
                console.print(f"[dim]Evaluation status: {status_str} (waiting for logs...)[/dim]")
        except APIError as exc:
            consecutive_errors += 1
            _handle_log_poll_error(eval_id, exc, consecutive_errors)
            continue

        time.sleep(poll_interval)


def _display_logs_once(eval_id: str, tail: int) -> None:
    client = APIClient()
    eval_data = _fetch_eval_status(client, eval_id)
    raw_logs = _fetch_logs(client, eval_id)
    logs = clean_logs(raw_logs) if raw_logs else ""

    if logs:
        for line in logs.splitlines()[-tail:]:
            console.print(line)
    else:
        console.print("[yellow]No logs available.[/yellow]")

    console.print()
    _print_eval_status(eval_data)


def _display_logs(
    eval_id: str,
    tail: int,
    follow: bool,
    poll_interval: float = HOSTED_LOGS_DEFAULT_POLL_INTERVAL_SECONDS,
) -> None:
    try:
        if follow:
            _display_logs_follow(eval_id, poll_interval)
        else:
            _display_logs_once(eval_id, tail)
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped watching logs. Evaluation continues running.[/dim]")
    except SystemExit:
        raise
    except APIError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1) from exc


def _resolve_hosted_environment(
    environment: str,
    *,
    env_path: Optional[str],
) -> tuple[str, str]:
    """Resolve an explicit slug or a local package's recorded upstream."""
    platform_slug = None
    if "/" in environment:
        try:
            owner, name, version = parse_env_id(environment)
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise SystemExit(1) from exc
        if version is not None:
            console.print("[red]Error:[/red] hosted evaluations accept only unversioned slugs")
            raise SystemExit(1)
        platform_slug = f"{owner}/{name}"
    if platform_slug is None:
        module_name = environment.replace("-", "_")
        metadata = find_environment_metadata(
            env_name=environment,
            env_path=Path(env_path) if env_path else Path("environments") / module_name,
            module_name=module_name,
        )
        owner = metadata.get("owner") if metadata else None
        name = metadata.get("name") if metadata else None
        if owner and name:
            platform_slug = f"{owner}/{name}"

    if platform_slug is None:
        console.print(
            "[red]Error:[/red] hosted evaluations require an upstream environment on the platform"
        )
        console.print(
            "[yellow]Use an environment slug or publish the local environment "
            "with `prime env push`.[/yellow]"
        )
        raise SystemExit(1)

    owner, env_name = platform_slug.split("/")
    api_client = APIClient()
    try:
        response = api_client.get(f"/environmentshub/{owner}/{env_name}/@latest")
    except APIError as exc:
        message = str(exc).lower()
        if "404" in message or "not found" in message:
            console.print(
                "[red]Error:[/red] hosted evaluations require an environment "
                "that is published to the platform"
            )
            console.print(f"[yellow]Publish {platform_slug} with `prime env push` first.[/yellow]")
            raise SystemExit(1) from exc
        raise

    details = response.get("data", response)
    environment_id = details.get("id")
    if not environment_id:
        console.print(f"[red]Error:[/red] could not resolve environment id for {platform_slug}")
        raise SystemExit(1)

    console.print(
        f"[dim]Hosted evaluations always use the latest published version of {platform_slug}.[/dim]"
    )
    console.print(f"[dim]Using hosted environment {platform_slug}@latest[/dim]")
    return platform_slug, str(environment_id)


def list_evals(config: EvalListConfig) -> None:
    """List evaluations."""
    output = config.output
    num = config.num
    page = config.page
    env = config.env

    _validate_output_format(output, ["table", "json"])

    if num < 1 or page < 1:
        console.print("[red]Error:[/red] --num and --page must be at least 1")
        raise SystemExit(1)

    try:
        api_client = APIClient()
        prime_config = Config()
        client = EvalsClient(api_client)

        skip = (page - 1) * num
        data = client.list_evaluations(
            env_name=env,
            team_id=prime_config.team_id,
            skip=skip,
            limit=num,
        )

        if output == "json":
            output_data_as_json(data, console)
            return

        evals = data.get("evaluations", [])

        if not evals:
            if page > 1:
                console.print("[yellow]No more results.[/yellow]")
            else:
                console.print("[yellow]No evaluations found.[/yellow]")
            return

        table = Table(title="Evaluations")
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
            console.print(
                f"\n[yellow]Showing page {page} of results. "
                f"Use --page {page + 1} to see more.[/yellow]"
            )
        else:
            console.print(f"\n[dim]Total: {total} evaluation(s)[/dim]")

    except EvalsAPIError as e:
        console.print(f"[red]API Error:[/red] {e}")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print(
            "[yellow]Response may contain invalid data. "
            "Try --output json to see raw response.[/yellow]"
        )
        raise SystemExit(1)


def get_eval(config: EvalGetConfig) -> None:
    """Show evaluation details."""
    eval_id = config.eval_id
    output = config.output

    _validate_output_format(output, ["json", "pretty"])

    api_client = APIClient()
    client = EvalsClient(api_client)
    data = client.get_evaluation(eval_id)
    format_output(data, output)


def get_samples(config: EvalSamplesConfig) -> None:
    eval_id = config.eval_id
    page = config.page
    num = config.num
    output = config.output

    _validate_output_format(output, ["json", "pretty"])

    api_client = APIClient()
    client = EvalsClient(api_client)
    data = client.get_samples(eval_id, page=page, limit=num)
    format_output(data, output)


def _load_eval_directory(directory: Path) -> dict:
    from verifiers.v1.cli.output import read_upload_data

    upload = read_upload_data(directory)
    if upload.invalid_results:
        preview = [f"line {error.line}: {error.reason}" for error in upload.invalid_results[:5]]
        suffix = ", ..." if len(upload.invalid_results) > 5 else ""
        console.print(
            f"[yellow]Warning: Skipped {len(upload.invalid_results)} invalid lines "
            f"in results.jsonl ({', '.join(preview)}{suffix})[/yellow]"
        )
    return upload.as_dict()


def _has_eval_files(directory: Path) -> bool:
    from verifiers.v1.cli.output import has_eval_artifacts

    return has_eval_artifacts(directory)


def _validate_eval_path(path_str: str) -> Path:
    from verifiers.v1.cli.output import resolve_eval_artifact_dir

    return resolve_eval_artifact_dir(path_str)


def _discover_eval_outputs() -> list[Path]:
    from verifiers.v1.cli.output import discover_eval_artifact_dirs

    return discover_eval_artifact_dirs()


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
    raise SystemExit(1)


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


def view_cmd(config: EvalViewConfig) -> None:
    """Launch the interactive evaluation viewer."""
    limit = config.limit
    env_dir = config.env_dir
    outputs_dir = config.outputs_dir

    if limit < 1:
        console.print("[red]Error:[/red] --limit must be at least 1")
        raise SystemExit(1)
    run_eval_view(env_dir=env_dir, outputs_dir=outputs_dir, limit=limit)


def push_eval(config: EvalPushConfig) -> None:
    """Push native or legacy evaluation data to Prime Evals.

    Both Verifiers V1 run artifacts and V0 metadata.json/results.jsonl outputs are accepted.

    \b
    Examples:
        prime eval push                                    # Push current dir or auto-discover
        prime eval push outputs/evals/gsm8k--gpt-4/abc123  # Push specific directory
        prime eval push --env owner/gsm8k                  # Push with environment override
        prime eval push --name "gsm8k smoke test"         # Override evaluation display name
        prime eval push --public                           # Create a public evaluation
        prime eval push --eval xyz789 --name "rerun"      # Update an existing evaluation name
    """
    config_path = config.config_path
    env_id = config.env_id
    run_id = config.run_id
    eval_id = config.eval_id
    name = config.name
    output = config.output
    is_public = config.is_public

    try:
        if eval_id and is_public:
            console.print(
                "[red]Error:[/red] The --public flag cannot be used with --eval-id. "
                "Visibility can only be set when creating a new evaluation."
            )
            raise SystemExit(1)

        if config_path is None and eval_id:
            console.print("[red]Error:[/red] Cannot use --eval-id with auto-discovery")
            console.print()
            console.print("[yellow]Tip:[/yellow] Specify an explicit path when using --eval-id:")
            console.print("  prime eval push /path/to/eval/data --eval-id <eval-id>")
            console.print("  prime eval push outputs/evals/env--model/run-id --eval-id <eval-id>")
            raise SystemExit(1)

        if config_path is None:
            current_dir = Path(".")
            if _has_eval_files(current_dir):
                result_eval_id = _push_single_eval(".", env_id, run_id, eval_id, is_public, name)
                if output == "json":
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
                raise SystemExit(1)

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

            if output == "json":
                output_data_as_json({"results": results}, console)

            if success_count < len(eval_dirs):
                raise SystemExit(1)

            return

        result_eval_id = _push_single_eval(config_path, env_id, run_id, eval_id, is_public, name)

        if output == "json":
            console.print()
            output_data_as_json({"evaluation_id": result_eval_id}, console)

    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)
    except json.JSONDecodeError as e:
        console.print(f"[red]Error:[/red] Invalid JSON in metadata.json: {e}")
        raise SystemExit(1)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)
    except InvalidEvaluationError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print()
        console.print("[yellow]Tip:[/yellow] You must provide one of:")
        console.print("  --eval <eval_id>     (to update an existing evaluation)")
        console.print("  --run-id <run_id>    (to link to an existing training run)")
        console.print("  --env <env>          (published environment slug, e.g., 'owner/gsm8k')")
        console.print("  [or ensure owner/name 'env' or 'env_id' is set in metadata.json]")
        raise SystemExit(1)
    except KeyError as e:
        console.print(f"[red]Error:[/red] Missing required field: {e}")
        console.print(
            "[yellow]Hint:[/yellow] metadata.json must contain 'env' (or 'env_id') and 'model'"
        )
        raise SystemExit(1)
    except EvalsAPIError as e:
        console.print(f"[red]API Error:[/red] {e}")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)


def logs_cmd(config: EvalLogsConfig) -> None:
    """Get logs for a hosted evaluation."""
    eval_id = config.eval_id
    tail = config.tail
    follow = config.follow
    poll_interval = config.poll_interval

    _display_logs(eval_id, tail, follow, poll_interval=poll_interval)


def stop_cmd(config: EvalStopConfig) -> None:
    """Stop a running hosted evaluation."""
    eval_id = config.eval_id

    try:
        client = APIClient()
        result = client.patch(f"/hosted-evaluations/{eval_id}/cancel")
        message = result.get("message") or f"Evaluation {eval_id} cancelled."
        console.print(f"[green]✓ {message}[/green]")
        console.print(f"[dim]View results:[/dim] {get_eval_viewer_url(eval_id)}")
    except APIError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1) from exc


def submit_eval_cmd(config: EvalSubmitConfig) -> None:
    """Submit one environment or a strict ``[[eval]]`` TOML file to the hosted V0 API."""
    environment = config.environment
    env_path = config.env_path
    poll_interval = config.poll_interval
    follow = config.follow
    model = config.model
    num_examples = config.num_examples
    rollouts_per_example = config.rollouts_per_example
    env_args = config.env_args
    extra_env_kwargs = config.extra_env_kwargs
    timeout_minutes = config.timeout_minutes
    allow_sandbox_access = config.allow_sandbox_access
    allow_instances_access = config.allow_instances_access
    allow_tunnel_access = config.allow_tunnel_access
    custom_secrets = config.custom_secrets
    sampling_args = config.sampling_args
    eval_name = config.eval_name
    max_concurrent = config.max_concurrent
    max_retries = config.max_retries
    state_columns = config.state_columns
    independent_scoring = config.independent_scoring
    verbose = config.verbose
    header = config.header
    api_client_type = config.api_client_type
    api_base_url = config.api_base_url
    api_key_var = config.api_key_var

    if environment is None:
        console.print("[red]Error:[/red] Missing argument 'ENVIRONMENT'.")
        console.print(f"[dim]Example: {EVAL_SUBMIT_EXAMPLE_COMMAND}[/dim]")
        raise SystemExit(2)

    targets = (
        _load_hosted_eval_configs(Path(environment))
        if Path(environment).suffix == ".toml"
        else [HostedEvalConfig(env_id=environment)]
    )
    updates = {
        "model": model,
        "num_examples": num_examples,
        "rollouts_per_example": rollouts_per_example,
        "env_args": _parse_json_object_option(env_args, "--env-args"),
        "extra_env_kwargs": _parse_json_object_option(extra_env_kwargs, "--extra-env-kwargs"),
        "timeout_minutes": timeout_minutes,
        "allow_sandbox_access": allow_sandbox_access,
        "allow_instances_access": allow_instances_access,
        "allow_tunnel_access": allow_tunnel_access,
        "custom_secrets": _parse_string_map_option(custom_secrets, "--custom-secrets"),
        "sampling_args": _parse_json_object_option(sampling_args, "--sampling-args"),
        "name": eval_name,
        "max_concurrent": max_concurrent,
        "max_retries": max_retries,
        "state_columns": state_columns,
        "independent_scoring": independent_scoring,
        "verbose": verbose,
        "headers": header,
        "api_client_type": api_client_type,
        "api_base_url": api_base_url,
        "api_key_var": api_key_var,
    }
    overrides = {key: value for key, value in updates.items() if value is not None}
    try:
        targets = [
            HostedEvalConfig.model_validate({**target.model_dump(), **overrides})
            for target in targets
        ]
    except ValidationError as exc:
        console.print(f"[red]Error:[/red] invalid hosted evaluation: {exc}")
        raise SystemExit(1) from exc

    if follow and len(targets) > 1:
        console.print(
            "[red]Error:[/red] `--follow` is only supported for a single hosted evaluation"
        )
        raise SystemExit(1)

    grouped_targets: dict[str, dict[str, Any]] = {}
    target_order: list[str] = []
    for target in targets:
        group_key = json.dumps(target.model_dump(exclude={"env_id"}), sort_keys=True, default=str)
        if group_key not in grouped_targets:
            grouped_targets[group_key] = {
                "target": target,
                "targets": [],
                "platform_slugs": [],
                "environment_ids": [],
            }
            target_order.append(group_key)
        grouped_targets[group_key]["targets"].append(target)

    try:
        for group_key in target_order:
            group = grouped_targets[group_key]
            for grouped_target in group["targets"]:
                platform_slug, environment_id = _resolve_hosted_environment(
                    grouped_target.env_id,
                    env_path=env_path,
                )
                group["platform_slugs"].append(platform_slug)
                group["environment_ids"].append(environment_id)
    except APIError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1) from exc

    all_platform_slugs: list[str] = []
    all_evaluation_ids: list[str] = []
    try:
        for group_key in target_order:
            group = grouped_targets[group_key]
            target = group["target"]
            result = _create_hosted_evaluations(
                target,
                environment_ids=group["environment_ids"],
            )
            all_platform_slugs.extend(group["platform_slugs"])
            all_evaluation_ids.extend(result.get("evaluation_ids") or [result["evaluation_id"]])
    except APIError as exc:
        console.print(f"[red]Hosted evaluation failed:[/red] {exc}")
        raise SystemExit(1) from exc

    if follow:
        console.print("[green]✓ Hosted evaluation started[/green]")
        console.print(f"[cyan]Environment:[/cyan] {all_platform_slugs[0]}")
        console.print(f"[cyan]Evaluation ID:[/cyan] {all_evaluation_ids[0]}")
        console.print()
        _display_logs(
            all_evaluation_ids[0],
            tail=HOSTED_LOGS_DEFAULT_TAIL_LINES,
            follow=True,
            poll_interval=poll_interval,
        )
        return

    console.print("[green]✓ Hosted evaluation started[/green]")
    if len(all_platform_slugs) == 1:
        console.print(f"[cyan]Environment:[/cyan] {all_platform_slugs[0]}")
        console.print(f"[cyan]Evaluation ID:[/cyan] {all_evaluation_ids[0]}")
        console.print(f"[green]View results:[/green] {get_eval_viewer_url(all_evaluation_ids[0])}")
        console.print("[dim]View logs:[/dim] prime eval logs " + all_evaluation_ids[0] + " -f")
        return

    console.print(f"[cyan]Environments:[/cyan] {', '.join(all_platform_slugs)}")
    console.print(f"[cyan]Evaluation IDs:[/cyan] {', '.join(all_evaluation_ids)}")
    console.print("[dim]View logs:[/dim] prime eval logs <evaluation-id> -f")
    return


# --- inlined config schemas (previously in evals_configs) ---
class EvalGetConfig(BaseConfig):
    """Show evaluation details."""

    eval_id: str = Field(..., description="The ID of the evaluation to retrieve")
    output: str = Field(
        "json", validation_alias=AliasChoices("output", "o"), description="json|pretty"
    )


class EvalListConfig(BaseConfig):
    """List evaluations."""

    output: str = Field(
        "table", validation_alias=AliasChoices("output", "o"), description="table|json"
    )
    num: int = Field(20, validation_alias=AliasChoices("num", "n"), description="Items per page")
    page: int = Field(1, validation_alias=AliasChoices("page", "p"), description="Page number")
    env: str | None = Field(
        None,
        validation_alias=AliasChoices("env", "env_name", "e"),
        description="Filter by environment (e.g., 'gsm8k' or 'owner/gsm8k')",
    )


class EvalLogsConfig(BaseConfig):
    """Get logs for a hosted evaluation."""

    eval_id: str = Field(..., description="Evaluation id to get logs for")
    tail: int = Field(
        1000, validation_alias=AliasChoices("tail", "n"), description="Number of lines to show"
    )
    follow: bool = Field(
        False, validation_alias=AliasChoices("follow", "f"), description="Follow log output"
    )
    poll_interval: float = Field(5.0, description="Polling interval in seconds when following logs")


class EvalPushConfig(BaseConfig):
    """Push native or legacy evaluation data to Prime Evals."""

    config_path: str | None = Field(
        None,
        description="Native V1 or legacy evaluation run directory. Auto-discovers when omitted.",
    )
    env_id: str | None = Field(
        None,
        validation_alias=AliasChoices("env_id", "env", "e"),
        description="Published environment slug (owner/name).",
    )
    run_id: str | None = Field(
        None,
        validation_alias=AliasChoices("run_id", "r"),
        description="Link to existing training run id",
    )
    eval_id: str | None = Field(
        None,
        validation_alias=AliasChoices("eval_id", "eval"),
        description="Push to existing evaluation id",
    )
    name: str | None = Field(None, description="Explicit evaluation name override")
    output: str = Field(
        "pretty", validation_alias=AliasChoices("output", "o"), description="json|pretty"
    )
    is_public: bool = Field(
        False,
        validation_alias=AliasChoices("is_public", "public"),
        description="Make the pushed evaluation public. Evaluations are private by default.",
    )


class EvalSamplesConfig(BaseConfig):
    """"""

    eval_id: str = Field(..., description="The ID of the evaluation")
    page: int = Field(1, validation_alias=AliasChoices("page", "p"), description="Page number")
    num: int = Field(100, validation_alias=AliasChoices("num", "n"), description="Items per page")
    output: str = Field(
        "json", validation_alias=AliasChoices("output", "o"), description="json|pretty"
    )


class EvalStopConfig(BaseConfig):
    """Stop a running hosted evaluation."""

    eval_id: str = Field(..., description="Evaluation id to stop")


class EvalSubmitConfig(BaseConfig):
    """Submit a hosted V0 evaluation"""

    environment: str | None = Field(
        None, description="Environment name/slug or V0 TOML config path"
    )
    env_path: str | None = Field(
        None,
        description="Environment directory used for upstream resolution.",
    )
    poll_interval: float = Field(
        10.0, description="Polling interval in seconds for hosted evaluation status"
    )
    follow: bool = Field(
        False, description="Follow hosted evaluation status and stream logs until completion"
    )
    model: str | None = Field(
        None, validation_alias=AliasChoices("model", "m"), description="Inference model"
    )
    num_examples: int | None = Field(
        None,
        validation_alias=AliasChoices("num_examples", "n"),
        description="Examples per environment",
    )
    rollouts_per_example: int | None = Field(
        None,
        validation_alias=AliasChoices("rollouts_per_example", "r"),
        description="Rollouts per example",
    )
    env_args: str | None = Field(None, description="V0 load_environment arguments as JSON")
    extra_env_kwargs: str | None = Field(
        None, description="V0 post-load environment arguments as JSON"
    )
    timeout_minutes: int | None = Field(
        None, description="Timeout in minutes for hosted evaluation"
    )
    allow_sandbox_access: bool | None = Field(
        None, description="Allow sandbox read/write access for hosted evaluations"
    )
    allow_instances_access: bool | None = Field(
        None, description="Allow instance creation and management for hosted evaluations"
    )
    allow_tunnel_access: bool | None = Field(
        None, description="Allow tunnel creation and management for hosted evaluations"
    )
    custom_secrets: str | None = Field(
        None, description='Custom secrets for hosted eval as JSON (e.g. \'{"API_KEY":"xxx"}\')'
    )
    sampling_args: str | None = Field(
        None,
        description="Sampling arguments as JSON.",
    )
    eval_name: str | None = Field(None, description="Custom name for the hosted evaluation")
    max_concurrent: int | None = Field(None, description="Maximum concurrent rollouts")
    max_retries: int | None = Field(None, description="Retries per rollout")
    state_columns: list[str] | None = Field(
        None,
        description="State columns to retain.",
    )
    independent_scoring: bool | None = Field(None, description="Score rollouts independently")
    verbose: bool | None = Field(None, description="Enable verbose evaluator logs")
    header: list[str] | None = Field(
        None, description="Extra HTTP header as 'Name: Value'; repeat as needed"
    )
    api_client_type: str | None = Field(None, description="V0 model client type")
    api_base_url: str | None = Field(None, description="V0 model API base URL")
    api_key_var: str | None = Field(
        None, description="Environment variable containing the model API key"
    )


class EvalViewConfig(BaseConfig):
    """Launch the interactive evaluation viewer."""

    limit: int = Field(
        50, validation_alias=AliasChoices("limit", "n"), description="Max evaluation rows to load"
    )
    env_dir: str | None = Field(
        None,
        validation_alias=AliasChoices("env_dir", "e"),
        description="Path to environments directory",
    )
    outputs_dir: str | None = Field(
        None,
        validation_alias=AliasChoices("outputs_dir", "o"),
        description="Path to outputs directory",
    )
