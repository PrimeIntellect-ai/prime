import argparse
import inspect
import json
import re
import time
from functools import wraps
from pathlib import Path
from typing import Any, Optional

import typer
from click.core import ParameterSource
from prime_evals import EvalsAPIError, EvalsClient, InvalidEvaluationError
from rich.progress import Progress
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
from ..utils.display import get_eval_viewer_url
from ..utils.env_metadata import find_environment_metadata
from ..utils.eval_push import load_results_jsonl
from ..utils.hosted_eval import (
    EvalStatus,
    HostedEvalConfig,
    clean_logs,
    get_new_log_lines,
)
from ..utils.v1_results import is_v1_eval_dir
from ..verifiers_bridge import (
    DEFAULT_ENV_DIR_PATH,
    DEFAULT_MODEL,
    _parse_value_option,
    _partition_eval_args,
    _resolve_environment_reference,
    _split_owner_and_name,
    is_help_request,
    print_eval_run_help,
    run_eval_passthrough,
    run_eval_view,
)

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
HOSTED_RUN_DEFAULT_NUM_EXAMPLES = 5
HOSTED_RUN_DEFAULT_ROLLOUTS_PER_EXAMPLE = 3
HOSTED_LOGS_RATE_LIMIT_THRESHOLD = 3
HOSTED_LOGS_RATE_LIMIT_WAIT_SECONDS = 30
HOSTED_LOGS_RETRY_WAIT_SECONDS = 10
HOSTED_LOGS_STATUS_UPDATE_EVERY_POLLS = 6
EVAL_TABLE_MAX_TEXT_WIDTH = 30
EVAL_RUN_EXAMPLE_COMMAND = "prime eval run gsm8k -n 10"
EVAL_HOSTED_LABEL = "HOSTED"
EVAL_LOCAL_LABEL = "LOCAL"
# Legacy verifiers config fields/flags are accepted through the parser only so
# Prime can reject them with the hosted-specific unsupported-option message.
HOSTED_EVAL_CONFIG_EXTRA_FIELDS = {
    "debug",
    "timeout_minutes",
    "allow_sandbox_access",
    "allow_instances_access",
    "allow_tunnel_access",
    "eval_name",
}
HOSTED_LEGACY_UNSUPPORTED_FLAGS = {"--debug", "--tui", "-u"}
HOSTED_EVAL_CONFIG_FIELD_TYPES: dict[str, tuple[type[Any], str]] = {
    "env_dir_path": (str, "a non-empty string"),
    "num_examples": (int, "an integer"),
    "rollouts_per_example": (int, "an integer"),
    "timeout_minutes": (int, "an integer"),
    "allow_sandbox_access": (bool, "a boolean"),
    "allow_instances_access": (bool, "a boolean"),
    "allow_tunnel_access": (bool, "a boolean"),
    "max_tokens": (int, "an integer"),
    "max_concurrent": (int, "an integer"),
    "max_retries": (int, "an integer"),
    "independent_scoring": (bool, "a boolean"),
    "verbose": (bool, "a boolean"),
    "api_client_type": (str, "a non-empty string"),
    "api_base_url": (str, "a non-empty string"),
    "api_key_var": (str, "a non-empty string"),
    "eval_name": (str, "a non-empty string"),
}
HOSTED_SUPPORTED_VERIFIERS_FIELDS = {
    "api_base_url",
    "api_client_type",
    "api_key_var",
    "env_args",
    "env_dir_path",
    "extra_env_kwargs",
    "header",
    "independent_scoring",
    "max_concurrent",
    "verbose",
    "max_retries",
    "max_tokens",
    "model",
    "num_examples",
    "rollouts_per_example",
    "sampling_args",
    "state_columns",
    "temperature",
}
HOSTED_SUPPORTED_TOML_FIELDS = {
    "allow_instances_access",
    "allow_sandbox_access",
    "allow_tunnel_access",
    "api_base_url",
    "api_client_type",
    "api_key_var",
    "endpoint_id",
    "endpoints_path",
    "env_id",
    "env_args",
    "env_dir_path",
    "eval_name",
    "extra_env_kwargs",
    "header",
    "headers",
    "independent_scoring",
    "max_concurrent",
    "verbose",
    "max_retries",
    "max_tokens",
    "model",
    "num_examples",
    "rollouts_per_example",
    "sampling_args",
    "state_columns",
    "temperature",
    "timeout_minutes",
}


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


def _validate_output_format(output: str, allowed: list[str]) -> None:
    if output not in allowed:
        console.print(f"[red]Error:[/red] output must be one of: {', '.join(allowed)}")
        raise typer.Exit(1)


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
        raise typer.Exit(1) from exc

    if type(parsed) is not dict:
        console.print(f"[red]Error:[/red] {option_name} must be a JSON object")
        raise typer.Exit(1)

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
            raise typer.Exit(1)

    return parsed


def _validate_json_object_field(merged: dict[str, Any], field_name: str) -> None:
    field_value = merged.get(field_name)
    if field_value is None:
        return
    if not isinstance(field_value, dict):
        console.print(f"[red]Error:[/red] hosted eval config `{field_name}` must be a TOML table")
        raise typer.Exit(1)

    try:
        json.dumps(field_value)
    except (TypeError, ValueError) as exc:
        console.print(
            "[red]Error:[/red] hosted eval config "
            f"`{field_name}` must contain only JSON-serializable values"
        )
        raise typer.Exit(1) from exc


def _coerce_hosted_headers(raw: dict[str, Any]) -> list[str] | None:
    from verifiers.cli.commands.eval import build_extra_headers

    try:
        normalized = build_extra_headers(raw)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    if not normalized:
        return None
    return [f"{name}: {value}" for name, value in normalized.items()]


def _parse_verifiers_eval_namespace(
    environment: str, passthrough_args: list[str], sampling_args: Optional[str]
) -> tuple[argparse.Namespace, set[str], dict[str, str]]:
    from verifiers.cli.commands.eval import build_parser

    argv = [environment, *passthrough_args]
    if sampling_args is not None:
        argv.extend(["--sampling-args", sampling_args])

    explicit_parser = build_parser()
    option_names_by_dest = {}
    for action in explicit_parser._actions:
        if action.option_strings:
            action.default = argparse.SUPPRESS
            option_names_by_dest[action.dest] = next(
                (option for option in action.option_strings if option.startswith("--")),
                action.option_strings[0],
            )

    try:
        parsed = build_parser().parse_args(argv)
        explicit = explicit_parser.parse_args(argv)
    except SystemExit as exc:
        raise typer.Exit(exc.code) from exc

    provided_dests = {dest for dest, value in vars(explicit).items() if dest != "env_id_or_config"}
    return parsed, provided_dests, option_names_by_dest


def _reject_unsupported_hosted_verifiers_args(
    provided_dests: set[str], option_names_by_dest: dict[str, str]
) -> None:
    unsupported_flags = [
        option_name
        for dest, option_name in option_names_by_dest.items()
        if dest in provided_dests and dest not in HOSTED_SUPPORTED_VERIFIERS_FIELDS
    ]
    if not unsupported_flags:
        return

    console.print(
        "[red]Error:[/red] hosted eval CLI does not support: "
        + ", ".join(f"`{flag}`" for flag in unsupported_flags)
    )
    raise typer.Exit(1)


def _reject_legacy_unsupported_hosted_flags(passthrough_args: list[str]) -> None:
    unsupported_flags = []
    for arg in passthrough_args:
        flag = arg.split("=", 1)[0]
        if flag in HOSTED_LEGACY_UNSUPPORTED_FLAGS and flag not in unsupported_flags:
            unsupported_flags.append(flag)

    if not unsupported_flags:
        return

    console.print(
        "[red]Error:[/red] hosted eval CLI does not support: "
        + ", ".join(f"`{flag}`" for flag in unsupported_flags)
    )
    raise typer.Exit(1)


def _freeze_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze_json_value(nested)) for key, nested in value.items()))
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _validate_hosted_config_field(
    merged: dict[str, Any], field_name: str, expected_type: type[Any], description: str
) -> None:
    field_value = merged.get(field_name)
    if field_value is None:
        return
    if type(field_value) is not expected_type:
        console.print(f"[red]Error:[/red] `{field_name}` must be {description}")
        raise typer.Exit(1)
    if expected_type is str and not field_value:
        console.print(f"[red]Error:[/red] `{field_name}` must be {description}")
        raise typer.Exit(1)


def _resolve_hosted_config_model(raw_config: dict[str, Any], config_path: Path) -> str:
    raw_endpoint_id = raw_config.get("endpoint_id")
    raw_model = raw_config.get("model")

    if raw_endpoint_id is not None and raw_model is not None:
        console.print(
            "[red]Error:[/red] hosted eval config cannot set both `endpoint_id` and `model`"
        )
        raise typer.Exit(1)

    if raw_endpoint_id is None:
        if raw_model is None:
            return DEFAULT_MODEL
        if type(raw_model) is not str or not raw_model:
            console.print("[red]Error:[/red] `model` must be a non-empty string")
            raise typer.Exit(1)
        return raw_model

    if type(raw_endpoint_id) is not str or not raw_endpoint_id:
        console.print("[red]Error:[/red] `endpoint_id` must be a non-empty string")
        raise typer.Exit(1)

    endpoints_path = raw_config.get("endpoints_path", "./configs/endpoints.toml")
    if type(endpoints_path) is not str or not endpoints_path:
        console.print("[red]Error:[/red] `endpoints_path` must be a non-empty string")
        raise typer.Exit(1)

    endpoints_path_obj = Path(endpoints_path)
    if "endpoints_path" in raw_config and not endpoints_path_obj.is_absolute():
        endpoints_path = str((config_path.parent / endpoints_path_obj).resolve())

    try:
        from verifiers.utils.eval_utils import load_endpoints, resolve_endpoints_file
    except ImportError as exc:
        console.print(
            "[red]Error:[/red] verifiers is required to resolve `endpoint_id`. "
            "Install the `verifiers` package or use `model` instead."
        )
        raise typer.Exit(1) from exc

    resolved_endpoints_file = resolve_endpoints_file(endpoints_path)
    if resolved_endpoints_file is None or resolved_endpoints_file.suffix != ".toml":
        console.print(
            "[red]Error:[/red] `endpoint_id` requires an endpoints.toml registry "
            "via `endpoints_path`"
        )
        raise typer.Exit(1)

    endpoints = load_endpoints(endpoints_path)
    if raw_endpoint_id not in endpoints:
        console.print(
            f"[red]Error:[/red] endpoint_id '{raw_endpoint_id}' not found in {endpoints_path}"
        )
        raise typer.Exit(1)

    endpoint_group = endpoints[raw_endpoint_id]
    endpoint_models = {entry["model"] for entry in endpoint_group}
    if len(endpoint_models) != 1:
        console.print(
            f"[red]Error:[/red] endpoint_id '{raw_endpoint_id}' resolves to multiple models: "
            f"{sorted(endpoint_models)}"
        )
        raise typer.Exit(1)

    return endpoint_group[0]["model"]


def _validate_single_hosted_eval_config(
    merged: dict[str, Any], config_path: Path
) -> dict[str, Any]:
    from verifiers.cli.commands.eval import merge_sampling_args

    unsupported_fields = sorted(
        field_name for field_name in merged if field_name not in HOSTED_SUPPORTED_TOML_FIELDS
    )
    if unsupported_fields:
        console.print(
            "[red]Error:[/red] hosted eval config does not support: "
            + ", ".join(f"`{field}`" for field in unsupported_fields)
        )
        raise typer.Exit(1)

    env_id = merged.get("env_id")
    if type(env_id) is not str or not env_id:
        console.print("[red]Error:[/red] hosted eval config requires a non-empty `env_id`")
        raise typer.Exit(1)

    _validate_json_object_field(merged, "env_args")
    _validate_json_object_field(merged, "sampling_args")
    _validate_json_object_field(merged, "extra_env_kwargs")
    state_columns = merged.get("state_columns")
    if state_columns is not None and (
        not isinstance(state_columns, list)
        or any(type(item) is not str or not item for item in state_columns)
    ):
        console.print("[red]Error:[/red] `state_columns` must be a list of non-empty strings")
        raise typer.Exit(1)

    headers = _coerce_hosted_headers(merged)
    merged.pop("header", None)
    merged.pop("headers", None)
    if headers is not None:
        merged["headers"] = headers

    for field_name, (expected_type, description) in HOSTED_EVAL_CONFIG_FIELD_TYPES.items():
        _validate_hosted_config_field(merged, field_name, expected_type, description)

    temperature = merged.get("temperature")
    if temperature is not None and type(temperature) not in {int, float}:
        console.print("[red]Error:[/red] `temperature` must be a number")
        raise typer.Exit(1)
    if type(temperature) is int:
        merged["temperature"] = float(temperature)

    merged["sampling_args"] = (
        merge_sampling_args(
            merged.get("sampling_args"),
            max_tokens=merged.pop("max_tokens", None),
            temperature=merged.pop("temperature", None),
            prefer_existing_keys=True,
        )
        or None
    )

    merged["model"] = _resolve_hosted_config_model(merged, config_path)
    return merged


def _load_hosted_eval_configs(config_path_str: str) -> list[dict[str, Any]]:
    from verifiers.utils.eval_utils import load_toml_config

    config_path = Path(config_path_str)
    try:
        loaded_configs = load_toml_config(
            config_path,
            extra_valid_fields=HOSTED_EVAL_CONFIG_EXTRA_FIELDS,
        )
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    return [
        _validate_single_hosted_eval_config(dict(config), config_path) for config in loaded_configs
    ]


def _fetch_eval_status(client: APIClient, eval_id: str) -> dict[str, Any]:
    return client.get(f"/evaluations/{eval_id}")


def _fetch_logs(client: APIClient, eval_id: str) -> str:
    response = client.get(f"/hosted-evaluations/{eval_id}/logs")
    return response.get("logs") or ""


def _build_hosted_evaluation_payload(config: HostedEvalConfig) -> dict[str, Any]:
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
        "environment_ids": [config.environment_id],
        "inference_model": config.inference_model,
        "eval_config": eval_config,
    }
    if config.name:
        payload["name"] = config.name

    return payload


def _create_hosted_evaluations(
    config: HostedEvalConfig, environment_ids: Optional[list[str]] = None
) -> dict[str, Any]:
    client = APIClient()
    payload = _build_hosted_evaluation_payload(config)

    if environment_ids is not None:
        payload["environment_ids"] = environment_ids

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
        raise typer.Exit(1) from exc

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
                    raise typer.Exit(1)
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
    except typer.Exit:
        raise
    except APIError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc


def _resolve_hosted_environment(
    environment: str,
    *,
    env_dir_path: Optional[str],
    env_path: Optional[str],
) -> tuple[str, str]:
    resolved = _resolve_environment_reference(environment, env_dir_path or DEFAULT_ENV_DIR_PATH)

    platform_slug = resolved.platform_slug or resolved.upstream_slug
    if platform_slug is None and env_path:
        metadata = find_environment_metadata(
            env_name=resolved.env_name,
            env_path=Path(env_path),
            module_name=resolved.env_name.replace("-", "_"),
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
        raise typer.Exit(1)

    parts = _split_owner_and_name(platform_slug)
    if parts is None:
        console.print(f"[red]Error:[/red] invalid environment slug: {platform_slug}")
        raise typer.Exit(1)

    owner, env_name = parts
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
            raise typer.Exit(1) from exc
        raise

    details = response.get("data", response)
    environment_id = details.get("id")
    if not environment_id:
        console.print(f"[red]Error:[/red] could not resolve environment id for {platform_slug}")
        raise typer.Exit(1)

    if resolved.install_mode == "local" and resolved.recommend_push:
        console.print(
            "[yellow]Local environment code differs from the latest published version of "
            f"{platform_slug}.[/yellow]"
        )
    console.print(
        f"[dim]Hosted evaluations always use the latest published version of {platform_slug}.[/dim]"
    )
    console.print(f"[dim]Using hosted environment {platform_slug}@latest[/dim]")
    return platform_slug, str(environment_id)


@subcommands_app.command("list", epilog=LIST_EVALS_JSON_HELP)
@handle_errors
def list_evals(
    output: str = typer.Option("table", "--output", "-o", help="table|json"),
    num: int = typer.Option(20, "--num", "-n", help="Items per page"),
    page: int = typer.Option(1, "--page", "-p", help="Page number"),
    env: Optional[str] = typer.Option(
        None,
        "--env",
        "--env-name",
        "-e",
        help="Filter by environment (e.g., 'gsm8k' or 'owner/gsm8k')",
    ),
) -> None:
    """List evaluations."""
    _validate_output_format(output, ["table", "json"])

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
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print(
            "[yellow]Response may contain invalid data. "
            "Try --output json to see raw response.[/yellow]"
        )
        raise typer.Exit(1)


@subcommands_app.command("get", epilog=EVAL_DETAIL_JSON_HELP)
@handle_errors
def get_eval(
    eval_id: str = typer.Argument(..., help="The ID of the evaluation to retrieve"),
    output: str = typer.Option("json", "--output", "-o", help="json|pretty"),
) -> None:
    _validate_output_format(output, ["json", "pretty"])

    api_client = APIClient()
    client = EvalsClient(api_client)
    data = client.get_evaluation(eval_id)
    format_output(data, output)


@subcommands_app.command("samples", epilog=EVAL_SAMPLES_JSON_HELP)
@handle_errors
def get_samples(
    eval_id: str = typer.Argument(..., help="The ID of the evaluation"),
    page: int = typer.Option(1, "--page", "-p", help="Page number"),
    num: int = typer.Option(100, "--num", "-n", help="Items per page"),
    output: str = typer.Option("json", "--output", "-o", help="json|pretty"),
) -> None:
    _validate_output_format(output, ["json", "pretty"])

    api_client = APIClient()
    client = EvalsClient(api_client)
    data = client.get_samples(eval_id, page=page, limit=num)
    format_output(data, output)


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

    candidate = path.parent if path.is_file() else path
    if is_v1_eval_dir(candidate):
        raise ValueError(
            f"'{candidate}' is a v1 eval run (config.toml + results.jsonl). Pushing v1 eval "
            "results to the platform isn't supported yet — the platform backend isn't v1-aware. "
            "Inspect them locally with `prime eval view`."
        )

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


@subcommands_app.command("view")
def view_cmd(
    limit: int = typer.Option(50, "--limit", "-n", help="Max evaluation rows to load"),
    env_dir: Optional[str] = typer.Option(
        None, "--env-dir", "-e", help="Path to environments directory"
    ),
    outputs_dir: Optional[str] = typer.Option(
        None, "--outputs-dir", "-o", help="Path to outputs directory"
    ),
) -> None:
    """Launch the interactive evaluation viewer."""
    if limit < 1:
        console.print("[red]Error:[/red] --limit must be at least 1")
        raise typer.Exit(1)
    run_eval_view(env_dir=env_dir, outputs_dir=outputs_dir, limit=limit)


@subcommands_app.command("tui")
def tui_cmd(
    _limit: int = typer.Option(
        50, "--limit", "-n", help="Deprecated; use `prime eval view --limit`."
    ),
    _env_dir: Optional[str] = typer.Option(
        None, "--env-dir", "-e", help="Deprecated; use `prime eval view --env-dir`."
    ),
    _outputs_dir: Optional[str] = typer.Option(
        None, "--outputs-dir", "-o", help="Deprecated; use `prime eval view --outputs-dir`."
    ),
) -> None:
    """Deprecated alias for the evaluation viewer."""
    console.print("[yellow]Deprecated:[/yellow] `prime eval tui` has moved. Use `prime eval view`.")
    raise typer.Exit(1)


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
    output: str = typer.Option("pretty", "--output", "-o", help="json|pretty"),
    is_public: bool = typer.Option(
        False,
        "--public",
        help="Make the pushed evaluation public. Evaluations are private by default.",
    ),
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

            if output == "json":
                output_data_as_json({"results": results}, console)

            if success_count < len(eval_dirs):
                raise typer.Exit(1)

            return

        result_eval_id = _push_single_eval(config_path, env_id, run_id, eval_id, is_public, name)

        if output == "json":
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
        "Run evaluations or manage results (list, get, push, samples).\n\n"
        "By default, 'prime eval <environment>' runs 'prime eval run <environment>'."
    ),
    no_args_is_help=True,
)

app.add_typer(subcommands_app, name="")


@app.command("logs", no_args_is_help=True)
def logs_cmd(
    eval_id: str = typer.Argument(..., help="Evaluation id to get logs for"),
    tail: int = typer.Option(
        HOSTED_LOGS_DEFAULT_TAIL_LINES,
        "--tail",
        "-n",
        help="Number of lines to show",
    ),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
    poll_interval: float = typer.Option(
        HOSTED_LOGS_DEFAULT_POLL_INTERVAL_SECONDS,
        "--poll-interval",
        help="Polling interval in seconds when following logs",
    ),
) -> None:
    """Get logs for a hosted evaluation."""
    _display_logs(eval_id, tail, follow, poll_interval=poll_interval)


@app.command("stop", no_args_is_help=True)
def stop_cmd(
    eval_id: str = typer.Argument(..., help="Evaluation id to stop"),
) -> None:
    """Stop a running hosted evaluation."""
    try:
        client = APIClient()
        result = client.patch(f"/hosted-evaluations/{eval_id}/cancel")
        message = result.get("message") or f"Evaluation {eval_id} cancelled."
        console.print(f"[green]✓ {message}[/green]")
        console.print(f"[dim]View results:[/dim] {get_eval_viewer_url(eval_id)}")
    except APIError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc


def _hosted_eval_config_from_run_args(
    environment: str,
    passthrough_args: list[str],
    *,
    sampling_args: Optional[str],
    eval_name: Optional[str],
    timeout_minutes: Optional[int],
    allow_sandbox_access: bool,
    allow_instances_access: bool,
    allow_tunnel_access: bool,
    custom_secrets: Optional[str],
) -> HostedEvalConfig:
    """Build a ``HostedEvalConfig`` from the same parsed run config the local v1 path consumes.

    Hosted submission is gated until the platform backend understands the v1 eval format; this keeps
    the request/payload machinery wired and surfaces config errors early."""
    parsed = _partition_eval_args(passthrough_args)
    sampling = dict(parsed.sampling)
    if sampling_args:
        sampling.update(json.loads(sampling_args))

    def _count(flag: str, default: int) -> int:
        raw = _parse_value_option(parsed.forwarded, flag, None)
        return int(raw) if raw is not None else default

    return HostedEvalConfig(
        environment_id=environment,
        inference_model=parsed.model or DEFAULT_MODEL,
        num_examples=_count("--num-tasks", HOSTED_RUN_DEFAULT_NUM_EXAMPLES),
        rollouts_per_example=_count("--num-rollouts", HOSTED_RUN_DEFAULT_ROLLOUTS_PER_EXAMPLE),
        name=eval_name,
        timeout_minutes=timeout_minutes,
        allow_sandbox_access=allow_sandbox_access,
        allow_instances_access=allow_instances_access,
        allow_tunnel_access=allow_tunnel_access,
        custom_secrets=_parse_string_map_option(custom_secrets, "--custom-secrets"),
        sampling_args=sampling or None,
    )


@app.command(
    "run",
    help="Run an evaluation with API models (default provider = Prime Inference)",
    no_args_is_help=True,
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
        "help_option_names": [],
    },
)
def run_eval_cmd(
    ctx: typer.Context,
    environment: Optional[str] = typer.Argument(
        None,
        help="Environment name/slug or TOML config path",
    ),
    skip_upload: bool = typer.Option(
        False,
        "--skip-upload",
        help="Skip uploading results to Prime Evals Hub (results are uploaded by default)",
    ),
    env_path: Optional[str] = typer.Option(
        None,
        "--env-path",
        help=(
            "Path to the environment directory "
            "(used to locate .prime/.env-metadata.json for upstream resolution)"
        ),
    ),
    hosted: bool = typer.Option(
        False,
        "--hosted",
        help="Run the evaluation on the platform instead of locally",
    ),
    poll_interval: float = typer.Option(
        HOSTED_RUN_DEFAULT_POLL_INTERVAL_SECONDS,
        "--poll-interval",
        help="Polling interval in seconds for hosted evaluation status",
    ),
    follow: bool = typer.Option(
        False,
        "--follow",
        help="Follow hosted evaluation status and stream logs until completion",
    ),
    timeout_minutes: Optional[int] = typer.Option(
        None,
        "--timeout-minutes",
        help="Timeout in minutes for hosted evaluation",
    ),
    allow_sandbox_access: bool = typer.Option(
        False,
        "--allow-sandbox-access",
        help="Allow sandbox read/write access for hosted evaluations",
    ),
    allow_instances_access: bool = typer.Option(
        False,
        "--allow-instances-access",
        help="Allow instance creation and management for hosted evaluations",
    ),
    allow_tunnel_access: bool = typer.Option(
        False,
        "--allow-tunnel-access",
        help="Allow tunnel creation and management for hosted evaluations",
    ),
    custom_secrets: Optional[str] = typer.Option(
        None,
        "--custom-secrets",
        help='Custom secrets for hosted eval as JSON (e.g. \'{"API_KEY":"xxx"}\')',
    ),
    sampling_args: Optional[str] = typer.Option(
        None,
        "--sampling-args",
        help=(
            "Sampling args as JSON for local or hosted evals. "
            'Example: {"temperature": 0.7, "extra_body": {"provider": {"order": ["azure"]}}}'
        ),
    ),
    eval_name: Optional[str] = typer.Option(
        None,
        "--eval-name",
        help="Custom name for the hosted evaluation",
    ),
) -> None:
    """Run an evaluation with local-first environment resolution."""
    passthrough_args = list(ctx.args)

    if is_help_request(environment or "", passthrough_args):
        print_eval_run_help()
        raise typer.Exit(0)

    if environment is None:
        console.print("[red]Error:[/red] Missing argument 'ENVIRONMENT'.")
        console.print(f"[dim]Example: {EVAL_RUN_EXAMPLE_COMMAND}[/dim]")
        raise typer.Exit(2)

    if environment.startswith("-"):
        console.print("[red]Error:[/red] Environment/config must be the first argument.")
        console.print(f"[dim]Example: {EVAL_RUN_EXAMPLE_COMMAND}[/dim]")
        raise typer.Exit(2)

    poll_interval_was_provided = (
        ctx.get_parameter_source("poll_interval") == ParameterSource.COMMANDLINE
    )
    local_passthrough_args = list(passthrough_args)
    if sampling_args is not None:
        local_passthrough_args.extend(["--sampling-args", sampling_args])

    if not hosted:
        hosted_only_args = {
            "--follow": follow,
            "--poll-interval": poll_interval_was_provided,
            "--timeout-minutes": timeout_minutes is not None,
            "--allow-sandbox-access": allow_sandbox_access,
            "--allow-instances-access": allow_instances_access,
            "--allow-tunnel-access": allow_tunnel_access,
            "--custom-secrets": custom_secrets is not None,
            "--eval-name": eval_name is not None,
        }
        used_hosted_only_args = [flag for flag, used in hosted_only_args.items() if used]
        if used_hosted_only_args:
            console.print(
                "[red]Error:[/red] hosted-only options require `--hosted`: "
                + ", ".join(used_hosted_only_args)
            )
            raise typer.Exit(1)

    if hosted:
        # Build the hosted request from the same parsed run config the local v1 path uses, then
        # gate submission: the platform backend isn't v1-aware yet, so hosted v1 evals aren't
        # supported. The HostedEvalConfig / payload / polling machinery stays in place for when the
        # platform learns the v1 eval format.
        hosted_config = _hosted_eval_config_from_run_args(
            environment,
            passthrough_args,
            sampling_args=sampling_args,
            eval_name=eval_name,
            timeout_minutes=timeout_minutes,
            allow_sandbox_access=allow_sandbox_access,
            allow_instances_access=allow_instances_access,
            allow_tunnel_access=allow_tunnel_access,
            custom_secrets=custom_secrets,
        )
        raise NotImplementedError(
            "Hosted v1 evaluations are not supported yet: the platform backend isn't v1-aware "
            f"(parsed request for model {hosted_config.inference_model!r}). "
            "Run locally with `prime eval run <env>` (without --hosted)."
        )

    run_eval_passthrough(
        environment=environment,
        passthrough_args=local_passthrough_args,
        skip_upload=skip_upload,
        env_path=env_path,
    )
