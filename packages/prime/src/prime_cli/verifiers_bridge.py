"""Shared verifiers command bridge logic for prime CLI commands."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import toml
import typer
from rich.console import Console

from prime_cli.core import Config

from .api.inference import InferenceAPIError, InferenceClient
from .client import APIClient, APIError
from .utils.env_metadata import find_environment_metadata
from .utils.eval_push import push_eval_results_to_hub
from .verifiers_plugin import PrimeVerifiersPlugin, load_verifiers_prime_plugin

console = Console()

DEFAULT_MODEL = "openai/gpt-4.1-mini"
DEFAULT_ENV_DIR_PATH = "./environments"
MODULE_TO_PRIME_COMMAND = {
    "verifiers.cli.commands.eval": "prime eval run",
    "verifiers.cli.commands.gepa": "prime gepa run",
    "verifiers.cli.commands.init": "prime env init",
    "verifiers.cli.commands.install": "prime env install",
    "verifiers.cli.commands.build": "prime env build",
    "verifiers.cli.commands.setup": "prime lab setup",
    "verifiers.cli.tui": "prime eval tui",
}


@dataclass(frozen=True)
class ResolvedEnvironment:
    original: str
    env_name: str
    install_mode: str
    install_slug: Optional[str] = None
    upstream_slug: Optional[str] = None


def is_help_request(primary_arg: str, passthrough_args: list[str]) -> bool:
    if primary_arg in ("-h", "--help"):
        return True
    return any(arg in ("-h", "--help") for arg in passthrough_args)


def _sanitize_help_text(help_text: str, module_name: str, prime_command: str) -> str:
    lines = help_text.splitlines()
    for idx, line in enumerate(lines):
        if line.lower().startswith("usage:"):
            suffix = line.split(":", 1)[1].strip()
            parts = suffix.split(maxsplit=1)
            suffix = parts[1] if len(parts) > 1 else ""
            lines[idx] = f"Usage: {prime_command}{(' ' + suffix) if suffix else ''}"
            break

    sanitized = "\n".join(lines)
    alias_map = dict(MODULE_TO_PRIME_COMMAND)
    alias_map[module_name] = prime_command
    for module_alias, command_alias in sorted(
        alias_map.items(), key=lambda pair: len(pair[0]), reverse=True
    ):
        sanitized = sanitized.replace(f"python -m {module_alias}", command_alias)
        sanitized = sanitized.replace(module_alias, command_alias)

    sanitized = re.sub(r"\bvf-[a-z0-9-]+\b", prime_command, sanitized)
    return sanitized.rstrip() + "\n"


def _load_help_text(module_name: str, prime_command: str) -> str:
    plugin = load_verifiers_prime_plugin(console=console)
    command = plugin.build_module_command(module_name, ["--help"])
    result = subprocess.run(command, capture_output=True, text=True)
    help_text = result.stdout
    if result.stderr:
        if help_text and not help_text.endswith("\n"):
            help_text += "\n"
        help_text += result.stderr

    if not help_text.strip():
        raise RuntimeError(f"Unable to load help text from {module_name}")

    return _sanitize_help_text(help_text, module_name, prime_command)


def _write_help(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def print_eval_run_help() -> None:
    try:
        help_text = _load_help_text(
            load_verifiers_prime_plugin(console=console).eval_module,
            "prime eval run",
        )
    except Exception as exc:
        console.print(f"[red]Failed to load help for prime eval run:[/red] {exc}")
        raise typer.Exit(1) from exc
    _write_help(help_text)
    _write_help(
        "\nPrime-only options:\n"
        "  --skip-upload        Skip uploading evaluation results to the platform.\n"
        "  --env-path PATH      Explicit path for upstream environment metadata.\n"
    )


def print_gepa_run_help() -> None:
    try:
        help_text = _load_help_text(
            load_verifiers_prime_plugin(console=console).gepa_module,
            "prime gepa run",
        )
    except Exception as exc:
        console.print(f"[red]Failed to load help for prime gepa run:[/red] {exc}")
        raise typer.Exit(1) from exc
    _write_help(help_text)


def print_env_init_help() -> None:
    try:
        help_text = _load_help_text(
            load_verifiers_prime_plugin(console=console).init_module,
            "prime env init",
        )
    except Exception as exc:
        console.print(f"[red]Failed to load help for prime env init:[/red] {exc}")
        raise typer.Exit(1) from exc
    _write_help(help_text)


def print_env_build_help() -> None:
    try:
        help_text = _load_help_text(
            load_verifiers_prime_plugin(console=console).build_module,
            "prime env build",
        )
    except Exception as exc:
        console.print(f"[red]Failed to load help for prime env build:[/red] {exc}")
        raise typer.Exit(1) from exc
    _write_help(help_text)


def print_lab_setup_help() -> None:
    try:
        help_text = _load_help_text(
            load_verifiers_prime_plugin(console=console).setup_module,
            "prime lab setup",
        )
    except Exception as exc:
        console.print(f"[red]Failed to load help for prime lab setup:[/red] {exc}")
        raise typer.Exit(1) from exc
    _write_help(help_text)
    _write_help(
        "\nPrime-only options:\n"
        "  --agents TEXT       Comma-separated coding agents to scaffold.\n"
        "  --no-interactive    Disable interactive coding-agent prompts.\n"
    )


def run_eval_tui(env_dir: Optional[str], outputs_dir: Optional[str]) -> None:
    plugin = load_verifiers_prime_plugin(console=console)
    env = os.environ.copy()
    env["VF_ENV_DIR"] = env_dir or "./environments"
    env["VF_OUTPUTS_DIR"] = outputs_dir or "./outputs"
    command = plugin.build_module_command(plugin.tui_module)
    _run_command(command, env=env)


def _parse_value_option(args: list[str], long_flag: str, short_flag: str) -> Optional[str]:
    for idx, arg in enumerate(args):
        if arg == long_flag or arg == short_flag:
            if idx + 1 < len(args):
                return args[idx + 1]
            return None
        if arg.startswith(f"{long_flag}="):
            return arg.split("=", 1)[1]
        if short_flag and arg.startswith(short_flag) and len(arg) > len(short_flag):
            return arg[len(short_flag) :]
    return None


def _has_flag(args: list[str], long_flag: str, short_flag: str) -> bool:
    for arg in args:
        if arg == long_flag or arg == short_flag:
            return True
        if arg.startswith(f"{long_flag}="):
            return True
    return False


def _is_config_target(raw: str) -> bool:
    if raw.endswith(".toml"):
        return True
    path = Path(raw)
    return path.is_file() and path.suffix == ".toml"


def _split_version(ref: str) -> tuple[str, Optional[str]]:
    if "@" not in ref:
        return ref, None
    base, version = ref.rsplit("@", 1)
    return base, version


def _is_slug_reference(base_ref: str) -> bool:
    if "/" not in base_ref:
        return False
    if base_ref.startswith("./") or base_ref.startswith("../") or base_ref.startswith("/"):
        return False
    if base_ref.endswith(".toml"):
        return False
    return True


def _find_local_env_dir(env_name: str, env_dir_path: str) -> Optional[Path]:
    env_root = Path(env_dir_path)
    expected = env_root / env_name.replace("-", "_")
    if expected.exists() and expected.is_dir():
        return expected
    return None


def _fetch_user_slug(client: APIClient) -> Optional[str]:
    try:
        response = client.get("/user/whoami")
    except APIError:
        return None

    data = response.get("data") if isinstance(response, dict) else None
    if not isinstance(data, dict):
        return None

    slug = data.get("slug")
    if isinstance(slug, str) and slug:
        return slug
    return None


def _fetch_active_team_slug(client: APIClient, active_team_id: Optional[str]) -> Optional[str]:
    if not active_team_id:
        return None

    try:
        response = client.get("/user/teams")
    except APIError:
        return None

    teams = response.get("data") if isinstance(response, dict) else None
    if not isinstance(teams, list):
        return None

    active_team_str = str(active_team_id)
    for team in teams:
        if not isinstance(team, dict):
            continue
        if str(team.get("teamId")) != active_team_str:
            continue
        slug = team.get("slug")
        if isinstance(slug, str) and slug:
            return slug

    return None


def _remote_env_exists(client: APIClient, owner_slug: str, env_name: str) -> bool:
    try:
        client.get(f"/environmentshub/{owner_slug}/{env_name}/@latest")
        return True
    except APIError as exc:
        message = str(exc).lower()
        if "404" in message or "not found" in message:
            return False
        return False


def _choose_remote_owner(env_name: str, candidates: list[tuple[str, str]]) -> tuple[str, str]:
    if len(candidates) == 1:
        return candidates[0]

    if not sys.stdin.isatty():
        selected = candidates[0]
        console.print(
            "[yellow]Warning:[/yellow] Multiple remote owners matched "
            f"'{env_name}'. Non-interactive mode selected {selected[1]}."
        )
        return selected

    console.print(f"[cyan]Multiple remote environments found for '{env_name}':[/cyan]")
    for idx, (label, slug) in enumerate(candidates, start=1):
        console.print(f"  [cyan]({idx})[/cyan] {slug} [dim]({label})[/dim]")

    default_idx = 1
    while True:
        selection = typer.prompt("Select owner", type=int, default=default_idx)
        if 1 <= selection <= len(candidates):
            return candidates[selection - 1]
        console.print(f"[red]Invalid selection.[/red] Enter 1-{len(candidates)}.")


def _resolve_environment_reference(env_reference: str, env_dir_path: str) -> ResolvedEnvironment:
    base_ref, version = _split_version(env_reference)

    if _is_slug_reference(base_ref):
        parts = base_ref.split("/", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            console.print(f"[red]Invalid environment reference: {env_reference}[/red]")
            raise typer.Exit(1)
        owner, env_name = parts
        install_slug = f"{owner}/{env_name}"
        if version:
            install_slug = f"{install_slug}@{version}"
        return ResolvedEnvironment(
            original=env_reference,
            env_name=env_name,
            install_mode="remote",
            install_slug=install_slug,
            upstream_slug=f"{owner}/{env_name}",
        )

    env_name = base_ref
    local_dir = _find_local_env_dir(env_name, env_dir_path)
    if local_dir is not None:
        return ResolvedEnvironment(
            original=env_reference,
            env_name=env_name,
            install_mode="local",
        )

    try:
        config = Config()
        client = APIClient(require_auth=False)
    except Exception:
        return ResolvedEnvironment(
            original=env_reference,
            env_name=env_name,
            install_mode="none",
        )

    owner_candidates: list[tuple[str, str]] = []
    user_slug = _fetch_user_slug(client)
    if user_slug:
        owner_candidates.append(("personal", user_slug))

    team_slug = _fetch_active_team_slug(client, config.team_id)
    if team_slug and team_slug != user_slug:
        owner_candidates.append(("team", team_slug))

    found: list[tuple[str, str]] = []
    for label, owner_slug in owner_candidates:
        if _remote_env_exists(client, owner_slug, env_name):
            found.append((label, owner_slug))

    if not found:
        return ResolvedEnvironment(
            original=env_reference,
            env_name=env_name,
            install_mode="none",
        )

    selected_label, selected_owner = _choose_remote_owner(env_name, found)
    install_slug = f"{selected_owner}/{env_name}"
    if version:
        install_slug = f"{install_slug}@{version}"
    console.print(
        f"[dim]Using remote environment {selected_owner}/{env_name} ({selected_label})[/dim]"
    )
    return ResolvedEnvironment(
        original=env_reference,
        env_name=env_name,
        install_mode="remote",
        install_slug=install_slug,
        upstream_slug=f"{selected_owner}/{env_name}",
    )


def _run_command(command: list[str], env: Optional[dict[str, str]] = None) -> None:
    result = subprocess.run(command, env=env)
    if result.returncode != 0:
        raise typer.Exit(result.returncode)


def _install_local_environment(
    plugin: PrimeVerifiersPlugin, env_name: str, env_dir_path: str
) -> None:
    command = plugin.build_module_command(
        plugin.install_module,
        [env_name, "--path", env_dir_path],
    )
    _run_command(command)


def _is_installed(env_name: str, version: Optional[str]) -> bool:
    from .commands.env import _is_environment_installed

    return _is_environment_installed(env_name, version)


def _install_remote_environment(env_slug: str) -> None:
    from .commands.env import _install_single_environment

    slug_no_version, version = _split_version(env_slug)
    if "/" not in slug_no_version:
        console.print(f"[red]Invalid remote environment slug: {env_slug}[/red]")
        raise typer.Exit(1)
    _, env_name = slug_no_version.split("/", 1)

    if _is_installed(env_name, version):
        return

    if not _install_single_environment(env_slug):
        raise typer.Exit(1)


def _prepare_single_environment(
    plugin: PrimeVerifiersPlugin, env_reference: str, env_dir_path: str
) -> ResolvedEnvironment:
    resolved = _resolve_environment_reference(env_reference, env_dir_path)
    if resolved.install_mode == "local":
        console.print(f"[dim]Using local environment '{resolved.env_name}'[/dim]")
        _install_local_environment(plugin, resolved.env_name, env_dir_path)
        return resolved
    if resolved.install_mode == "remote":
        if resolved.install_slug is None:
            console.print(f"[red]Could not resolve remote slug for {env_reference}[/red]")
            raise typer.Exit(1)
        _install_remote_environment(resolved.install_slug)
        return resolved

    if not _is_installed(resolved.env_name, version=None):
        console.print(
            f"[yellow]Warning:[/yellow] No local checkout or matching remote environment found "
            f"for '{resolved.env_name}'. Continuing with installed package resolution."
        )
    return resolved


def _collect_eval_config_envs(config_path: Path, fallback_env_dir: str) -> list[tuple[str, str]]:
    try:
        raw = toml.load(config_path)
    except Exception as exc:
        console.print(
            f"[yellow]Warning:[/yellow] Could not parse eval config {config_path}: {exc}. "
            "Skipping pre-install."
        )
        return []

    if not isinstance(raw, dict):
        return []

    eval_entries = raw.get("eval")
    if not isinstance(eval_entries, list):
        return []

    global_env_dir = raw.get("env_dir_path")
    if not isinstance(global_env_dir, str):
        global_env_dir = fallback_env_dir

    resolved: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for entry in eval_entries:
        if not isinstance(entry, dict):
            continue
        env_id = entry.get("env_id")
        if not isinstance(env_id, str) or not env_id:
            continue
        env_dir_path = entry.get("env_dir_path")
        if not isinstance(env_dir_path, str):
            env_dir_path = global_env_dir
        key = (env_id, env_dir_path)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(key)
    return resolved


def _collect_gepa_config_env(config_path: Path, fallback_env_dir: str) -> Optional[tuple[str, str]]:
    try:
        raw = toml.load(config_path)
    except Exception as exc:
        console.print(
            f"[yellow]Warning:[/yellow] Could not parse GEPA config {config_path}: {exc}. "
            "Skipping pre-install."
        )
        return None

    if not isinstance(raw, dict):
        return None

    env_table = raw.get("env")
    if not isinstance(env_table, dict):
        return None

    env_id = env_table.get("env_id")
    if not isinstance(env_id, str) or not env_id:
        return None

    env_dir_path = raw.get("env_dir_path")
    if not isinstance(env_dir_path, str):
        env_dir_path = fallback_env_dir
    return (env_id, env_dir_path)


def _add_default_inference_and_key_args(
    passthrough_args: list[str], config: Config
) -> tuple[list[str], dict[str, str], str, str]:
    args = list(passthrough_args)
    env = os.environ.copy()

    model = _parse_value_option(args, "--model", "-m") or DEFAULT_MODEL
    configured_base = (config.inference_url or "").strip().rstrip("/")
    base = _parse_value_option(args, "--api-base-url", "-b")
    if base:
        base = base.rstrip("/")
    elif configured_base:
        base = configured_base
        args.extend(["-b", base])
    else:
        console.print(
            "[red]Inference URL not configured.[/red] Check [bold]prime config view[/bold]."
        )
        raise typer.Exit(1)

    api_key_var = _parse_value_option(args, "--api-key-var", "-k")
    if api_key_var is None:
        env["PRIME_API_KEY"] = config.api_key
        args.extend(["-k", "PRIME_API_KEY"])

    return args, env, model, base


def _validate_model(model: str, inference_base_url: str, configured_base_url: str) -> None:
    if inference_base_url != configured_base_url:
        return
    client = InferenceClient()
    try:
        client.retrieve_model(model)
    except InferenceAPIError as exc:
        console.print(
            f"[red]Invalid model:[/red] {exc} \n\n"
            "[b]Use 'prime inference models' to see available models.[/b]"
        )
        raise typer.Exit(1) from exc


def _build_job_id(env_name: str, model: str) -> str:
    eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_uuid = str(uuid.uuid4())[:8]
    sanitized_env = env_name.replace("-", "_").replace("/", "_")
    sanitized_model = model.replace("/", "_").replace("-", "_")
    return f"{sanitized_env}_{sanitized_model}_{eval_timestamp}_{job_uuid}"


def run_eval_passthrough(
    environment: str,
    passthrough_args: list[str],
    *,
    skip_upload: bool,
    env_path: Optional[str],
) -> None:
    plugin = load_verifiers_prime_plugin(console=console)
    config = Config()

    if not config.api_key:
        console.print(
            "[red]No API key configured.[/red] "
            "Run [bold]prime login[/bold] or [bold]prime config set-api-key[/bold]."
        )
        raise typer.Exit(1)

    args, env, model, base_url = _add_default_inference_and_key_args(passthrough_args, config)
    _validate_model(model, base_url, (config.inference_url or "").strip().rstrip("/"))

    env_dir_path = _parse_value_option(args, "--env-dir-path", "-p") or DEFAULT_ENV_DIR_PATH
    run_target = environment
    upstream_slug: Optional[str] = None
    env_name_for_upload: Optional[str] = None

    if _is_config_target(environment):
        for env_ref, ref_env_dir in _collect_eval_config_envs(Path(environment), env_dir_path):
            _prepare_single_environment(plugin, env_ref, ref_env_dir)
    else:
        resolved = _prepare_single_environment(plugin, environment, env_dir_path)
        run_target = resolved.env_name
        upstream_slug = resolved.upstream_slug
        env_name_for_upload = resolved.env_name

    if not skip_upload and not _has_flag(args, "--save-results", "-s"):
        args.append("-s")

    job_target = env_name_for_upload or Path(environment).stem
    job_id = _build_job_id(job_target, model)
    args.extend(["--header", f"X-PI-Job-Id: {job_id}"])

    if config.team_id:
        args.extend(["--header", f"X-Prime-Team-ID: {config.team_id}"])

    console.print(f"[dim]Eval job_id: {job_id}[/dim]")
    command = plugin.build_module_command(plugin.eval_module, [run_target, *args])
    _run_command(command, env=env)

    if skip_upload:
        console.print("[dim]Skipped uploading evaluation results[/dim]")
        return

    if _is_config_target(environment):
        console.print(
            "[yellow]Evaluation completed. Automatic upload is skipped for "
            "config-driven runs.[/yellow]"
        )
        return

    upload_env_name = env_name_for_upload or environment
    if upstream_slug is None:
        check_path = Path(env_path) if env_path else Path.cwd()
        metadata = find_environment_metadata(
            env_name=upload_env_name,
            env_path=check_path,
            module_name=upload_env_name.replace("-", "_"),
        )
        if metadata and metadata.get("owner") and metadata.get("name"):
            upstream_slug = f"{metadata.get('owner')}/{metadata.get('name')}"
            console.print(f"[dim]Using upstream environment {upstream_slug}[/dim]")

    if upstream_slug is None:
        console.print(
            "[dim]No upstream environment found. "
            "Skipped uploading evaluation results to platform.\n"
            "Use `prime env push` to set an upstream, or use `--env-path` to specify the "
            "correct environment path.[/dim]"
        )
        return

    try:
        push_eval_results_to_hub(
            env_name=upload_env_name,
            model=model,
            job_id=job_id,
            env_path=Path(env_path) if env_path else None,
            upstream_slug=upstream_slug,
        )
    except Exception as exc:
        console.print(f"[red]Failed to push results to hub:[/red] {exc}")
        console.print("[yellow]Evaluation completed but results were not pushed.[/yellow]")
        raise typer.Exit(1) from exc


def run_gepa_passthrough(environment_or_config: str, passthrough_args: list[str]) -> None:
    plugin = load_verifiers_prime_plugin(console=console)
    config = Config()

    if not config.api_key:
        console.print(
            "[red]No API key configured.[/red] "
            "Run [bold]prime login[/bold] or [bold]prime config set-api-key[/bold]."
        )
        raise typer.Exit(1)

    args, env, _model, _base_url = _add_default_inference_and_key_args(passthrough_args, config)
    env_dir_path = _parse_value_option(args, "--env-dir-path", "-p") or DEFAULT_ENV_DIR_PATH

    run_target = environment_or_config
    if _is_config_target(environment_or_config):
        config_env = _collect_gepa_config_env(Path(environment_or_config), env_dir_path)
        if config_env is not None:
            _prepare_single_environment(plugin, config_env[0], config_env[1])
    else:
        resolved = _prepare_single_environment(plugin, environment_or_config, env_dir_path)
        run_target = resolved.env_name

    command = plugin.build_module_command(plugin.gepa_module, [run_target, *args])
    _run_command(command, env=env)
