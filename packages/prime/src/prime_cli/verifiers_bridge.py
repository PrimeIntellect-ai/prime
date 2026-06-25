"""Shared verifiers command bridge logic for prime CLI commands."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import NoReturn, Optional

import click
import httpx
import toml
import typer

from prime_cli.core import Config

from .api.inference import InferenceAPIError, InferenceClient, InferencePaymentRequiredError
from .client import APIClient, APIError
from .utils.env_metadata import find_environment_metadata, get_environment_metadata
from .utils.eval_push import convert_eval_results, load_results_jsonl, push_eval_results_to_hub
from .utils.plain import get_console, is_plain_mode
from .verifiers_plugin import (
    V1_EVAL_MODULE,
    PrimeVerifiersPlugin,
    load_verifiers_prime_plugin,
    resolve_workspace_python,
)

console = get_console()

DEFAULT_MODEL = "openai/gpt-4.1-mini"
DEFAULT_ENV_DIR_PATH = "./environments"
DEFAULT_ENDPOINTS_PATH = "./configs/endpoints.toml"
PRIME_SLUG = "primeintellect"
INTERNAL_ENV_DISPLAY_HEADER = "X-Prime-Eval-Env-Display"
LEGACY_EVAL_MODULE = "verifiers.cli.commands.eval"
EVAL_PREFLIGHT_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=60.0, pool=60.0)
PROTOCOL_VERSIONS = (1, 1)
MODULE_TO_PRIME_COMMAND = {
    "verifiers.cli.commands.eval": "prime eval run",
    "verifiers.v1.cli.eval.main": "prime eval run",
    "verifiers.cli.commands.gepa": "prime gepa run",
    "verifiers.cli.commands.init": "prime env init",
    "verifiers.v1.cli.init": "prime env init",
    "verifiers.cli.commands.install": "prime env install",
    "verifiers.cli.commands.build": "prime env build",
    "verifiers.cli.commands.setup": "prime lab setup",
    "verifiers.cli.tui": "prime eval view",
}

MODULE_TO_CONSOLE_SCRIPT = {
    "verifiers.v1.cli.eval.main": "eval",
    "verifiers.v1.cli.init": "init",
}


@dataclass(frozen=True)
class ResolvedEnvironment:
    original: str
    env_name: str
    install_mode: str
    install_slug: Optional[str] = None
    upstream_slug: Optional[str] = None
    env_display_id: Optional[str] = None
    platform_slug: Optional[str] = None
    platform_url: Optional[str] = None
    recommend_push: bool = False
    push_reason: Optional[str] = None
    local_env_path: Optional[Path] = None


@dataclass(frozen=True)
class EndpointResolution:
    model: str
    base_url: str


def exec_eval_process(
    args: Sequence[str], *, cwd: Path | None = None, plain: bool = False
) -> NoReturn:
    workspace = (cwd or Path.cwd()).resolve()
    command = [resolve_workspace_python(workspace), "-m", V1_EVAL_MODULE]
    env = os.environ.copy()
    if plain:
        env.update(NO_COLOR="1", PYDANTIC_CONFIG_PLAIN="1")
    result = subprocess.run(
        [*command, "--protocol-version"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic output"
        raise click.ClickException(f"Verifiers exited with status {result.returncode}: {detail}")
    try:
        protocol = json.loads(result.stdout)
        versions = (
            protocol["protocol_version"],
            protocol["trace_schema_version"],
        )
        compatible = versions == PROTOCOL_VERSIONS and {"run", "resolve"}.issubset(
            protocol["operations"]
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        compatible = False
        versions = None
    if not compatible:
        raise click.ClickException(
            f"Unsupported Verifiers protocol {versions}; expected {PROTOCOL_VERSIONS}"
        )

    run_args = list(args)
    if not any(arg in ("-h", "--help") for arg in run_args):
        result = subprocess.run(
            [*command, "resolve", "--format", "json", *run_args],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic output"
            raise click.ClickException(
                f"Verifiers resolve exited with status {result.returncode}: {detail}"
            )
        try:
            resolved = json.loads(result.stdout)
            versions = (
                resolved["protocol_version"],
                resolved["trace_schema_version"],
            )
            compatible = (
                resolved["operation"] == "resolve"
                and versions == PROTOCOL_VERSIONS
                and isinstance(resolved["run_id"], str)
                and bool(resolved["run_id"])
                and isinstance(resolved["output_dir"], str)
                and isinstance(resolved["resume"], bool)
                and isinstance(resolved["config"], dict)
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            compatible = False
            versions = None
        if not compatible:
            raise click.ClickException(
                f"Unsupported Verifiers resolve response {versions}; expected {PROTOCOL_VERSIONS}"
            )
        if not resolved["resume"]:
            run_args.extend(("--uuid", resolved["run_id"]))

    command.extend(("run", *run_args))
    if workspace != Path.cwd():
        os.chdir(workspace)
    os.execvpe(command[0], command, env)


def is_help_request(primary_arg: str, passthrough_args: list[str]) -> bool:
    if primary_arg in ("-h", "--help"):
        return True
    return any(arg in ("-h", "--help") for arg in passthrough_args)


def _sanitize_help_text(help_text: str, module_name: str, prime_command: str) -> str:
    lines = help_text.splitlines()
    console_script = MODULE_TO_CONSOLE_SCRIPT.get(module_name)
    for idx, line in enumerate(lines):
        if line.lower().startswith("usage:"):
            suffix = line.split(":", 1)[1].strip()
            python_module_prefix = re.compile(
                rf"^python(?:\d+(?:\.\d+)?)?\s+-m\s+{re.escape(module_name)}(?:\s+|$)"
            )
            module_prefix = re.compile(rf"^{re.escape(module_name)}(?:\s+|$)")
            console_prefix = f"uv run {console_script}" if console_script else None
            match = python_module_prefix.match(suffix) or module_prefix.match(suffix)
            if console_prefix and suffix.startswith(console_prefix):
                suffix = suffix[len(console_prefix) :].lstrip()
            elif match is not None:
                suffix = suffix[match.end() :].lstrip()
            else:
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

    if console_script:
        sanitized = sanitized.replace(f"uv run {console_script}", prime_command)
        module_script = module_name.rsplit(".", 1)[-1] + ".py"
        sanitized = re.sub(
            rf"(?im)^usage: (?:{re.escape(console_script)}|{re.escape(module_script)})\b",
            f"Usage: {prime_command}",
            sanitized,
        )

    sanitized = re.sub(r"\bvf-[a-z0-9-]+\b", prime_command, sanitized)
    if prime_command in {"prime eval run", "prime gepa run"}:
        sanitized = re.sub(r"\benv_id_or_config\b", "environment", sanitized)
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


def _append_eval_options(help_text: str) -> str:
    extra_lines = [
        "  --skip-upload               V0 only: skip uploading evaluation results.",
        "  --env-path PATH             V0/hosted: path for environment metadata.",
        "  --hosted                    Run the evaluation on the platform instead of locally.",
        "  stop EVAL_ID                Cancel a running hosted evaluation.",
        "  --poll-interval FLOAT       Polling interval in seconds for hosted evaluations.",
        "  --follow                    Follow hosted evaluation logs until completion.",
        "  --timeout-minutes INTEGER   Timeout in minutes for hosted evaluations.",
        "  --allow-sandbox-access      Allow sandbox read/write access for hosted evaluations.",
        "  --allow-instances-access    "
        "Allow instance creation and management for hosted evaluations.",
        "  --allow-tunnel-access       "
        "Allow tunnel creation and management for hosted evaluations.",
        "  --custom-secrets JSON       Custom sandbox secrets for hosted evaluations.",
        "  --eval-name TEXT            Custom name for the hosted evaluation.",
    ]
    lines = help_text.rstrip("\n").splitlines()
    for extra_line in extra_lines:
        if extra_line not in lines:
            lines.append(extra_line)
    return "\n".join(lines) + "\n"


def print_eval_run_help(args: Optional[list[str]] = None, *, compatibility: bool = False) -> None:
    if not compatibility:
        exec_eval_process(args or ["--help"], plain=is_plain_mode())
    try:
        help_text = _load_help_text(
            load_verifiers_prime_plugin(console=console).eval_module,
            "prime eval run",
        )
    except Exception as exc:
        console.print(f"[red]Failed to load help for prime eval run:[/red] {exc}")
        raise typer.Exit(1) from exc
    _write_help(_append_eval_options(help_text))


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


def run_eval_view(env_dir: Optional[str], outputs_dir: Optional[str], limit: int = 50) -> None:
    from prime_lab_app import run_eval_view as run_prime_eval_view

    run_prime_eval_view(
        limit=limit,
        env_dir=env_dir or "./environments",
        outputs_dir=outputs_dir or "./outputs",
        workspace=Path.cwd(),
    )


def _parse_value_option(
    args: list[str], long_flag: str, short_flag: Optional[str]
) -> Optional[str]:
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


def _resolve_endpoint_alias(args: list[str], model: str) -> Optional[EndpointResolution]:
    endpoints_path = _parse_value_option(args, "--endpoints-path", "-e") or DEFAULT_ENDPOINTS_PATH
    try:
        from verifiers.utils.eval_utils import load_endpoints, resolve_endpoints_file
    except ImportError:
        return None

    endpoints_file = resolve_endpoints_file(endpoints_path)
    if endpoints_file is None or not endpoints_file.exists():
        return None

    try:
        endpoints = load_endpoints(str(endpoints_file))
    except (ImportError, AttributeError, OSError, ValueError):
        return None

    endpoint_group = endpoints.get(model)
    if not endpoint_group:
        return None

    endpoint = endpoint_group[0]
    return EndpointResolution(
        model=str(endpoint.get("model") or model),
        base_url=str(endpoint.get("url") or "").rstrip("/"),
    )


def _provider_base_url(provider: Optional[str]) -> Optional[str]:
    if not provider:
        return None
    try:
        from verifiers.scripts.eval import PROVIDER_CONFIGS
    except ImportError:
        return None

    provider_config = PROVIDER_CONFIGS.get(provider)
    if not provider_config:
        return None
    url = provider_config.get("url")
    if not isinstance(url, str) or not url:
        return None
    return url.rstrip("/")


def _env_dir_path_arg(args: list[str]) -> str:
    return _parse_value_option(args, "--env-dir-path", None) or DEFAULT_ENV_DIR_PATH


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


def _split_owner_and_name(slug: str) -> Optional[tuple[str, str]]:
    if "/" not in slug:
        return None
    owner, name = slug.split("/", 1)
    if not owner or not name:
        return None
    return owner, name


def _environment_url_from_slug(slug: str) -> Optional[str]:
    parts = _split_owner_and_name(slug)
    if parts is None:
        return None
    try:
        frontend_url = Config().frontend_url.rstrip("/")
    except Exception:
        frontend_url = "https://app.primeintellect.ai"
    return f"{frontend_url}/dashboard/environments/{slug}"


def _find_local_env_dir(env_name: str, env_dir_path: str) -> Optional[Path]:
    env_root = Path(env_dir_path)
    expected = env_root / env_name.replace("-", "_")
    if expected.exists() and expected.is_dir():
        return expected
    return None


def _is_valid_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value.lower())
    )


def _should_skip_directory(name: str) -> bool:
    if name.startswith("."):
        return True
    if name in {"dist", "__pycache__", "build", "outputs"}:
        return True
    if name.endswith(".egg-info"):
        return True
    return False


def _compute_local_content_hash(env_path: Path) -> Optional[str]:
    import hashlib

    if not env_path.exists() or not env_path.is_dir():
        return None

    hasher = hashlib.sha256()
    items_to_hash: list[tuple[str, Path]] = []

    for pattern in ("pyproject.toml", "*.py", "README.md"):
        for file_path in env_path.glob(pattern):
            if file_path.is_file():
                items_to_hash.append(("file", file_path))

    for subdir in sorted(env_path.iterdir(), key=lambda p: p.name):
        if not subdir.is_dir() or _should_skip_directory(subdir.name):
            continue
        items_to_hash.append(("dir", subdir))
        for file_path in sorted(subdir.rglob("*")):
            if not file_path.is_file():
                continue
            rel_parts = file_path.relative_to(env_path).parts
            if any(_should_skip_directory(part) for part in rel_parts[:-1]):
                continue
            if file_path.name.startswith("."):
                continue
            items_to_hash.append(("file", file_path))

    items_to_hash.sort(key=lambda item: item[1].relative_to(env_path).as_posix().lower())

    for item_type, path in items_to_hash:
        rel = path.relative_to(env_path).as_posix()
        if item_type == "dir":
            hasher.update(f"dir:{rel}".encode("utf-8"))
            continue
        hasher.update(f"file:{rel}".encode("utf-8"))
        try:
            with open(path, "rb") as f:
                hasher.update(f.read())
        except OSError:
            return None

    return hasher.hexdigest()


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


def _fetch_remote_env_details(
    client: APIClient, owner_slug: str, env_name: str, version: str = "latest"
) -> Optional[dict]:
    try:
        response = client.get(f"/environmentshub/{owner_slug}/{env_name}/@{version}")
        details = response.get("data", response) if isinstance(response, dict) else None
        if isinstance(details, dict):
            return details
        return {}
    except APIError as exc:
        message = str(exc).lower()
        if "404" in message or "not found" in message:
            return None
        return None


def _extract_remote_version_details(details: Optional[dict]) -> tuple[Optional[str], Optional[str]]:
    if not isinstance(details, dict):
        return None, None

    version_block = details.get("latest_version")
    if isinstance(version_block, dict):
        semantic = version_block.get("semantic_version")
        content_hash = version_block.get("content_hash") or version_block.get("sha256")
    else:
        semantic = details.get("semantic_version")
        content_hash = details.get("content_hash") or details.get("sha256")

    if not isinstance(semantic, str):
        semantic = None
    if not isinstance(content_hash, str):
        content_hash = None
    return semantic, content_hash


def _resolve_local_env_display(
    env_name: str,
    local_dir: Path,
    client: Optional[APIClient],
) -> tuple[str, Optional[str], Optional[str], bool, Optional[str]]:
    metadata = get_environment_metadata(local_dir)
    owner = metadata.get("owner") if isinstance(metadata, dict) else None
    tracked_name = metadata.get("name") if isinstance(metadata, dict) else None
    tracked_slug: Optional[str] = None
    if isinstance(owner, str) and isinstance(tracked_name, str) and owner and tracked_name:
        tracked_slug = f"{owner}/{tracked_name}"

    if tracked_slug is None:
        return f"{env_name} (local only)", None, None, True, "local_only"

    platform_url = _environment_url_from_slug(tracked_slug)
    tracking_parts = _split_owner_and_name(tracked_slug)
    remote_details = None
    if client and tracking_parts is not None:
        remote_details = _fetch_remote_env_details(client, tracking_parts[0], tracking_parts[1])

    if remote_details is None:
        return (
            f"{env_name} (local - ahead of {tracked_slug})",
            tracked_slug,
            platform_url,
            True,
            "ahead",
        )

    remote_version, remote_hash = _extract_remote_version_details(remote_details)
    local_hash = _compute_local_content_hash(local_dir)
    local_meta_hash = metadata.get("content_hash") if isinstance(metadata, dict) else None
    if local_hash is None and _is_valid_hash(local_meta_hash):
        local_hash = str(local_meta_hash)

    local_version = metadata.get("version") if isinstance(metadata, dict) else None
    if not isinstance(local_version, str):
        local_version = None

    in_sync = False
    if _is_valid_hash(local_hash) and _is_valid_hash(remote_hash):
        in_sync = str(local_hash) == str(remote_hash)
    elif local_version and remote_version:
        in_sync = local_version == remote_version

    if in_sync:
        return tracked_slug, tracked_slug, platform_url, False, None

    return (
        f"{env_name} (local - ahead of {tracked_slug})",
        tracked_slug,
        platform_url,
        True,
        "ahead",
    )


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
        upstream_slug = f"{owner}/{env_name}"
        return ResolvedEnvironment(
            original=env_reference,
            env_name=env_name,
            install_mode="remote",
            install_slug=install_slug,
            upstream_slug=upstream_slug,
            env_display_id=upstream_slug,
            platform_slug=upstream_slug,
            platform_url=_environment_url_from_slug(upstream_slug),
        )

    env_name = base_ref
    local_dir = _find_local_env_dir(env_name, env_dir_path)
    try:
        config = Config()
    except Exception:
        config = None

    client: Optional[APIClient] = None
    try:
        client = APIClient(require_auth=False)
    except Exception:
        client = None

    if local_dir is not None:
        env_display_id, platform_slug, platform_url, recommend_push, push_reason = (
            _resolve_local_env_display(
                env_name=env_name,
                local_dir=local_dir,
                client=client,
            )
        )
        upstream_slug = platform_slug if (platform_slug and not recommend_push) else None
        return ResolvedEnvironment(
            original=env_reference,
            env_name=env_name,
            install_mode="local",
            upstream_slug=upstream_slug,
            env_display_id=env_display_id,
            platform_slug=platform_slug,
            platform_url=platform_url,
            recommend_push=recommend_push,
            push_reason=push_reason,
            local_env_path=local_dir,
        )

    if config is None or client is None:
        return ResolvedEnvironment(
            original=env_reference,
            env_name=env_name,
            install_mode="none",
            env_display_id=env_name,
        )

    owner_candidates: list[tuple[str, str]] = []
    user_slug = _fetch_user_slug(client)
    if user_slug:
        owner_candidates.append(("personal", user_slug))

    team_slug = _fetch_active_team_slug(client, config.team_id)
    if team_slug and team_slug != user_slug:
        owner_candidates.append(("team", team_slug))

    found: list[tuple[str, str, dict]] = []
    for label, owner_slug in owner_candidates:
        details = _fetch_remote_env_details(client, owner_slug, env_name)
        if details is not None:
            found.append((label, owner_slug, details))

    if not found:
        official_details = _fetch_remote_env_details(client, PRIME_SLUG, env_name)
        if official_details is not None:
            found.append(("official", PRIME_SLUG, official_details))

    if not found:
        return ResolvedEnvironment(
            original=env_reference,
            env_name=env_name,
            install_mode="none",
            env_display_id=env_name,
        )

    selected_label, selected_owner = _choose_remote_owner(
        env_name, [(label, slug) for label, slug, _ in found]
    )
    install_slug = f"{selected_owner}/{env_name}"
    if version:
        install_slug = f"{install_slug}@{version}"
    upstream_slug = f"{selected_owner}/{env_name}"
    console.print(
        f"[dim]Using remote environment {selected_owner}/{env_name} ({selected_label})[/dim]"
    )
    return ResolvedEnvironment(
        original=env_reference,
        env_name=env_name,
        install_mode="remote",
        install_slug=install_slug,
        upstream_slug=upstream_slug,
        env_display_id=upstream_slug,
        platform_slug=upstream_slug,
        platform_url=_environment_url_from_slug(upstream_slug),
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
        if resolved.env_display_id:
            console.print(f"[dim]Resolved source: {resolved.env_display_id}[/dim]")
        _install_local_environment(plugin, resolved.env_name, env_dir_path)
        return resolved
    if resolved.install_mode == "remote":
        if resolved.install_slug is None:
            console.print(f"[red]Could not resolve remote slug for {env_reference}[/red]")
            raise typer.Exit(1)
        if resolved.env_display_id:
            console.print(f"[dim]Resolved source: {resolved.env_display_id}[/dim]")
        _install_remote_environment(resolved.install_slug)
        return resolved

    if not _is_installed(resolved.env_name, version=None):
        console.print(
            f"[yellow]Warning:[/yellow] No local checkout or matching remote environment found "
            f"for '{resolved.env_name}'. Continuing with installed package resolution."
        )
    return resolved


def _env_name_from_reference(env_reference: str) -> str:
    base_ref, _version = _split_version(env_reference)
    return base_ref.rsplit("/", 1)[-1]


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
    provider = _parse_value_option(args, "--provider", "-p")
    api_key_var = _parse_value_option(args, "--api-key-var", "-k")
    if api_key_var is None:
        env["PRIME_API_KEY"] = config.api_key

    if base:
        base = base.rstrip("/")
    elif provider is not None:
        base = _provider_base_url(provider) or ""
    elif endpoint := _resolve_endpoint_alias(args, model):
        return args, env, endpoint.model, endpoint.base_url
    elif configured_base:
        base = configured_base
        args.extend(["-b", base])
    else:
        console.print(
            "[red]Inference URL not configured.[/red] Check [bold]prime config view[/bold]."
        )
        raise typer.Exit(1)

    if api_key_var is None and provider is None:
        args.extend(["-k", "PRIME_API_KEY"])

    return args, env, model, base


def _validate_model(model: str, inference_base_url: str, configured_base_url: str) -> None:
    if inference_base_url != configured_base_url:
        return
    client = InferenceClient(timeout=EVAL_PREFLIGHT_TIMEOUT)
    try:
        client.retrieve_model(model)
    except httpx.TimeoutException:
        console.print(
            f"[yellow]Timed out validating model '{model}' during eval preflight.[/yellow] "
            "Continuing anyway because some thinking models take longer to warm up."
        )
        return
    except InferenceAPIError as exc:
        console.print(
            f"[red]Invalid model:[/red] {exc} \n\n"
            "[b]Use 'prime inference models' to see available models.[/b]"
        )
        raise typer.Exit(1) from exc


def _preflight_inference_billing(
    model: str, inference_base_url: str, configured_base_url: str
) -> None:
    if inference_base_url != configured_base_url:
        return

    client = InferenceClient(timeout=EVAL_PREFLIGHT_TIMEOUT)
    try:
        client.chat_completion(
            {
                "model": model,
                "messages": [{"role": "user", "content": "Reply with OK."}],
            }
        )
    except httpx.TimeoutException:
        console.print(
            f"[yellow]Timed out running the inference preflight probe for '{model}'.[/yellow] "
            "Continuing anyway because some thinking models take longer to warm up."
        )
        return
    except InferencePaymentRequiredError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


def _build_job_id(env_name: str, model: str) -> str:
    eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_uuid = str(uuid.uuid4())[:8]
    sanitized_env = env_name.replace("-", "_").replace("/", "_")
    sanitized_model = model.replace("/", "_").replace("-", "_")
    return f"{sanitized_env}_{sanitized_model}_{eval_timestamp}_{job_uuid}"


def _format_push_command(resolved: ResolvedEnvironment) -> str:
    if resolved.local_env_path is not None:
        base = f"prime env push --path {resolved.local_env_path}"
    else:
        base = "prime env push"
    if resolved.platform_slug:
        parts = _split_owner_and_name(resolved.platform_slug)
        if parts is not None:
            base += f" --owner {parts[0]}"
    return base


def _print_environment_source_footer(resolved: Optional[ResolvedEnvironment]) -> None:
    if resolved is None:
        return
    if resolved.platform_url:
        console.print(f"[dim]Environment URL: {resolved.platform_url}[/dim]")
    if resolved.recommend_push:
        if resolved.push_reason == "ahead" and resolved.platform_slug:
            console.print(
                f"[yellow]Local environment is ahead of {resolved.platform_slug}.[/yellow]"
            )
        elif resolved.push_reason == "local_only":
            console.print("[yellow]Local environment is not linked to an upstream.[/yellow]")
        else:
            console.print(
                "[yellow]Local environment differs from the current platform version.[/yellow]"
            )
        console.print(
            f"[dim]Publish the current local version with:[/dim] {_format_push_command(resolved)}"
        )


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

    legacy_eval = _has_flag(passthrough_args, "--save-results", "-s")
    # Hosted-eval sandboxes still emit the v0 CLI surface; v1 always saves results.
    if legacy_eval:
        args, env, model, base_url = _add_default_inference_and_key_args(passthrough_args, config)
        configured_base_url = (config.inference_url or "").strip().rstrip("/")
        _validate_model(model, base_url, configured_base_url)
        _preflight_inference_billing(model, base_url, configured_base_url)

        env_dir_path = _env_dir_path_arg(args)
        resolved_env = _prepare_single_environment(plugin, environment, env_dir_path)
        if resolved_env.env_display_id:
            args.extend(
                ["--header", f"{INTERNAL_ENV_DISPLAY_HEADER}: {resolved_env.env_display_id}"]
            )

        job_id = _build_job_id(resolved_env.env_name, model)
        args.extend(["--header", f"X-PI-Job-Id: {job_id}"])
        if config.team_id:
            args.extend(["--header", f"X-Prime-Team-ID: {config.team_id}"])

        console.print(f"[dim]Eval job_id: {job_id}[/dim]")
        command = plugin.build_module_command(LEGACY_EVAL_MODULE, [resolved_env.env_name, *args])
        _run_command(command, env=env)

        if skip_upload:
            _print_environment_source_footer(resolved_env)
            console.print("[dim]Skipped uploading evaluation results[/dim]")
            return

        push_eval_results_to_hub(
            env_name=resolved_env.env_name,
            model=model,
            job_id=job_id,
            env_path=Path(env_path) if env_path else None,
            upstream_slug=resolved_env.upstream_slug,
        )
        _print_environment_source_footer(resolved_env)
        return

    resume_dir = _parse_value_option(passthrough_args, "--resume", None)
    if resume_dir is not None:
        env = os.environ.copy()
        env["PRIME_API_KEY"] = config.api_key
        if config.team_id:
            env["PRIME_TEAM_ID"] = config.team_id
        command = plugin.build_module_command(plugin.eval_module, passthrough_args)
        _run_command(command, env=env)

        output_dir = Path(resume_dir)
        saved_config = toml.load(output_dir / "config.toml")
        model = saved_config["model"]
        taskset_id = saved_config["taskset"]["id"]
        job_target = _env_name_from_reference(taskset_id)
        saved_headers = saved_config.get("client", {}).get("headers", {})
        job_id = saved_headers.get("X-PI-Job-Id") or _build_job_id(job_target, model)
        display_id = saved_headers.get(INTERNAL_ENV_DISPLAY_HEADER)
        upstream_slug = display_id if isinstance(display_id, str) and "/" in display_id else None
        env_name_for_upload = job_target
        resolved_env = None
        is_config = False
    else:
        target = environment.removeprefix("@")
        is_config = _is_config_target(target)
        args = list(passthrough_args)
        env = os.environ.copy()
        env["PRIME_API_KEY"] = config.api_key
        if config.team_id:
            env["PRIME_TEAM_ID"] = config.team_id

        config_data = toml.load(target) if is_config else {}
        config_model = config_data.get("model") if isinstance(config_data, dict) else None
        model = _parse_value_option(args, "--model", "-m") or config_model or DEFAULT_MODEL
        if _parse_value_option(args, "--model", "-m") is None and not config_model:
            args.extend(["--model", model])

        configured_base_url = (config.inference_url or "").strip().rstrip("/")
        config_client = config_data.get("client") if isinstance(config_data, dict) else None
        config_base_url = config_client.get("base_url") if isinstance(config_client, dict) else None
        cli_base_url = _parse_value_option(args, "--client.base-url", None)
        base_url = cli_base_url or config_base_url or configured_base_url
        if not base_url:
            console.print(
                "[red]Inference URL not configured.[/red] Check [bold]prime config view[/bold]."
            )
            raise typer.Exit(1)
        if cli_base_url is None and not config_base_url:
            args.extend(["--client.base-url", base_url])
        dry_run_arg = _parse_value_option(args, "--dry-run", None)
        dry_run = config_data.get("dry_run") is True
        if dry_run_arg is not None:
            dry_run = dry_run_arg.lower() == "true"
        if not dry_run:
            _validate_model(model, base_url, configured_base_url)
            _preflight_inference_billing(model, base_url, configured_base_url)

        env_dir_path = _env_dir_path_arg(args)
        if _parse_value_option(args, "--env-dir-path", None) is not None:
            env_dir_index = next(
                idx
                for idx, arg in enumerate(args)
                if arg == "--env-dir-path" or arg.startswith("--env-dir-path=")
            )
            if args[env_dir_index] == "--env-dir-path":
                del args[env_dir_index : env_dir_index + 2]
            else:
                del args[env_dir_index]
        run_target = target
        upstream_slug: Optional[str] = None
        env_name_for_upload: Optional[str] = None
        resolved_env: Optional[ResolvedEnvironment] = None
        config_envs: list[tuple[str, str]] = []

        if is_config:
            taskset = config_data.get("taskset", {})
            if isinstance(taskset, dict) and isinstance(taskset.get("id"), str):
                config_envs = [(taskset["id"], env_dir_path)]
            for env_ref, ref_env_dir in config_envs:
                _prepare_single_environment(plugin, env_ref, ref_env_dir)
            run_target = f"@{target}"
        else:
            resolved_env = _prepare_single_environment(plugin, target, env_dir_path)
            run_target = resolved_env.env_name
            upstream_slug = resolved_env.upstream_slug
            env_name_for_upload = resolved_env.env_name

        job_target = env_name_for_upload
        if job_target is None and config_envs:
            job_target = _env_name_from_reference(config_envs[0][0])
        if job_target is None:
            job_target = Path(target).stem
        job_id = _build_job_id(job_target, model)
        output_dir_arg = _parse_value_option(args, "--output-dir", "-o")
        config_output_dir = config_data.get("output_dir") if isinstance(config_data, dict) else None
        configured_output_dir = output_dir_arg or config_output_dir
        output_dir = (
            Path(configured_output_dir)
            if configured_output_dir
            else Path("outputs", "evals", f"{job_target}--{model.replace('/', '--')}", job_id)
        )
        if configured_output_dir is None:
            args.extend(["--output-dir", str(output_dir)])

        headers = config_client.get("headers", {}) if isinstance(config_client, dict) else {}
        if not isinstance(headers, dict):
            console.print("[red]Error:[/red] client.headers must be a table")
            raise typer.Exit(2)
        cli_headers = _parse_value_option(args, "--client.headers", None)
        if cli_headers:
            try:
                parsed_headers = json.loads(cli_headers)
            except json.JSONDecodeError as exc:
                console.print("[red]Error:[/red] --client.headers must be valid JSON")
                raise typer.Exit(2) from exc
            if not isinstance(parsed_headers, dict):
                console.print("[red]Error:[/red] --client.headers must be a JSON object")
                raise typer.Exit(2)
            headers = {**headers, **parsed_headers}
            header_index = next(
                idx
                for idx, arg in enumerate(args)
                if arg == "--client.headers" or arg.startswith("--client.headers=")
            )
            if args[header_index] == "--client.headers":
                del args[header_index : header_index + 2]
            else:
                del args[header_index]
        headers["X-PI-Job-Id"] = job_id
        if resolved_env is not None and resolved_env.env_display_id:
            headers[INTERNAL_ENV_DISPLAY_HEADER] = resolved_env.env_display_id
        args.extend(["--client.headers", json.dumps(headers)])

        console.print(f"[dim]Eval job_id: {job_id}[/dim]")
        command = plugin.build_module_command(plugin.eval_module, [run_target, *args])
        _run_command(command, env=env)

    results_path = output_dir / "results.jsonl"
    if not results_path.exists():
        console.print("[dim]No rollout results produced.[/dim]")
        return
    results = convert_eval_results(load_results_jsonl(results_path))
    rollout_counts: dict[object, int] = {}
    for row in results:
        if "example_id" in row:
            example_id = row["example_id"]
            rollout_counts[example_id] = rollout_counts.get(example_id, 0) + 1
    rewards = [row["reward"] for row in results if isinstance(row.get("reward"), (int, float))]
    metadata = {
        "env": job_target,
        "model": model,
        "num_examples": len(rollout_counts),
        "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
    }
    rollout_totals = set(rollout_counts.values())
    if len(rollout_totals) == 1:
        metadata["rollouts_per_example"] = rollout_totals.pop()
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    if skip_upload:
        _print_environment_source_footer(resolved_env)
        console.print("[dim]Skipped uploading evaluation results[/dim]")
        return

    if is_config:
        console.print(
            "[yellow]Evaluation completed. Automatic upload is skipped for "
            "config-driven runs.[/yellow]"
        )
        return

    if resolved_env is not None and resolved_env.recommend_push:
        _print_environment_source_footer(resolved_env)
        console.print(
            "[yellow]Evaluation completed. Automatic upload is skipped until the local "
            "environment is published.[/yellow]"
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
        _print_environment_source_footer(resolved_env)
        console.print(
            "[dim]No upstream environment found. "
            "Skipped uploading evaluation results to platform.\n"
            "Use `prime env push` to set an upstream, or use `--env-path` to specify the "
            "correct environment path.[/dim]"
        )
        return

    if resolved_env is not None and resolved_env.platform_url:
        console.print(f"[dim]Environment URL: {resolved_env.platform_url}[/dim]")

    try:
        push_eval_results_to_hub(
            env_name=upload_env_name,
            model=model,
            job_id=job_id,
            env_path=Path(env_path) if env_path else None,
            upstream_slug=upstream_slug,
            output_dir=output_dir,
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
    env_dir_path = _env_dir_path_arg(args)

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
