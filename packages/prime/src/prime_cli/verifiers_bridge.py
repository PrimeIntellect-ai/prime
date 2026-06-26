"""Shared verifiers command bridge logic for prime CLI commands."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import NoReturn, Optional, cast

import toml
import typer

from prime_cli.core import Config

from .client import APIClient, APIError
from .utils.env_metadata import get_environment_metadata
from .utils.plain import get_console

console = get_console()

DEFAULT_MODEL = "openai/gpt-4.1-mini"
DEFAULT_ENV_DIR_PATH = "./environments"
DEFAULT_ENDPOINTS_PATH = "./configs/endpoints.toml"
PRIME_SLUG = "primeintellect"


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


def _venv_python(venv_root: Path) -> Path:
    if os.name == "nt":
        return venv_root / "Scripts" / "python.exe"
    return venv_root / "bin" / "python"


@lru_cache(maxsize=32)
def _python_can_import_verifiers(python_executable: str, cwd: str) -> bool:
    probe = "; ".join(
        (
            "import importlib.util",
            "spec = importlib.util.find_spec('verifiers')",
            "raise SystemExit(0 if spec else 1)",
        )
    )
    result = subprocess.run(
        [python_executable, "-c", probe],
        cwd=cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def resolve_workspace_python(cwd: Path | None = None) -> str:
    """Prefer a nearby project interpreter that can import Verifiers."""
    workspace = (cwd or Path.cwd()).resolve()
    candidates = [
        *(
            _venv_python(directory / ".venv")
            for directory in [workspace, *workspace.parents]
            if (directory / "pyproject.toml").is_file()
        ),
        *(
            _venv_python(Path(value))
            for value in (
                os.environ.get("UV_PROJECT_ENVIRONMENT"),
                os.environ.get("VIRTUAL_ENV"),
            )
            if value
        ),
    ]
    for candidate in candidates:
        if candidate.exists() and _python_can_import_verifiers(str(candidate), str(workspace)):
            return str(candidate)
    return sys.executable


def _load_verifiers_cli_modules() -> Mapping[str, str]:
    return cast(Mapping[str, str], getattr(import_module("verifiers.cli"), "CLI_MODULES"))


def build_verifiers_command(name: str, args: Sequence[str] = ()) -> list[str]:
    module = _load_verifiers_cli_modules()[name]
    return [resolve_workspace_python(), "-m", module, *args]


def exec_verifiers_process(name: str, args: Sequence[str], *, plain: bool = False) -> NoReturn:
    """Replace Prime with a Verifiers-owned CLI process."""
    command = build_verifiers_command(name, args)
    env = os.environ.copy()
    if plain:
        env.update(NO_COLOR="1", PYDANTIC_CONFIG_PLAIN="1")
    os.execvpe(command[0], command, env)


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


def _install_local_environment(env_name: str, env_dir_path: str) -> None:
    env_path = Path(env_dir_path) / env_name.replace("-", "_").lower()
    command = [
        "uv",
        "pip",
        "install",
        "--python",
        resolve_workspace_python(),
        "-e",
        str(env_path),
    ]
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


def _prepare_single_environment(env_reference: str, env_dir_path: str) -> ResolvedEnvironment:
    resolved = _resolve_environment_reference(env_reference, env_dir_path)
    if resolved.install_mode == "local":
        console.print(f"[dim]Using local environment '{resolved.env_name}'[/dim]")
        if resolved.env_display_id:
            console.print(f"[dim]Resolved source: {resolved.env_display_id}[/dim]")
        _install_local_environment(resolved.env_name, env_dir_path)
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


def run_gepa_passthrough(environment_or_config: str, passthrough_args: list[str]) -> None:
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
            _prepare_single_environment(config_env[0], config_env[1])
    else:
        resolved = _prepare_single_environment(environment_or_config, env_dir_path)
        run_target = resolved.env_name

    command = build_verifiers_command("gepa", [run_target, *args])
    _run_command(command, env=env)
