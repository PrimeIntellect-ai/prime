"""Prime-owned Lab setup, sync, and doctor services."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import toml
from rich import box
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .lab_agents import (
    agent_capability,
    agent_project_skills_dirs,
    agent_user_skills_dir,
    known_agent_names,
    write_agent_native_surface,
)
from .lab_hygiene import (
    LabHygieneOptions,
    append_lab_gitignore,
    missing_lab_gitignore_patterns,
    run_lab_hygiene_preflight,
    tracked_lab_hygiene_paths,
)

VERIFIERS_REPO = "primeintellect-ai/verifiers"
VERIFIERS_REF = "7d8a522df67308327cb9b8931ce6a5873a99834a"
VERIFIERS_CONFIG_REF = "main"
DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_RETRY_DELAY_SECONDS = 1.0
LAB_CONFIG_FOLDERS = ("rl", "gepa", "eval", "sft", "opd", "fft")

SUPPORTED_AGENTS = known_agent_names()
PRIME_SKILLS_MANIFEST = ".prime-managed.json"
WORKSPACE_SKILLS_DIR = Path(".prime") / "skills"
DeprecatedConfigField = tuple[tuple[str, ...], str]
RepoTreeEntry = tuple[str, str]
DEPRECATED_CONFIG_FIELDS: tuple[DeprecatedConfigField, ...] = (
    (("trajectory_strategy",), "`trajectory_strategy` is deprecated and ignored."),
    (("trajectoryStrategy",), "`trajectoryStrategy` is deprecated and ignored."),
    (("env_file",), "`env_file` is deprecated; use `env_files`."),
    (("oversampling_factor",), "`oversampling_factor` is outdated; remove it from Lab configs."),
    (("max_async_level",), "`max_async_level` is outdated; remove it from Lab configs."),
    (("max_off_policy_steps",), "`max_off_policy_steps` is outdated; remove it from Lab configs."),
)
_REPO_TREE_CACHE: dict[tuple[str, str, int], tuple[RepoTreeEntry, ...]] = {}

Emit = Callable[[str | RenderableType], None]
Runner = Callable[[Sequence[str], Path, Emit], int]


@dataclass(frozen=True)
class SkillSource:
    """Predefined repository source for managed Lab skills."""

    repo: str
    ref: str
    path: str = "skills"


@dataclass(frozen=True)
class LabSetupOptions:
    """Options for initializing a Lab workspace."""

    skip_agents_md: bool = False
    skip_install: bool = False
    agents: tuple[str, ...] = ()


@dataclass(frozen=True)
class LabSetupResult:
    """Result from a Lab setup run."""

    exit_code: int
    workspace: Path


@dataclass(frozen=True)
class LabSyncOptions:
    """Options for refreshing Lab workspace assets."""

    agents: tuple[str, ...] = ()
    skip_docs: bool = False
    no_agent: bool = False


@dataclass(frozen=True)
class LabSyncResult:
    """Result from a Lab asset sync."""

    exit_code: int
    workspace: Path


@dataclass(frozen=True)
class LabDoctorOptions:
    """Options for checking a Lab workspace."""

    fix: bool = False


@dataclass(frozen=True)
class LabDoctorCheck:
    """One deterministic Lab workspace check."""

    name: str
    status: str
    message: str
    remediation: str = ""


@dataclass(frozen=True)
class LabDoctorResult:
    """Result from a Lab workspace check."""

    exit_code: int
    workspace: Path
    checks: tuple[LabDoctorCheck, ...]


AGENTS_MD_SRC = (
    f"https://raw.githubusercontent.com/{VERIFIERS_REPO}/{VERIFIERS_REF}/assets/lab/AGENTS.md"
)
CLAUDE_MD_SRC = (
    f"https://raw.githubusercontent.com/{VERIFIERS_REPO}/{VERIFIERS_REF}/assets/lab/CLAUDE.md"
)
ENVS_AGENTS_MD_SRC = (
    f"https://raw.githubusercontent.com/{VERIFIERS_REPO}/{VERIFIERS_REF}"
    "/assets/lab/environments/AGENTS.md"
)
SKILL_SOURCES: tuple[SkillSource, ...] = (SkillSource(repo=VERIFIERS_REPO, ref=VERIFIERS_REF),)


def run_lab_setup(passthrough_args: list[str], *, console: Console | None = None) -> int:
    """Run the Lab workspace setup workflow from the CLI."""

    console = console or Console()
    try:
        options = parse_lab_setup_args(passthrough_args)
    except SystemExit as exc:
        return int(exc.code or 0)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 2

    result = run_lab_setup_service(
        options,
        workspace=Path.cwd(),
        emit=lambda item: _emit_to_console(console, item),
    )
    return result.exit_code


def run_lab_sync(passthrough_args: list[str], *, console: Console | None = None) -> int:
    """Refresh Lab skills and local agent guidance from the CLI."""

    console = console or Console()
    try:
        options = parse_lab_sync_args(passthrough_args)
    except SystemExit as exc:
        return int(exc.code or 0)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 2

    result = run_lab_sync_service(
        options,
        workspace=Path.cwd(),
        emit=lambda item: _emit_to_console(console, item),
    )
    return result.exit_code


def run_lab_doctor(passthrough_args: list[str], *, console: Console | None = None) -> int:
    """Check a Lab workspace from the CLI."""

    console = console or Console()
    try:
        options = parse_lab_doctor_args(passthrough_args)
    except SystemExit as exc:
        return int(exc.code or 0)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 2

    result = run_lab_doctor_service(options, workspace=Path.cwd())
    _print_lab_doctor_result(result, console)
    return result.exit_code


def parse_lab_setup_args(args: list[str]) -> LabSetupOptions:
    parser = argparse.ArgumentParser(
        prog="prime lab setup",
        description="Set up a Lab workspace.",
    )
    parser.add_argument(
        "--skip-agents-md",
        action="store_true",
        help="Skip AGENTS.md, CLAUDE.md, and environments/AGENTS.md.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip uv project initialization and verifiers installation.",
    )
    parser.add_argument(
        "--agents",
        "--agent",
        dest="agents",
        help=(
            "Comma-separated coding agents to scaffold, or 'all' for diagnostics. "
            f"Supported: {', '.join((*SUPPORTED_AGENTS, 'all'))}."
        ),
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Use setup defaults without prompts.",
    )
    namespace = parser.parse_args(args)
    return LabSetupOptions(
        skip_agents_md=bool(namespace.skip_agents_md),
        skip_install=bool(namespace.skip_install),
        agents=_resolve_setup_agents(
            namespace.agents,
            no_interactive=bool(namespace.no_interactive),
        ),
    )


def parse_lab_sync_args(args: list[str]) -> LabSyncOptions:
    parser = argparse.ArgumentParser(
        prog="prime lab sync",
        description="Refresh Lab skills and local agent guidance.",
    )
    parser.add_argument(
        "--agents",
        "--agent",
        dest="agents",
        help=(
            "Comma-separated coding agents to refresh, or 'all' for diagnostics. "
            f"Supported: {', '.join((*SUPPORTED_AGENTS, 'all'))}."
        ),
    )
    parser.add_argument(
        "--skip-docs",
        action="store_true",
        help="Skip AGENTS.md, CLAUDE.md, and environments/AGENTS.md refresh.",
    )
    parser.add_argument(
        "--no-agent",
        action="store_true",
        help="Refresh shared Lab assets without configuring coding-agent skill roots.",
    )
    namespace = parser.parse_args(args)
    if namespace.agents is not None and namespace.no_agent:
        raise ValueError("--agent and --no-agent cannot be used together.")
    return LabSyncOptions(
        agents=_resolve_explicit_agents(namespace.agents) if namespace.agents is not None else (),
        skip_docs=bool(namespace.skip_docs),
        no_agent=bool(namespace.no_agent),
    )


def parse_lab_doctor_args(args: list[str]) -> LabDoctorOptions:
    parser = argparse.ArgumentParser(
        prog="prime lab doctor",
        description="Check a Lab workspace and report deterministic remediation steps.",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply safe local remediations such as creating standard dirs and gitignore entries.",
    )
    namespace = parser.parse_args(args)
    return LabDoctorOptions(fix=bool(namespace.fix))


def run_lab_setup_service(
    options: LabSetupOptions,
    *,
    workspace: Path,
    emit: Emit | None = None,
    runner: Runner | None = None,
) -> LabSetupResult:
    """Initialize a Lab workspace without leaving the caller's UI."""

    workspace = workspace.expanduser().resolve()
    emit = emit or (lambda _text: None)
    runner = runner or _run_command
    try:
        _run_lab_setup_steps(options, workspace=workspace, emit=emit, runner=runner)
    except Exception as exc:
        emit(f"Setup failed: {exc}\n")
        return LabSetupResult(exit_code=1, workspace=workspace)
    return LabSetupResult(exit_code=0, workspace=workspace)


def run_lab_sync_service(
    options: LabSyncOptions,
    *,
    workspace: Path,
    emit: Emit | None = None,
) -> LabSyncResult:
    """Refresh Lab skills and local agent guidance."""

    workspace = workspace.expanduser().resolve()
    emit = emit or (lambda _text: None)
    try:
        _run_lab_sync_steps(options, workspace=workspace, emit=emit)
    except Exception as exc:
        emit(f"Sync failed: {exc}\n")
        return LabSyncResult(exit_code=1, workspace=workspace)
    return LabSyncResult(exit_code=0, workspace=workspace)


def run_lab_doctor_service(
    options: LabDoctorOptions,
    *,
    workspace: Path,
) -> LabDoctorResult:
    """Check a Lab workspace without network or platform side effects."""

    workspace = workspace.expanduser().resolve()
    checks = _lab_doctor_checks(options, workspace)
    exit_code = 1 if any(check.status == "FAIL" for check in checks) else 0
    return LabDoctorResult(exit_code=exit_code, workspace=workspace, checks=tuple(checks))


def _run_lab_setup_steps(
    options: LabSetupOptions,
    *,
    workspace: Path,
    emit: Emit,
    runner: Runner,
) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    emit(f"Setting up Lab workspace at {workspace}\n")
    if not options.agents:
        raise RuntimeError("No coding agent configured. Pass --agent to prime lab setup.")

    if not options.skip_install:
        _ensure_uv_project(workspace, emit, runner)

    (workspace / "configs").mkdir(exist_ok=True)
    (workspace / "environments").mkdir(exist_ok=True)
    _append_gitignore(workspace)
    managed_skill_names = _sync_prime_skills(emit)
    _prepare_workspace_skill_dir(workspace, managed_skill_names, emit)
    _prepare_agent_skill_dirs(workspace, options.agents, managed_skill_names, emit)
    _report_missing_agent_requirements(options.agents, emit)
    _prepare_agent_native_surfaces(workspace, options.agents, emit)
    _sync_lab_metadata(workspace, options.agents, setup_source="prime lab setup")

    if not options.skip_agents_md:
        _sync_workspace_guidance(workspace, options.agents, emit, force=True)

    _sync_config_templates(workspace, emit)
    _copy_setup_configs(workspace, emit)
    _write_lab_docs_index(workspace, options.agents)
    emit("\n")
    emit(_post_setup_call_to_action(options))
    run_lab_hygiene_preflight(
        LabHygieneOptions(fix=True),
        workspace=workspace,
        emit=emit,
    )


def _run_lab_sync_steps(
    options: LabSyncOptions,
    *,
    workspace: Path,
    emit: Emit,
) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "configs").mkdir(exist_ok=True)
    (workspace / "environments").mkdir(exist_ok=True)
    emit(f"Syncing Lab assets in {workspace}\n")
    agents = _resolve_sync_agents(workspace, options.agents, no_agent=options.no_agent)
    guidance_agents = agents
    if not guidance_agents and options.no_agent:
        guidance_agents = _workspace_agents_from_metadata(workspace)

    managed_skill_names = _sync_prime_skills(emit)
    _prepare_workspace_skill_dir(workspace, managed_skill_names, emit)
    if agents:
        _prepare_agent_skill_dirs(workspace, agents, managed_skill_names, emit)
        _report_missing_agent_requirements(agents, emit)
        _prepare_agent_native_surfaces(workspace, agents, emit)
        _sync_lab_metadata(workspace, agents, setup_source="prime lab sync")
    else:
        reason = (
            "--no-agent"
            if options.no_agent
            else "no configured agent; pass --agent to configure one"
        )
        emit(f"Skipped coding-agent skill roots ({reason})\n")
    _sync_config_templates(workspace, emit)

    if not options.skip_docs:
        _sync_workspace_guidance(workspace, guidance_agents, emit, force=True)
        _write_lab_docs_index(workspace, guidance_agents)

    run_lab_hygiene_preflight(
        LabHygieneOptions(fix=True),
        workspace=workspace,
        emit=emit,
    )
    emit("Lab sync completed\n")


def _sync_workspace_guidance(
    workspace: Path,
    agents: tuple[str, ...],
    emit: Emit,
    *,
    force: bool = False,
) -> None:
    _download_file(AGENTS_MD_SRC, workspace / "AGENTS.md", emit, force=force, quiet=True)
    if "claude" in agents:
        _download_file(CLAUDE_MD_SRC, workspace / "CLAUDE.md", emit, force=force, quiet=True)
    _download_file(
        ENVS_AGENTS_MD_SRC,
        workspace / "environments" / "AGENTS.md",
        emit,
        force=force,
        quiet=True,
    )
    emit("Refreshed workspace guidance\n")


def _sync_prime_skills(emit: Emit) -> tuple[str, ...]:
    skills_dir = _global_prime_skills_dir()
    skills_dir.parent.mkdir(parents=True, exist_ok=True)
    manifest_skills: dict[str, dict[str, str]] = {}
    manifest: dict[str, Any] = {
        "version": 1,
        "source": {"package": "prime", "version": _prime_package_version()},
        "skills": manifest_skills,
    }
    previous_manifest = _read_prime_skills_manifest(skills_dir)
    with tempfile.TemporaryDirectory(
        prefix=".skills-staging-",
        dir=str(skills_dir.parent),
    ) as staging_dir:
        staging_skills_dir = Path(staging_dir) / "skills"
        for source in SKILL_SOURCES:
            for skill_name in _discover_skill_names(source):
                _sync_prime_skill_source(
                    source,
                    skill_name,
                    staging_skills_dir,
                    manifest_skills,
                    emit,
                )
        managed_skill_names = tuple(manifest_skills)
        _replace_prime_skills(
            skills_dir,
            staging_skills_dir,
            previous_manifest,
            managed_skill_names,
            emit,
        )
        (skills_dir / PRIME_SKILLS_MANIFEST).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        emit(f"Refreshed {skills_dir}\n")
    return managed_skill_names


def _sync_prime_skill_source(
    source: SkillSource,
    skill_name: str,
    staging_skills_dir: Path,
    manifest_skills: dict[str, dict[str, str]],
    emit: Emit,
) -> None:
    if skill_name in manifest_skills:
        existing = manifest_skills[skill_name]
        raise RuntimeError(
            f"Skill '{skill_name}' is defined by both {existing['repo']} and {source.repo}."
        )
    skill_dir = staging_skills_dir / skill_name
    source_path = f"{source.path}/{skill_name}"
    _download_repo_directory(source.repo, source.ref, source_path, skill_dir, emit)
    skill_path = skill_dir / "SKILL.md"
    if not skill_path.is_file():
        raise RuntimeError(f"Skill '{skill_name}' is missing SKILL.md.")
    manifest_skills[skill_name] = {
        "repo": source.repo,
        "ref": source.ref,
        "path": source_path,
        "sha256": _hash_directory(skill_dir),
    }


def _download_repo_directory(
    repo: str,
    ref: str,
    source_path: str,
    dest: Path,
    emit: Emit,
    *,
    force: bool = True,
    missing_ok: bool = False,
) -> None:
    normalized_source_path = _normalize_repo_path(source_path)
    tree_entries = _repo_tree_entries(repo, ref)
    prefix = f"{normalized_source_path}/" if normalized_source_path else ""
    source_exists = any(
        path == normalized_source_path and entry_type == "tree" for path, entry_type in tree_entries
    )
    file_entries = [
        (path, path.removeprefix(prefix))
        for path, entry_type in tree_entries
        if entry_type == "blob" and path.startswith(prefix)
    ]
    if not source_exists and not file_entries:
        if missing_ok:
            return
        raise RuntimeError(f"Expected a directory listing for {repo}/{source_path}.")
    for entry_path, relative_path in sorted(file_entries):
        _download_file(
            _repo_raw_url(repo, ref, entry_path),
            dest / relative_path,
            emit,
            force=force,
            quiet=True,
        )


def _hash_directory(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _discover_skill_names(source: SkillSource) -> tuple[str, ...]:
    normalized_source_path = _normalize_repo_path(source.path)
    tree_entries = _repo_tree_entries(source.repo, source.ref)
    source_exists = any(
        path == normalized_source_path and entry_type == "tree" for path, entry_type in tree_entries
    )
    if not source_exists:
        raise RuntimeError(f"Expected a directory listing for {source.repo}/{source.path}.")
    prefix = f"{normalized_source_path}/" if normalized_source_path else ""
    names = {
        relative_path
        for path, entry_type in tree_entries
        if entry_type == "tree"
        and path.startswith(prefix)
        and (relative_path := path.removeprefix(prefix))
        and "/" not in relative_path
    }
    return tuple(sorted(names))


def _replace_prime_skills(
    skills_dir: Path,
    staging_skills_dir: Path,
    previous_manifest: dict[str, Any],
    managed_skill_names: tuple[str, ...],
    emit: Emit,
) -> None:
    skills_dir.mkdir(parents=True, exist_ok=True)
    managed = set(managed_skill_names)
    previous = previous_manifest.get("skills")
    previous_names = set(previous) if isinstance(previous, dict) else set()
    for stale_name in sorted(previous_names - managed):
        stale_path = skills_dir / stale_name
        _remove_path(stale_path)
        emit(f"Warning: removed stale managed skill {stale_path}\n")
    for skill_name in managed_skill_names:
        target = skills_dir / skill_name
        _remove_path(target)
        shutil.move(str(staging_skills_dir / skill_name), str(target))


def _read_prime_skills_manifest(skills_dir: Path) -> dict[str, Any]:
    path = skills_dir / PRIME_SKILLS_MANIFEST
    try:
        loaded = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _copy_setup_configs(workspace: Path, emit: Emit) -> None:
    template_configs = _global_lab_templates_dir() / "configs"
    workspace_configs = workspace / "configs"
    if template_configs.is_dir():
        _copy_lab_config_folders(template_configs, workspace_configs)
        missing_folders = _missing_cached_lab_config_folders(template_configs)
        if missing_folders:
            _download_lab_config_folders(
                template_configs,
                emit,
                force=False,
                folders=missing_folders,
            )
            _copy_lab_config_folders(template_configs, workspace_configs)
    else:
        _download_lab_config_folders(workspace_configs, emit, force=False)
    emit(f"Prepared {workspace_configs}\n")


def _sync_config_templates(workspace: Path, emit: Emit) -> None:
    global_template_root = _global_lab_templates_dir()
    global_template_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".templates-staging-",
        dir=str(global_template_root.parent),
    ) as staging_dir:
        staging_template_root = Path(staging_dir) / "templates"
        _download_lab_config_folders(staging_template_root / "configs", emit, force=True)
        _remove_path(global_template_root)
        shutil.move(str(staging_template_root), str(global_template_root))
    emit(f"Refreshed {global_template_root}\n")

    template_root = workspace / ".prime" / "lab" / "templates"
    if template_root.resolve(strict=False) == global_template_root.resolve(strict=False):
        return
    _remove_path(template_root)
    template_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(global_template_root, template_root)
    emit(f"Refreshed {template_root}\n")


def _download_lab_config_folders(
    dest: Path,
    emit: Emit,
    *,
    force: bool,
    folders: Sequence[str] = LAB_CONFIG_FOLDERS,
) -> None:
    for folder in folders:
        _download_repo_directory(
            VERIFIERS_REPO,
            VERIFIERS_CONFIG_REF,
            f"configs/{folder}",
            dest / folder,
            emit,
            force=force,
            missing_ok=True,
        )


def _copy_lab_config_folders(source_configs: Path, dest_configs: Path) -> None:
    for folder in LAB_CONFIG_FOLDERS:
        source_folder = source_configs / folder
        if not source_folder.is_dir():
            continue
        for source_path in sorted(path for path in source_folder.rglob("*") if path.is_file()):
            relative_path = source_path.relative_to(source_configs)
            dest_path = dest_configs / relative_path
            if dest_path.exists():
                continue
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, dest_path)


def _missing_cached_lab_config_folders(source_configs: Path) -> tuple[str, ...]:
    return tuple(
        folder
        for folder in LAB_CONFIG_FOLDERS
        if not _cached_lab_config_folder_complete(source_configs, folder)
    )


def _cached_lab_config_folder_complete(source_configs: Path, folder: str) -> bool:
    source_path = f"configs/{folder}"
    expected_files = tuple(
        path.removeprefix("configs/")
        for path, entry_type in _repo_tree_entries(VERIFIERS_REPO, VERIFIERS_CONFIG_REF)
        if entry_type == "blob" and path.startswith(f"{source_path}/")
    )
    if not expected_files:
        return True
    return all((source_configs / relative_path).is_file() for relative_path in expected_files)


def _global_lab_templates_dir() -> Path:
    return Path.home() / ".prime" / "lab" / "templates"


def _prepare_workspace_skill_dir(
    workspace: Path,
    managed_skill_names: tuple[str, ...],
    emit: Emit,
) -> None:
    skills_dir = workspace / WORKSPACE_SKILLS_DIR
    global_skills_dir = _global_prime_skills_dir()
    if skills_dir.resolve(strict=False) == global_skills_dir.resolve(strict=False):
        emit(f"Prepared {skills_dir}\n")
        return
    _prepare_managed_skill_dir(
        skills_dir,
        managed_skill_names,
        emit,
        source_root=global_skills_dir,
        managed_roots=(global_skills_dir,),
        prefer_link=False,
        protect_user_owned=False,
    )
    emit(f"Prepared {skills_dir}\n")


def _prepare_agent_skill_dirs(
    workspace: Path,
    agents: tuple[str, ...],
    managed_skill_names: tuple[str, ...],
    emit: Emit,
) -> None:
    prepared_dirs: set[Path] = set()
    workspace_skills_dir = workspace / WORKSPACE_SKILLS_DIR
    global_skills_dir = _global_prime_skills_dir()
    for agent in agents:
        project_dirs = agent_project_skills_dirs(agent, workspace)
        if project_dirs:
            for skills_dir in project_dirs:
                _prepare_managed_skill_dir_once(
                    skills_dir,
                    prepared_dirs,
                    managed_skill_names,
                    emit,
                    source_root=workspace_skills_dir,
                    managed_roots=(workspace_skills_dir, global_skills_dir),
                )
            continue
        user_skills_dir = agent_user_skills_dir(agent)
        if user_skills_dir is not None:
            _prepare_managed_skill_dir_once(
                user_skills_dir,
                prepared_dirs,
                managed_skill_names,
                emit,
                source_root=global_skills_dir,
                managed_roots=(global_skills_dir,),
            )


def _prepare_managed_skill_dir_once(
    skills_dir: Path,
    prepared_dirs: set[Path],
    managed_skill_names: tuple[str, ...],
    emit: Emit,
    *,
    source_root: Path,
    managed_roots: tuple[Path, ...],
) -> None:
    resolved_dir = skills_dir.resolve(strict=False)
    if resolved_dir in prepared_dirs:
        return
    prepared_dirs.add(resolved_dir)
    _prepare_managed_skill_dir(
        skills_dir,
        managed_skill_names,
        emit,
        source_root=source_root,
        managed_roots=managed_roots,
    )
    emit(f"Prepared {skills_dir}\n")


def _prepare_managed_skill_dir(
    skills_dir: Path,
    managed_skill_names: tuple[str, ...],
    emit: Emit,
    *,
    source_root: Path,
    managed_roots: tuple[Path, ...],
    prefer_link: bool = True,
    protect_user_owned: bool = True,
) -> None:
    _remove_stale_managed_skill_links(skills_dir, managed_roots, managed_skill_names)
    for skill_name in managed_skill_names:
        source_dir = source_root / skill_name
        if not source_dir.exists():
            continue
        target_dir = skills_dir / skill_name
        if protect_user_owned and _skill_target_is_user_owned(target_dir, managed_roots):
            emit(f"Skipped {target_dir} because a user-owned skill already exists\n")
            continue
        if protect_user_owned:
            _remove_managed_skill_target(target_dir, managed_roots)
        else:
            _remove_path(target_dir)
        _safe_link_or_copy_managed_skill_dir(
            source_dir,
            target_dir,
            prefer_link=prefer_link,
        )


def _report_missing_agent_requirements(agents: tuple[str, ...], emit: Emit) -> None:
    for agent in agents:
        capability = agent_capability(agent)
        for requirement in capability.missing_requirements():
            if requirement.install_command:
                install = " ".join(requirement.install_command)
                emit(
                    f"{capability.label} requires {requirement.binary}; "
                    f"install with `{install}` and rerun sync\n"
                )
            else:
                emit(
                    f"{capability.label} requires {requirement.binary}; install it and rerun sync\n"
                )


def _prepare_agent_native_surfaces(workspace: Path, agents: tuple[str, ...], emit: Emit) -> None:
    for agent in agents:
        paths = write_agent_native_surface(workspace, agent)
        for path in paths:
            try:
                display = str(path.relative_to(workspace))
            except ValueError:
                display = str(path)
            emit(f"Prepared {display}\n")


def _lab_doctor_checks(options: LabDoctorOptions, workspace: Path) -> list[LabDoctorCheck]:
    if options.fix:
        run_lab_hygiene_preflight(LabHygieneOptions(fix=True), workspace=workspace)

    metadata_path = workspace / ".prime" / "lab.json"
    metadata = _read_lab_metadata(workspace)
    agents = _workspace_agents_from_metadata(workspace)

    checks = [
        _path_check("Workspace metadata", metadata_path, "Run prime lab setup."),
        _path_check(
            "Python project",
            workspace / "pyproject.toml",
            "Run prime lab setup or uv init before launching local workflows.",
        ),
        _path_check("Configs directory", workspace / "configs", "Run prime lab doctor --fix."),
        _path_check(
            "Environments directory",
            workspace / "environments",
            "Run prime lab doctor --fix.",
        ),
        _gitignore_check(workspace),
        _tracked_lab_git_hygiene_check(workspace),
        _config_validity_check(workspace),
        _config_deprecated_fields_check(workspace),
        _config_environment_reference_check(workspace),
        _environment_source_hygiene_check(workspace),
        _managed_skill_manifest_check(),
        _global_lab_templates_check(),
        _workspace_managed_skills_check(workspace),
        _path_check(
            "Lab templates",
            workspace / ".prime" / "lab" / "templates" / "configs" / "rl" / "gsm8k.toml",
            "Run prime lab sync.",
            warning=True,
        ),
        _path_check(
            "Lab docs index",
            workspace / ".prime" / "lab" / "docs" / "index.md",
            "Run prime lab sync.",
            warning=True,
        ),
    ]

    if agents:
        for agent in agents:
            label = agent_capability(agent).label
            agent_skill_dirs = agent_project_skills_dirs(agent, workspace) or (
                (agent_user_skills_dir(agent),) if agent_user_skills_dir(agent) else ()
            )
            for agent_skill_dir in agent_skill_dirs:
                if agent_skill_dir is not None:
                    checks.append(_agent_managed_skills_check(label, agent_skill_dir, agent))
            checks.append(_agent_native_surface_check(agent, workspace))
    else:
        checks.append(
            LabDoctorCheck(
                name="Coding agent",
                status="WARN",
                message="No primary coding agent is configured.",
                remediation="Run prime lab setup and select an installed coding agent.",
            )
        )

    if not metadata and metadata_path.exists():
        checks.append(
            LabDoctorCheck(
                name="Workspace metadata JSON",
                status="FAIL",
                message=".prime/lab.json exists but is not valid Lab metadata.",
                remediation="Regenerate it with prime lab setup.",
            )
        )

    return checks


def _managed_skill_manifest_check() -> LabDoctorCheck:
    manifest = _read_prime_skills_manifest(_global_prime_skills_dir())
    skills = manifest.get("skills")
    if isinstance(skills, dict) and skills:
        return LabDoctorCheck(
            name="Global Lab skill cache",
            status="PASS",
            message=f"{len(skills)} managed skill(s) installed.",
        )
    return LabDoctorCheck(
        name="Global Lab skill cache",
        status="WARN",
        message=f"Missing {_global_prime_skills_dir() / PRIME_SKILLS_MANIFEST}",
        remediation="Run prime lab sync.",
    )


def _global_lab_templates_check() -> LabDoctorCheck:
    path = _global_lab_templates_dir() / "configs" / "rl" / "gsm8k.toml"
    if path.is_file():
        return LabDoctorCheck(
            name="Global Lab template cache",
            status="PASS",
            message=f"Installed at {_global_lab_templates_dir()}.",
        )
    return LabDoctorCheck(
        name="Global Lab template cache",
        status="WARN",
        message=f"Missing {_global_lab_templates_dir()}",
        remediation="Run prime lab sync.",
    )


def _workspace_managed_skills_check(workspace: Path) -> LabDoctorCheck:
    skill_names = _managed_skill_names_from_manifest()
    if not skill_names:
        return LabDoctorCheck(
            name="Workspace Lab skills",
            status="WARN",
            message="No managed Lab skills are installed.",
            remediation="Run prime lab sync.",
        )
    skills_dir = workspace / WORKSPACE_SKILLS_DIR
    missing = [
        skill_name
        for skill_name in skill_names
        if not (skills_dir / skill_name).exists() and not (skills_dir / skill_name).is_symlink()
    ]
    if not missing:
        return LabDoctorCheck(
            name="Workspace Lab skills",
            status="PASS",
            message=f"{len(skill_names)} managed skill link(s) are present.",
        )
    return LabDoctorCheck(
        name="Workspace Lab skills",
        status="WARN",
        message="Missing " + ", ".join(missing[:5]),
        remediation="Run prime lab sync.",
    )


def _agent_managed_skills_check(label: str, agent_skill_dir: Path, agent: str) -> LabDoctorCheck:
    skill_names = _managed_skill_names_from_manifest()
    if not skill_names:
        return LabDoctorCheck(
            name=f"{label} skills",
            status="WARN",
            message="No managed Lab skills are installed.",
            remediation="Run prime lab sync.",
        )
    missing = [
        skill_name
        for skill_name in skill_names
        if not (agent_skill_dir / skill_name).exists()
        and not (agent_skill_dir / skill_name).is_symlink()
    ]
    if not missing:
        return LabDoctorCheck(
            name=f"{label} skills",
            status="PASS",
            message=f"{len(skill_names)} managed skill link(s) are present.",
        )
    return LabDoctorCheck(
        name=f"{label} skills",
        status="WARN",
        message="Missing " + ", ".join(missing[:5]),
        remediation=f"Run prime lab sync --agent {agent}.",
    )


def _managed_skill_names_from_manifest() -> tuple[str, ...]:
    manifest = _read_prime_skills_manifest(_global_prime_skills_dir())
    skills = manifest.get("skills")
    if not isinstance(skills, dict):
        return ()
    return tuple(str(name) for name in skills)


def _agent_native_surface_check(agent: str, workspace: Path) -> LabDoctorCheck:
    capability = agent_capability(agent)
    name = f"{capability.label} native tools"
    path_based_surfaces = {"mcp_config", "acp_mcp", "droid_mcp_config", "pi_extension"}
    if capability.status == "not_supported":
        return LabDoctorCheck(
            name=name,
            status="WARN",
            message=f"{capability.label} is not yet supported.",
            remediation=capability.unsupported_reason or "Choose a Lab-supported coding agent.",
        )
    missing = capability.missing_requirements()
    if missing:
        remediations = [
            " ".join(requirement.install_command)
            if requirement.install_command
            else f"install {requirement.binary}"
            for requirement in missing
        ]
        return LabDoctorCheck(
            name=name,
            status="WARN",
            message="Missing " + ", ".join(requirement.binary for requirement in missing),
            remediation="Install selected agent dependency: " + ", ".join(remediations) + ".",
        )
    if capability.native_surface == "codex_app_server":
        return LabDoctorCheck(
            name=name,
            status="PASS",
            message=(
                f"{capability.label} receives native Lab tools through {capability.native_surface}."
            ),
        )
    if capability.native_surface == "letta_external_tools":
        return LabDoctorCheck(
            name=name,
            status="PASS",
            message=f"{capability.label} receives native Lab tools through external tools.",
        )
    if capability.native_surface == "none":
        return LabDoctorCheck(
            name=name,
            status="PASS",
            message=f"{capability.label} native Lab tools are not scaffolded by setup yet.",
        )
    if capability.native_surface not in path_based_surfaces:
        return LabDoctorCheck(
            name=name,
            status="WARN",
            message=f"Unknown native surface type: {capability.native_surface}.",
            remediation="Update Lab doctor native-surface handling.",
        )
    expected_paths = capability.resolved_surface_paths(workspace)
    if not expected_paths:
        return LabDoctorCheck(
            name=name,
            status="WARN",
            message=f"{capability.label} declares {capability.native_surface} but no path.",
            remediation=f"Add an expected surface path or update {capability.name} setup.",
        )
    missing_paths = [path for path in expected_paths if not path.exists()]
    if not missing_paths:
        return LabDoctorCheck(
            name=name,
            status="PASS",
            message=", ".join(str(path) for path in expected_paths)
            or f"{capability.label} receives Lab tools at session start.",
        )
    return LabDoctorCheck(
        name=name,
        status="WARN",
        message="Missing " + ", ".join(str(path) for path in missing_paths),
        remediation=f"Run prime lab sync --agent {capability.name}.",
    )


def _path_check(
    name: str,
    path: Path,
    remediation: str,
    *,
    warning: bool = False,
) -> LabDoctorCheck:
    if path.exists():
        return LabDoctorCheck(name=name, status="PASS", message=str(path))
    return LabDoctorCheck(
        name=name,
        status="WARN" if warning else "FAIL",
        message=f"Missing {path}",
        remediation=remediation,
    )


def _gitignore_check(workspace: Path) -> LabDoctorCheck:
    path = workspace / ".gitignore"
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    missing = _missing_gitignore_patterns(existing)
    if not missing:
        return LabDoctorCheck(
            name="Gitignore outputs",
            status="PASS",
            message="Standard output and generated source paths are ignored.",
        )
    return LabDoctorCheck(
        name="Gitignore outputs",
        status="WARN",
        message="Missing " + ", ".join(missing),
        remediation="Run prime lab doctor --fix to add standard output ignores.",
    )


def _tracked_lab_git_hygiene_check(workspace: Path) -> LabDoctorCheck:
    tracked_paths = tracked_lab_hygiene_paths(workspace)
    if not tracked_paths:
        return LabDoctorCheck(
            name="Tracked Lab git hygiene",
            status="PASS",
            message="No generated Lab guidance or outputs are tracked.",
        )
    shown = ", ".join(tracked_paths[:5])
    suffix = "" if len(tracked_paths) <= 5 else f" and {len(tracked_paths) - 5} more"
    return LabDoctorCheck(
        name="Tracked Lab git hygiene",
        status="FAIL",
        message="Tracked generated Lab files: " + shown + suffix,
        remediation="Run git rm --cached on the generated Lab files and keep them local only.",
    )


def _config_validity_check(workspace: Path) -> LabDoctorCheck:
    configs_dir = workspace / "configs"
    if not configs_dir.is_dir():
        return LabDoctorCheck(
            name="Config TOML",
            status="FAIL",
            message="Missing configs directory.",
            remediation="Run prime lab doctor --fix to create configs/.",
        )
    config_paths = sorted(configs_dir.rglob("*.toml"))
    if not config_paths:
        return LabDoctorCheck(
            name="Config TOML",
            status="WARN",
            message="No TOML configs found.",
            remediation="Run prime lab setup or save a config copy from Lab.",
        )
    invalid: list[str] = []
    for path in config_paths:
        try:
            toml.loads(path.read_text(encoding="utf-8"))
        except (OSError, toml.TomlDecodeError):
            invalid.append(str(path.relative_to(workspace)))
    if invalid:
        return LabDoctorCheck(
            name="Config TOML",
            status="FAIL",
            message="Invalid " + ", ".join(invalid[:5]),
            remediation="Open the config in Lab or fix the TOML before launch.",
        )
    return LabDoctorCheck(
        name="Config TOML",
        status="PASS",
        message=f"{len(config_paths)} config file(s) parse cleanly.",
    )


def _config_deprecated_fields_check(workspace: Path) -> LabDoctorCheck:
    configs_dir = workspace / "configs"
    if not configs_dir.is_dir():
        return LabDoctorCheck(
            name="Config deprecated fields",
            status="FAIL",
            message="Missing configs directory.",
            remediation="Run prime lab doctor --fix to create configs/.",
        )
    config_paths = sorted(configs_dir.rglob("*.toml"))
    if not config_paths:
        return LabDoctorCheck(
            name="Config deprecated fields",
            status="WARN",
            message="No TOML configs found.",
            remediation="Run prime lab setup or save a config copy from Lab.",
        )
    findings: list[str] = []
    for path in config_paths:
        try:
            parsed = toml.loads(path.read_text(encoding="utf-8"))
        except (OSError, toml.TomlDecodeError):
            continue
        for field_path, message in _deprecated_config_fields(parsed):
            location = ".".join(field_path)
            findings.append(f"{path.relative_to(workspace)}:{location} ({message})")
    if findings:
        return LabDoctorCheck(
            name="Config deprecated fields",
            status="WARN",
            message="Deprecated " + ", ".join(findings[:5]),
            remediation="Remove deprecated config fields or replace them with the suggested names.",
        )
    return LabDoctorCheck(
        name="Config deprecated fields",
        status="PASS",
        message="No deprecated config fields found.",
    )


def _deprecated_config_fields(config: dict[str, Any]) -> list[DeprecatedConfigField]:
    findings: list[DeprecatedConfigField] = []
    for field_path, message in DEPRECATED_CONFIG_FIELDS:
        value: Any = config
        for part in field_path:
            if not isinstance(value, dict) or part not in value:
                break
            value = value[part]
        else:
            findings.append((field_path, message))
    return findings


def _config_environment_reference_check(workspace: Path) -> LabDoctorCheck:
    configs_dir = workspace / "configs"
    if not configs_dir.is_dir():
        return LabDoctorCheck(
            name="Config environment refs",
            status="FAIL",
            message="Missing configs directory.",
            remediation="Run prime lab doctor --fix to create configs/.",
        )
    config_paths = sorted(configs_dir.rglob("*.toml"))
    if not config_paths:
        return LabDoctorCheck(
            name="Config environment refs",
            status="WARN",
            message="No TOML configs found.",
            remediation="Run prime lab setup or save a config copy from Lab.",
        )
    missing_local: list[str] = []
    local_names = _local_environment_names(workspace)
    for path in config_paths:
        try:
            parsed = toml.loads(path.read_text(encoding="utf-8"))
        except (OSError, toml.TomlDecodeError):
            continue
        for ref in _config_environment_refs(parsed):
            if "/" not in ref and local_names and ref not in local_names:
                missing_local.append(f"{path.relative_to(workspace)}:{ref}")
    if missing_local:
        return LabDoctorCheck(
            name="Config environment refs",
            status="WARN",
            message="Local env not found: " + ", ".join(missing_local[:5]),
            remediation="Fix the environment id or add the local environment source.",
        )
    return LabDoctorCheck(
        name="Config environment refs",
        status="PASS",
        message="Environment references resolve as local or hosted ids.",
    )


def _config_environment_refs(config: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    environment = config.get("environment")
    if isinstance(environment, dict):
        refs.append(str(environment.get("id") or environment.get("name") or ""))
    for key in ("env", "environments"):
        value = config.get(key)
        if isinstance(value, list):
            refs.extend(_environment_refs_from_list(value, id_key="id"))
    evals = config.get("eval")
    if isinstance(evals, list):
        refs.extend(_environment_refs_from_list(evals, id_key="env_id"))
    return [ref for ref in refs if ref]


def _environment_refs_from_list(values: list[Any], *, id_key: str) -> list[str]:
    refs: list[str] = []
    for value in values:
        if isinstance(value, str):
            refs.append(_split_env_ref(value)[0])
        elif isinstance(value, dict):
            refs.append(str(value.get(id_key) or value.get("id") or value.get("name") or ""))
    return refs


def _split_env_ref(value: str) -> tuple[str, str]:
    if "@" in value:
        env_id, version = value.rsplit("@", 1)
        return env_id.strip(), version.strip()
    if ":" in value and "/" in value:
        env_id, version = value.rsplit(":", 1)
        return env_id.strip(), version.strip()
    return value.strip(), ""


def _local_environment_names(workspace: Path) -> set[str]:
    envs_dir = workspace / "environments"
    if not envs_dir.is_dir():
        return set()
    return {path.name for path in envs_dir.iterdir() if path.is_dir()}


def _environment_source_hygiene_check(workspace: Path) -> LabDoctorCheck:
    envs_dir = workspace / "environments"
    if not envs_dir.is_dir():
        return LabDoctorCheck(
            name="Environment source hygiene",
            status="FAIL",
            message="Missing environments directory.",
            remediation="Run prime lab doctor --fix to create environments/.",
        )
    generated_paths: list[str] = []
    for env_dir in sorted(path for path in envs_dir.iterdir() if path.is_dir()):
        for relative in ("outputs", "dist", "__pycache__"):
            generated = env_dir / relative
            if generated.exists():
                generated_paths.append(str(generated.relative_to(workspace)))
        generated_paths.extend(
            str(path.relative_to(workspace)) for path in sorted(env_dir.glob("*.egg-info"))
        )
        generated_paths.extend(
            str(path.relative_to(workspace)) for path in sorted(env_dir.rglob("*.pyc"))[:5]
        )
    if generated_paths:
        return LabDoctorCheck(
            name="Environment source hygiene",
            status="WARN",
            message="Generated artifacts present: " + ", ".join(generated_paths[:5]),
            remediation="Remove generated outputs before pushing environment source.",
        )
    env_count = len([path for path in envs_dir.iterdir() if path.is_dir()])
    return LabDoctorCheck(
        name="Environment source hygiene",
        status="PASS",
        message=f"{env_count} local environment source dir(s) look clean.",
    )


def _ensure_uv_project(workspace: Path, emit: Emit, runner: Runner) -> None:
    if not (workspace / "pyproject.toml").is_file():
        emit("No pyproject.toml found; running uv init\n")
        _check_command(["uv", "init"], workspace, emit, runner)
        _remove_if_exists(workspace / "main.py")
        _remove_if_exists(workspace / ".python-version")
        _append_gitignore(workspace)
    else:
        emit("Found pyproject.toml\n")

    emit("Adding verifiers dependency\n")
    _check_command(["uv", "add", "verifiers"], workspace, emit, runner)


def _post_setup_call_to_action(options: LabSetupOptions) -> Panel:
    primary_agent = options.agents[0] if options.agents else "your coding agent"
    prompt_heading = f"ask {primary_agent}"
    prompt_body = (
        "I want to train a model for <my task domain>. Propose an initial environment "
        "scaffold including relevant tools, and come up with a good method to generate "
        "a small sample synthetic dataset. Run a quick eval baseline, inspect the "
        "results, and then decide how we should iterate on refining the implementation."
    )
    prompt_text = Text(
        prompt_body,
        style="italic",
    )

    command_table = Table.grid(padding=(0, 1))
    command_table.add_row("[bold green]$[/bold green]", "prime env init my-env")
    command_table.add_row(
        "[bold green]$[/bold green]", "prime eval run my-env -m openai/gpt-5.4-nano -n 5"
    )
    command_table.add_row("[bold green]$[/bold green]", "prime eval view")
    command_table.add_row("[bold green]$[/bold green]", "prime rl run configs/rl/qwen-3-5.toml")
    command_table.add_row(
        "[bold green]$[/bold green]", "prime gepa run my-env -m openai/gpt-5.4-nano"
    )

    header_text = Text.assemble(
        ("idea -> environment -> eval -> training", "dim"),
    )

    content = Group(
        header_text,
        Panel(
            prompt_text,
            title=prompt_heading,
            border_style="magenta",
            box=box.ROUNDED,
            padding=(1, 2),
            expand=False,
        ),
        Panel(
            command_table,
            title="quick commands",
            border_style="green",
            box=box.ROUNDED,
            padding=(0, 1),
            expand=False,
        ),
    )

    return Panel(
        content,
        title="[bold white]get started[/bold white]",
        border_style="bright_blue",
        box=box.DOUBLE,
        padding=(1, 2),
        expand=False,
    )


def _emit_to_console(console: Console, item: str | RenderableType) -> None:
    if isinstance(item, str):
        console.print(item.rstrip("\n"), markup=False)
    else:
        console.print(item)


def _write_lab_docs_index(workspace: Path, agents: tuple[str, ...]) -> None:
    docs_dir = workspace / ".prime" / "lab" / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    guidance_files = [
        "- `AGENTS.md`",
        "- `environments/AGENTS.md`",
    ]
    if "claude" in agents:
        guidance_files.insert(1, "- `CLAUDE.md`")
    index = docs_dir / "index.md"
    index.write_text(
        "\n".join(
            [
                "# Lab Agent Context",
                "",
                "Use these local files and public docs as the first context sources for Lab work.",
                "",
                "## Local workspace guidance",
                "",
                *guidance_files,
                "- `~/.prime/skills/*/SKILL.md`",
                "- `.prime/lab/templates/configs/**`",
                "",
                "## Prime docs",
                "",
                "- Prime CLI: https://github.com/PrimeIntellect-ai/prime",
                "- Verifiers: https://github.com/PrimeIntellect-ai/verifiers",
                "- Environments Hub: https://app.primeintellect.ai/dashboard/environments",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _sync_lab_metadata(workspace: Path, agents: tuple[str, ...], *, setup_source: str) -> None:
    prime_dir = workspace / ".prime"
    prime_dir.mkdir(exist_ok=True)
    path = prime_dir / "lab.json"
    metadata = _read_lab_metadata(workspace)
    metadata["setup_source"] = setup_source
    metadata["choices"] = {
        "agents": list(agents),
        "primary_agent": agents[0],
        "use_multiple_agents": len(agents) > 1,
    }
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_lab_metadata(workspace: Path) -> dict[str, Any]:
    path = workspace / ".prime" / "lab.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        raw = {}
    return raw if isinstance(raw, dict) else {}


def _workspace_agents_from_metadata(workspace: Path) -> tuple[str, ...]:
    choices = _read_lab_metadata(workspace).get("choices")
    if not isinstance(choices, dict):
        return ()
    agents = choices.get("agents")
    if isinstance(agents, list):
        parsed = tuple(
            agent.strip() for agent in agents if isinstance(agent, str) and agent.strip()
        )
        if parsed:
            return parsed
    primary_agent = choices.get("primary_agent")
    if isinstance(primary_agent, str) and primary_agent.strip():
        return (primary_agent.strip(),)
    return ()


def _resolve_sync_agents(
    workspace: Path,
    agents: tuple[str, ...],
    *,
    no_agent: bool,
) -> tuple[str, ...]:
    if no_agent:
        return ()
    if agents:
        return agents
    stored_agents = _workspace_agents_from_metadata(workspace)
    if stored_agents:
        return stored_agents
    return ()


def _download_file(
    url: str,
    dest: Path,
    emit: Emit,
    *,
    force: bool = False,
    quiet: bool = False,
) -> None:
    if dest.exists() and not force:
        if not quiet:
            emit(f"{dest.name} already exists\n")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    content = _read_url(url, emit=emit)
    dest.write_bytes(content)
    if not quiet:
        emit(f"Downloaded {dest}\n")


def _download_json(url: str) -> Any:
    try:
        return json.loads(_read_url(url).decode("utf-8"))
    except (RuntimeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Failed to download {url}") from exc


def _repo_raw_url(repo: str, ref: str, source_path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{ref}/{source_path}"


def _github_tree_url(repo: str, ref: str) -> str:
    return f"https://api.github.com/repos/{repo}/git/trees/{ref}?recursive=1"


def _normalize_repo_path(path: str) -> str:
    return path.strip("/")


def _repo_tree_entries(repo: str, ref: str) -> tuple[RepoTreeEntry, ...]:
    cache_key = (repo, ref, id(_download_json))
    cached = _REPO_TREE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    payload = _download_json(_github_tree_url(repo, ref))
    if not isinstance(payload, dict) or not isinstance(payload.get("tree"), list):
        raise RuntimeError(f"Expected a repository tree for {repo}/{ref}.")
    if payload.get("truncated") is True:
        raise RuntimeError(f"Repository tree for {repo}/{ref} is too large to download.")
    entries: list[RepoTreeEntry] = []
    for entry in payload["tree"]:
        if not isinstance(entry, dict):
            continue
        path = entry.get("path")
        entry_type = entry.get("type")
        if isinstance(path, str) and isinstance(entry_type, str):
            entries.append((_normalize_repo_path(path), entry_type))
    tree_entries = tuple(entries)
    _REPO_TREE_CACHE[cache_key] = tree_entries
    return tree_entries


def _read_url(url: str, *, emit: Emit | None = None) -> bytes:
    last_exc: BaseException | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            with urlopen(url, timeout=60) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            last_exc = exc
            if attempt == DOWNLOAD_ATTEMPTS:
                break
            if emit is not None:
                emit(f"Download failed for {url}; retrying ({attempt + 1}/{DOWNLOAD_ATTEMPTS})\n")
            time.sleep(DOWNLOAD_RETRY_DELAY_SECONDS * attempt)
    raise RuntimeError(f"Failed to download {url}") from last_exc


def _prime_package_version() -> str:
    try:
        return metadata.version("prime")
    except metadata.PackageNotFoundError:
        return "unknown"


def _run_command(command: Sequence[str], cwd: Path, emit: Emit) -> int:
    process = subprocess.Popen(
        list(command),
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if process.stdout is not None:
        for line in process.stdout:
            emit(line)
    return process.wait()


def _check_command(command: Sequence[str], cwd: Path, emit: Emit, runner: Runner) -> None:
    emit(f"Running: {' '.join(command)}\n")
    code = runner(command, cwd, emit)
    if code != 0:
        raise RuntimeError(f"{' '.join(command)} exited with {code}")


def _resolve_setup_agents(value: str | None, *, no_interactive: bool) -> tuple[str, ...]:
    if value is not None:
        return _resolve_explicit_agents(value)
    if no_interactive:
        return ("codex",)
    if sys.stdin.isatty():
        return _prompt_for_agents()
    raise ValueError(
        "No --agent provided and stdin is not interactive. "
        "Pass --agent codex, --agent amp, or another supported agent."
    )


def _prompt_for_agents() -> tuple[str, ...]:
    print(f"Supported coding agents: {', '.join(SUPPORTED_AGENTS)}")
    while True:
        try:
            raw_primary = _prompt_input("Primary coding agent [codex]: ").strip()
            primary = raw_primary or "codex"
            selected = [_normalize_supported_agent(primary, allow_all=False)]
            break
        except EOFError as exc:
            raise ValueError("Agent selection was cancelled.") from exc
        except ValueError as exc:
            print(exc)

    try:
        use_multiple_raw = _prompt_input("Using multiple coding agents? [y/N]: ").strip().lower()
    except EOFError as exc:
        raise ValueError("Agent selection was cancelled.") from exc

    if use_multiple_raw in {"y", "yes"}:
        while True:
            try:
                additional_raw = _prompt_input("Additional agents (comma-separated): ").strip()
                additional_agents = _parse_agents(additional_raw) if additional_raw else []
                break
            except EOFError as exc:
                raise ValueError("Agent selection was cancelled.") from exc
            except ValueError as exc:
                print(exc)
        for agent in additional_agents:
            if agent not in selected:
                selected.append(agent)
    return tuple(selected)


def _prompt_input(prompt: str) -> str:
    return input(prompt)


def _resolve_explicit_agents(value: str) -> tuple[str, ...]:
    parsed = _parse_agents(value)
    if parsed:
        return tuple(parsed)
    raise ValueError(
        "No valid coding agents provided. Supported values: "
        + ", ".join((*SUPPORTED_AGENTS, "all"))
    )


def _parse_agents(value: str | None) -> list[str]:
    if value and value.strip().lower() == "all":
        return list(SUPPORTED_AGENTS)
    raw_agents = value.split(",") if value else []
    agents: list[str] = []
    seen: set[str] = set()
    for raw_agent in raw_agents:
        if not raw_agent.strip():
            continue
        agent = _normalize_supported_agent(raw_agent, allow_all=False)
        if agent in seen:
            continue
        seen.add(agent)
        agents.append(agent)
    return agents


def _normalize_supported_agent(raw_agent: str, *, allow_all: bool) -> str:
    raw_name = raw_agent.strip().lower()
    if allow_all and raw_name == "all":
        return raw_name
    agent = agent_capability(raw_name).name
    if agent not in SUPPORTED_AGENTS:
        raise ValueError(
            f"Unsupported coding agent '{raw_agent}'. Supported values: "
            + ", ".join((*SUPPORTED_AGENTS, "all"))
        )
    return agent


def _append_gitignore(workspace: Path) -> None:
    append_lab_gitignore(workspace)


def _missing_gitignore_patterns(existing: str) -> list[str]:
    return missing_lab_gitignore_patterns(existing)


def _global_prime_skills_dir() -> Path:
    return Path.home() / ".prime" / "skills"


def _safe_link_or_copy_managed_skill_dir(
    source: Path,
    target: Path,
    *,
    prefer_link: bool = True,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not prefer_link:
        shutil.copytree(source, target, dirs_exist_ok=True)
        (target / ".prime-managed-link").write_text(
            str(source.resolve(strict=False)),
            encoding="utf-8",
        )
        return
    try:
        target.symlink_to(os.path.relpath(source, start=target.parent), target_is_directory=True)
    except OSError:
        shutil.copytree(source, target, dirs_exist_ok=True)
        (target / ".prime-managed-link").write_text(
            str(source.resolve(strict=False)),
            encoding="utf-8",
        )


def _skill_target_is_user_owned(target: Path, managed_roots: tuple[Path, ...]) -> bool:
    if not (target.exists() or target.is_symlink()):
        return False
    return not _is_managed_skill_target(target, managed_roots)


def _remove_stale_managed_skill_links(
    skills_dir: Path,
    managed_roots: tuple[Path, ...],
    managed_skill_names: tuple[str, ...],
) -> None:
    if not skills_dir.exists():
        return
    managed = set(managed_skill_names)
    for target in skills_dir.iterdir():
        if target.name in managed:
            continue
        _remove_managed_skill_target(target, managed_roots)


def _remove_managed_skill_target(target: Path, managed_roots: tuple[Path, ...]) -> None:
    if not _is_managed_skill_target(target, managed_roots):
        return
    _remove_path(target)


def _remove_path(path: Path) -> None:
    if not (path.exists() or path.is_symlink()):
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _is_managed_skill_target(target: Path, managed_roots: tuple[Path, ...]) -> bool:
    try:
        resolved = target.resolve(strict=False)
        roots = tuple(root.resolve(strict=False) for root in managed_roots)
        if target.is_symlink() and any(
            resolved == root or root in resolved.parents for root in roots
        ):
            return True
    except OSError:
        return False

    marker = target / ".prime-managed-link"
    if not marker.is_file():
        return False
    try:
        resolved = Path(marker.read_text(encoding="utf-8").strip()).resolve(strict=False)
        roots = tuple(root.resolve(strict=False) for root in managed_roots)
    except OSError:
        return False
    return any(resolved == root or root in resolved.parents for root in roots)


def _remove_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _print_lab_doctor_result(result: LabDoctorResult, console: Console) -> None:
    table = Table(title=f"Lab workspace check: {result.workspace}")
    table.add_column("Status", no_wrap=True)
    table.add_column("Check", no_wrap=True)
    table.add_column("Message")
    table.add_column("Fix")
    for check in result.checks:
        style = {
            "PASS": "green",
            "WARN": "yellow",
            "FAIL": "red",
        }.get(check.status, "dim")
        table.add_row(
            f"[{style}]{check.status}[/{style}]",
            check.name,
            check.message,
            check.remediation,
        )
    console.print(table)


__all__ = [
    "LabDoctorCheck",
    "LabDoctorOptions",
    "LabDoctorResult",
    "LabSetupOptions",
    "LabSetupResult",
    "LabSyncOptions",
    "LabSyncResult",
    "SUPPORTED_AGENTS",
    "parse_lab_doctor_args",
    "parse_lab_setup_args",
    "parse_lab_sync_args",
    "run_lab_doctor",
    "run_lab_doctor_service",
    "run_lab_setup",
    "run_lab_setup_service",
    "run_lab_sync",
    "run_lab_sync_service",
]
