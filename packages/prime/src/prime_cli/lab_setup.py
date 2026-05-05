"""Prime-owned Lab setup, sync, and doctor services."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import metadata, resources
from pathlib import Path
from typing import Any

import toml
from rich.console import Console
from rich.table import Table

from .lab_agents import (
    agent_capability,
    agent_user_skills_dir,
    known_agent_names,
    write_agent_native_surface,
)

PRIME_RL_REPO = "primeintellect-ai/prime-rl"
PRIME_RL_REF = "main"
PRIME_RL_INSTALL_SCRIPT_REF = "main"

LAB_SKILLS = (
    "create-environments",
    "browse-environments",
    "review-environments",
    "evaluate-environments",
    "optimize-with-environments",
    "train-with-environments",
    "brainstorm",
)
LAB_WIDGETS_SKILL = "lab-widgets"
PRIME_MANAGED_SKILLS = (*LAB_SKILLS, LAB_WIDGETS_SKILL)
SUPPORTED_AGENTS = known_agent_names()
LAB_GITIGNORE_PATTERNS = (
    "./outputs",
    "./environments/*/outputs",
    "./environments/*/dist",
    "./environments/*/*.egg-info",
    "./environments/*/__pycache__",
    "*.pyc",
)
CONFIG_TEMPLATE_PATHS = (
    "configs/endpoints.toml",
    "configs/gepa/base.toml",
    "configs/gepa/wordle.toml",
    "configs/eval/minimal.toml",
    "configs/eval/multi-env.toml",
    "configs/rl/alphabet-sort.toml",
    "configs/rl/gsm8k.toml",
    "configs/rl/math-python.toml",
    "configs/rl/reverse-text.toml",
    "configs/rl/wiki-search.toml",
    "configs/rl/wordle.toml",
)
PRIME_SKILLS_MANIFEST = ".prime-managed.json"

Emit = Callable[[str], None]
Runner = Callable[[Sequence[str], Path, Emit], int]


@dataclass(frozen=True)
class LabSetupOptions:
    """Options for initializing a Lab workspace."""

    prime_rl: bool = False
    skip_agents_md: bool = False
    skip_install: bool = False
    agents: tuple[str, ...] = ("codex",)


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
        emit=lambda text: console.print(text.rstrip("\n"), markup=False),
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
        emit=lambda text: console.print(text.rstrip("\n"), markup=False),
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
        "--prime-rl",
        action="store_true",
        help="Install prime-rl and download prime-rl configs.",
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
        help="Comma-separated coding agents to scaffold, or 'all' for diagnostics.",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Use setup defaults without prompts.",
    )
    namespace = parser.parse_args(args)
    return LabSetupOptions(
        prime_rl=bool(namespace.prime_rl),
        skip_agents_md=bool(namespace.skip_agents_md),
        skip_install=bool(namespace.skip_install),
        agents=tuple(_parse_agents(namespace.agents)),
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
        help="Comma-separated coding agents to refresh, or 'all' for diagnostics.",
    )
    parser.add_argument(
        "--skip-docs",
        action="store_true",
        help="Skip AGENTS.md, CLAUDE.md, and environments/AGENTS.md refresh.",
    )
    namespace = parser.parse_args(args)
    return LabSyncOptions(
        agents=tuple(_parse_agents(namespace.agents)) if namespace.agents else (),
        skip_docs=bool(namespace.skip_docs),
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
    runner: Runner | None = None,
) -> LabSyncResult:
    """Refresh Lab skills, local agent guidance, and Lab-owned agent bridges."""

    workspace = workspace.expanduser().resolve()
    emit = emit or (lambda _text: None)
    runner = runner or _run_command
    try:
        _run_lab_sync_steps(options, workspace=workspace, emit=emit, runner=runner)
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

    if not options.skip_install:
        _ensure_uv_project(workspace, emit, runner)

    (workspace / "configs").mkdir(exist_ok=True)
    (workspace / "environments").mkdir(exist_ok=True)
    _sync_prime_skills(emit)
    _prepare_agent_skill_dirs(options.agents, emit)
    _report_missing_agent_requirements(options.agents, emit)
    _prepare_agent_native_surfaces(workspace, options.agents, emit)
    _sync_lab_metadata(workspace, options.agents)

    if not options.skip_agents_md:
        _copy_asset("docs/AGENTS.md", workspace / "AGENTS.md", emit, force=True)
        _copy_asset("docs/CLAUDE.md", workspace / "CLAUDE.md", emit, force=True)
        _copy_asset(
            "docs/environments/AGENTS.md",
            workspace / "environments" / "AGENTS.md",
            emit,
            force=True,
        )

    if options.prime_rl:
        _install_prime_rl(workspace, emit, runner)
        _install_environments_to_prime_rl(workspace, emit, runner)

    _copy_setup_configs(workspace, emit)
    emit("Lab setup completed\n")


def _run_lab_sync_steps(
    options: LabSyncOptions,
    *,
    workspace: Path,
    emit: Emit,
    runner: Runner,
) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "configs").mkdir(exist_ok=True)
    (workspace / "environments").mkdir(exist_ok=True)
    emit(f"Syncing Lab assets in {workspace}\n")
    agents = options.agents or _workspace_agents_from_metadata(workspace) or ("codex",)

    _sync_prime_skills(emit)
    _prepare_agent_skill_dirs(agents, emit)
    _report_missing_agent_requirements(agents, emit)
    _prepare_agent_native_surfaces(workspace, agents, emit)
    _sync_lab_metadata(workspace, agents)
    _sync_config_templates(workspace, emit)

    if not options.skip_docs:
        _copy_asset("docs/AGENTS.md", workspace / "AGENTS.md", emit, force=True)
        _copy_asset("docs/CLAUDE.md", workspace / "CLAUDE.md", emit, force=True)
        _copy_asset(
            "docs/environments/AGENTS.md",
            workspace / "environments" / "AGENTS.md",
            emit,
            force=True,
        )
        _write_lab_docs_index(workspace)

    emit("Lab sync completed\n")


def _sync_prime_skills(emit: Emit) -> None:
    manifest: dict[str, Any] = {
        "version": 1,
        "source": {"package": "prime", "version": _prime_package_version()},
        "skills": {},
    }
    for skill_name in PRIME_MANAGED_SKILLS:
        skill_dir = _global_prime_skills_dir() / skill_name
        content = _asset_text(f"skills/{skill_name}/SKILL.md")
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
        manifest["skills"][skill_name] = {
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        }
        emit(f"Installed {skill_dir}\n")
    manifest_path = _global_prime_skills_dir() / PRIME_SKILLS_MANIFEST
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _copy_setup_configs(workspace: Path, emit: Emit) -> None:
    for asset_path in CONFIG_TEMPLATE_PATHS:
        _copy_asset(asset_path, workspace / asset_path, emit)


def _sync_config_templates(workspace: Path, emit: Emit) -> None:
    template_root = workspace / ".prime" / "lab" / "templates"
    for asset_path in CONFIG_TEMPLATE_PATHS:
        _copy_asset(asset_path, template_root / asset_path, emit, force=True)


def _prepare_agent_skill_dirs(agents: tuple[str, ...], emit: Emit) -> None:
    prime_skills_dir = _global_prime_skills_dir()
    for agent in agents:
        skills_dir = agent_user_skills_dir(agent)
        if skills_dir is None:
            continue
        _remove_stale_managed_skill_links(skills_dir, prime_skills_dir)
        for skill_name in PRIME_MANAGED_SKILLS:
            source_dir = prime_skills_dir / skill_name
            if not source_dir.exists():
                continue
            target_dir = skills_dir / skill_name
            if _skill_target_is_user_owned(target_dir, prime_skills_dir):
                emit(f"Skipped {target_dir} because a user-owned skill already exists\n")
                continue
            _remove_managed_skill_target(target_dir, prime_skills_dir)
            _safe_link_or_copy_managed_skill_dir(source_dir, target_dir)
        emit(f"Prepared {skills_dir}\n")


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
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "configs").mkdir(parents=True, exist_ok=True)
        (workspace / "environments").mkdir(parents=True, exist_ok=True)
        _append_gitignore(workspace)

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
        _config_validity_check(workspace),
        _config_environment_reference_check(workspace),
        _environment_source_hygiene_check(workspace),
        _path_check(
            "Lab skills",
            _global_prime_skills_dir() / "create-environments" / "SKILL.md",
            "Run prime lab sync.",
            warning=True,
        ),
        _path_check(
            "Lab action skill",
            _global_prime_skills_dir() / LAB_WIDGETS_SKILL / "SKILL.md",
            "Run prime lab sync.",
            warning=True,
        ),
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
            agent_skill_dir = agent_user_skills_dir(agent)
            label = agent_capability(agent).label
            if agent_skill_dir is not None:
                checks.append(
                    _path_check(
                        f"{label} skills",
                        agent_skill_dir / "create-environments",
                        f"Run prime lab sync --agent {agent}.",
                        warning=True,
                    )
                )
                checks.append(
                    _path_check(
                        f"{label} Lab actions",
                        agent_skill_dir / LAB_WIDGETS_SKILL,
                        f"Run prime lab sync --agent {agent}.",
                        warning=True,
                    )
                )
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


def _agent_native_surface_check(agent: str, workspace: Path) -> LabDoctorCheck:
    capability = agent_capability(agent)
    name = f"{capability.label} native tools"
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
    if capability.native_surface in {"codex_app_server", "claude_sdk", "pi_acp"}:
        return LabDoctorCheck(
            name=name,
            status="PASS",
            message=(
                f"{capability.label} receives native Lab tools through {capability.native_surface}."
            ),
        )
    expected_paths = capability.resolved_surface_paths(workspace)
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
    missing = [pattern for pattern in LAB_GITIGNORE_PATTERNS if pattern not in existing]
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


def _install_prime_rl(workspace: Path, emit: Emit, runner: Runner) -> None:
    if (workspace / "prime-rl").is_dir():
        emit("prime-rl already exists\n")
    else:
        install_url = (
            f"https://raw.githubusercontent.com/{PRIME_RL_REPO}/{PRIME_RL_INSTALL_SCRIPT_REF}"
            "/scripts/install.sh"
        )
        emit(f"Installing prime-rl from {install_url}\n")
        _check_command(["bash", "-c", f"curl -sSL {install_url} | bash"], workspace, emit, runner)

    _check_command(["git", "checkout", PRIME_RL_REF], workspace / "prime-rl", emit, runner)
    _check_command(["uv", "sync"], workspace / "prime-rl", emit, runner)
    _check_command(["uv", "sync", "--all-extras"], workspace / "prime-rl", emit, runner)


def _install_environments_to_prime_rl(workspace: Path, emit: Emit, runner: Runner) -> None:
    envs_dir = workspace / "environments"
    prime_rl_python = workspace / "prime-rl" / ".venv" / "bin" / "python"
    if not envs_dir.is_dir() or not prime_rl_python.exists():
        return
    env_paths = [
        f"-e environments/{path.name}"
        for path in sorted(envs_dir.iterdir())
        if path.is_dir() and (path / "pyproject.toml").is_file()
    ]
    if not env_paths:
        return
    emit(f"Installing {len(env_paths)} local environments into prime-rl\n")
    code = runner(
        ["uv", "pip", "install", "--python", str(prime_rl_python), *env_paths],
        workspace,
        emit,
    )
    if code != 0:
        emit("Local environment install into prime-rl failed; continuing\n")


def _write_lab_docs_index(workspace: Path) -> None:
    docs_dir = workspace / ".prime" / "lab" / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
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
                "- `AGENTS.md`",
                "- `CLAUDE.md`",
                "- `environments/AGENTS.md`",
                "- `~/.prime/skills/*/SKILL.md`",
                "- `.prime/lab/templates/configs/**`",
                "",
                "## Prime docs",
                "",
                "- Prime CLI: https://github.com/PrimeIntellect-ai/prime-cli",
                "- Verifiers: https://github.com/PrimeIntellect-ai/verifiers",
                "- Environments Hub: https://app.primeintellect.ai/dashboard/environments",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _sync_lab_metadata(workspace: Path, agents: tuple[str, ...]) -> None:
    prime_dir = workspace / ".prime"
    prime_dir.mkdir(exist_ok=True)
    path = prime_dir / "lab.json"
    metadata = _read_lab_metadata(workspace)
    metadata["setup_source"] = "prime lab setup"
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
        parsed = tuple(str(agent) for agent in agents if str(agent))
        if parsed:
            return parsed
    primary_agent = choices.get("primary_agent")
    if primary_agent:
        return (str(primary_agent),)
    return ()


def _copy_asset(asset_path: str, dest: Path, emit: Emit, *, force: bool = False) -> None:
    if dest.exists() and not force:
        emit(f"{dest.name} already exists\n")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(_asset_text(asset_path), encoding="utf-8")
    emit(f"Prepared {dest}\n")


def _asset_text(asset_path: str) -> str:
    raw_assets = resources.files("prime_cli.lab_assets").joinpath("assets.json").read_text()
    payload = json.loads(raw_assets)
    value = payload.get(asset_path)
    if not isinstance(value, str):
        raise KeyError(f"Unknown Lab setup asset: {asset_path}")
    return value


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


def _parse_agents(value: str | None) -> list[str]:
    if value and value.strip().lower() == "all":
        return list(SUPPORTED_AGENTS)
    raw_agents = value.split(",") if value else ["codex"]
    agents: list[str] = []
    seen: set[str] = set()
    for raw_agent in raw_agents:
        raw_name = raw_agent.strip().lower()
        if not raw_name:
            continue
        agent = agent_capability(raw_name).name
        if agent not in SUPPORTED_AGENTS:
            raise ValueError(
                f"Unsupported coding agent '{raw_agent}'. Supported values: "
                + ", ".join((*SUPPORTED_AGENTS, "all"))
            )
        if agent in seen:
            continue
        seen.add(agent)
        agents.append(agent)
    return agents or ["codex"]


def _append_gitignore(workspace: Path) -> None:
    path = workspace / ".gitignore"
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    missing = [pattern for pattern in LAB_GITIGNORE_PATTERNS if pattern not in existing]
    if missing:
        section = "\n# Lab generated artifacts\n" + "\n".join(missing) + "\n"
        path.write_text(existing.rstrip() + section + "\n", encoding="utf-8")


def _global_prime_skills_dir() -> Path:
    return Path.home() / ".prime" / "skills"


def _safe_link_or_copy_managed_skill_dir(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.symlink_to(os.path.relpath(source, start=target.parent), target_is_directory=True)
    except OSError:
        shutil.copytree(source, target, dirs_exist_ok=True)
        (target / ".prime-managed-link").write_text(
            str(source.resolve(strict=False)),
            encoding="utf-8",
        )


def _skill_target_is_user_owned(target: Path, prime_skills_dir: Path) -> bool:
    if not (target.exists() or target.is_symlink()):
        return False
    return not _is_managed_skill_target(target, prime_skills_dir)


def _remove_stale_managed_skill_links(skills_dir: Path, prime_skills_dir: Path) -> None:
    if not skills_dir.exists():
        return
    for target in skills_dir.iterdir():
        if target.name in PRIME_MANAGED_SKILLS:
            continue
        _remove_managed_skill_target(target, prime_skills_dir)


def _remove_managed_skill_target(target: Path, prime_skills_dir: Path) -> None:
    if not _is_managed_skill_target(target, prime_skills_dir):
        return
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.is_dir():
        shutil.rmtree(target)


def _is_managed_skill_target(target: Path, prime_skills_dir: Path) -> bool:
    try:
        resolved = target.resolve(strict=False)
        prime_root = prime_skills_dir.resolve(strict=False)
        if target.is_symlink() and (resolved == prime_root or prime_root in resolved.parents):
            return True
    except OSError:
        return False

    marker = target / ".prime-managed-link"
    if not marker.is_file():
        return False
    try:
        resolved = Path(marker.read_text(encoding="utf-8").strip()).resolve(strict=False)
        prime_root = prime_skills_dir.resolve(strict=False)
    except OSError:
        return False
    return resolved == prime_root or prime_root in resolved.parents


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
