"""Prime-owned Lab setup service."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import tomli
from rich.console import Console
from rich.table import Table

VERIFIERS_REPO = "primeintellect-ai/verifiers"
PRIME_RL_REPO = "primeintellect-ai/prime-rl"
VERIFIERS_REF = "main"
PRIME_RL_REF = "main"
PRIME_RL_INSTALL_SCRIPT_REF = "main"

ENDPOINTS_SRC = (
    f"https://raw.githubusercontent.com/{VERIFIERS_REPO}/refs/heads/{VERIFIERS_REF}"
    "/configs/endpoints.toml"
)
AGENTS_MD_SRC = (
    f"https://raw.githubusercontent.com/{VERIFIERS_REPO}/refs/heads/{VERIFIERS_REF}"
    "/assets/lab/AGENTS.md"
)
CLAUDE_MD_SRC = (
    f"https://raw.githubusercontent.com/{VERIFIERS_REPO}/refs/heads/{VERIFIERS_REF}"
    "/assets/lab/CLAUDE.md"
)
ENVS_AGENTS_MD_SRC = (
    f"https://raw.githubusercontent.com/{VERIFIERS_REPO}/refs/heads/{VERIFIERS_REF}"
    "/assets/lab/environments/AGENTS.md"
)

LAB_SKILLS = (
    "create-environments",
    "browse-environments",
    "review-environments",
    "evaluate-environments",
    "optimize-with-environments",
    "train-with-environments",
    "brainstorm",
)
SUPPORTED_AGENTS = ("codex", "claude", "cursor", "opencode", "amp", "pi")
AGENT_SKILLS_DIR_MAP = {
    "amp": ".agents/skills",
    "pi": ".pi/skills",
}
AGENT_SKILL_NAME_MAP: dict[str, dict[str, str]] = {}
LAB_GITIGNORE_PATTERNS = (
    "./outputs",
    "./environments/*/outputs",
    "./environments/*/dist",
    "./environments/*/*.egg-info",
    "./environments/*/__pycache__",
    "*.pyc",
)

ConfigSpec = tuple[str, str, str]
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


PRIME_RL_CONFIGS: tuple[ConfigSpec, ...] = (
    (
        VERIFIERS_REPO,
        "configs/local/prime-rl/wiki-search.toml",
        "configs/prime-rl/wiki-search.toml",
    ),
)
RL_CONFIGS: tuple[ConfigSpec, ...] = tuple(
    (VERIFIERS_REPO, path, path)
    for path in (
        "configs/rl/alphabet-sort.toml",
        "configs/rl/gsm8k.toml",
        "configs/rl/math-python.toml",
        "configs/rl/reverse-text.toml",
        "configs/rl/wiki-search.toml",
        "configs/rl/wordle.toml",
    )
)
GEPA_CONFIGS: tuple[ConfigSpec, ...] = tuple(
    (VERIFIERS_REPO, path, path)
    for path in (
        "configs/gepa/base.toml",
        "configs/gepa/wordle.toml",
    )
)
EVAL_CONFIGS: tuple[ConfigSpec, ...] = tuple(
    (VERIFIERS_REPO, path, path)
    for path in (
        "configs/eval/minimal.toml",
        "configs/eval/multi-env.toml",
    )
)


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
        help="Comma-separated coding agents to scaffold.",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Accepted for compatibility; TUI and CLI setup resolve agents explicitly.",
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
        help="Comma-separated coding agents to refresh.",
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
) -> LabSyncResult:
    """Refresh Lab skills and local agent guidance without installing dependencies."""

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

    if not options.skip_install:
        _ensure_uv_project(workspace, emit, runner)

    (workspace / "configs").mkdir(exist_ok=True)
    (workspace / "environments").mkdir(exist_ok=True)
    _sync_prime_skills(workspace, emit)
    _prepare_agent_skill_dirs(workspace, options.agents, emit)
    _sync_lab_metadata(workspace, options.agents)

    if not options.skip_agents_md:
        _download_file(AGENTS_MD_SRC, workspace / "AGENTS.md", emit, force=True)
        _download_file(CLAUDE_MD_SRC, workspace / "CLAUDE.md", emit, force=True)
        _download_file(
            ENVS_AGENTS_MD_SRC,
            workspace / "environments" / "AGENTS.md",
            emit,
            force=True,
        )

    if options.prime_rl:
        _install_prime_rl(workspace, emit, runner)
        _install_environments_to_prime_rl(workspace, emit, runner)

    _download_file(ENDPOINTS_SRC, workspace / "configs" / "endpoints.toml", emit)

    configs: list[ConfigSpec] = []
    if options.prime_rl:
        configs.extend(PRIME_RL_CONFIGS)
    configs.extend(GEPA_CONFIGS)
    configs.extend(EVAL_CONFIGS)
    if not options.prime_rl:
        configs.extend(RL_CONFIGS)
    _download_configs(workspace, _dedupe_config_destinations(configs), emit)
    emit("Lab setup completed\n")


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
    agents = options.agents or _workspace_agents_from_metadata(workspace) or ("codex",)

    _sync_prime_skills(workspace, emit)
    _prepare_agent_skill_dirs(workspace, agents, emit)
    _sync_lab_metadata(workspace, agents)
    _sync_config_templates(workspace, emit)

    if not options.skip_docs:
        _download_file(AGENTS_MD_SRC, workspace / "AGENTS.md", emit, force=True)
        _download_file(CLAUDE_MD_SRC, workspace / "CLAUDE.md", emit, force=True)
        _download_file(
            ENVS_AGENTS_MD_SRC,
            workspace / "environments" / "AGENTS.md",
            emit,
            force=True,
        )
        _write_lab_docs_index(workspace)

    emit("Lab sync completed\n")


def _sync_config_templates(workspace: Path, emit: Emit) -> None:
    template_root = workspace / ".prime" / "lab" / "templates"
    _download_file(ENDPOINTS_SRC, template_root / "configs" / "endpoints.toml", emit, force=True)
    configs: list[ConfigSpec] = []
    configs.extend(GEPA_CONFIGS)
    configs.extend(EVAL_CONFIGS)
    configs.extend(RL_CONFIGS)
    for repo, source_path, dest_path in _dedupe_config_destinations(configs):
        ref = PRIME_RL_REF if repo == PRIME_RL_REPO else VERIFIERS_REF
        src = f"https://raw.githubusercontent.com/{repo}/refs/heads/{ref}/{source_path}"
        _download_file(src, template_root / dest_path, emit, force=True)


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
                "- `.prime/skills/*/SKILL.md`",
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


def _lab_doctor_checks(options: LabDoctorOptions, workspace: Path) -> list[LabDoctorCheck]:
    if options.fix:
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "configs").mkdir(parents=True, exist_ok=True)
        (workspace / "environments").mkdir(parents=True, exist_ok=True)
        _append_gitignore(workspace)

    metadata_path = workspace / ".prime" / "lab.json"
    metadata = _read_lab_metadata(workspace)
    agents = _workspace_agents_from_metadata(workspace)
    primary_agent = agents[0] if agents else ""

    checks = [
        _path_check(
            "Workspace metadata",
            metadata_path,
            "Run prime lab setup.",
        ),
        _path_check(
            "Python project",
            workspace / "pyproject.toml",
            "Run prime lab setup or uv init before launching local workflows.",
        ),
        _path_check(
            "Configs directory",
            workspace / "configs",
            "Run prime lab doctor --fix to create configs/.",
        ),
        _path_check(
            "Environments directory",
            workspace / "environments",
            "Run prime lab doctor --fix to create environments/.",
        ),
        _gitignore_check(workspace),
        _config_validity_check(workspace),
        _config_environment_reference_check(workspace),
        _environment_source_hygiene_check(workspace),
        _path_check(
            "Lab skills",
            workspace / ".prime" / "skills" / "create-environments" / "SKILL.md",
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

    if primary_agent:
        agent_skill_dir = workspace / AGENT_SKILLS_DIR_MAP.get(
            primary_agent, f".{primary_agent}/skills"
        )
        checks.append(
            _path_check(
                "Agent skills",
                agent_skill_dir / "create-environments",
                f"Run prime lab sync --agent {primary_agent}.",
                warning=True,
            )
        )
    else:
        checks.append(
            LabDoctorCheck(
                name="Coding agent",
                status="WARN",
                message="No primary coding agent is configured.",
                remediation="Run prime lab setup or configure an agent from Home.",
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
            tomli.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomli.TOMLDecodeError):
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
    local_names = _local_environment_names(workspace)
    unpinned: list[str] = []
    missing_local: list[str] = []
    for path in config_paths:
        try:
            parsed = tomli.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomli.TOMLDecodeError):
            continue
        for ref in _config_environment_refs(parsed):
            if not ref.version:
                unpinned.append(f"{path.relative_to(workspace)}:{ref.env_id}")
            if "/" not in ref.env_id and local_names and ref.env_id not in local_names:
                missing_local.append(f"{path.relative_to(workspace)}:{ref.env_id}")
    if missing_local:
        return LabDoctorCheck(
            name="Config environment refs",
            status="WARN",
            message="Local env not found: " + ", ".join(missing_local[:5]),
            remediation="Fix the environment id or add the local environment source.",
        )
    if unpinned:
        return LabDoctorCheck(
            name="Config environment refs",
            status="WARN",
            message="Unpinned env versions: " + ", ".join(unpinned[:5]),
            remediation="Pin environment versions before committing reproducible configs.",
        )
    return LabDoctorCheck(
        name="Config environment refs",
        status="PASS",
        message="Environment references are pinned or local.",
    )


@dataclass(frozen=True)
class _ConfigEnvironmentRef:
    env_id: str
    version: str


def _config_environment_refs(config: dict[str, Any]) -> list[_ConfigEnvironmentRef]:
    refs: list[_ConfigEnvironmentRef] = []
    for key in ("env", "environments"):
        value = config.get(key)
        if isinstance(value, list):
            refs.extend(_environment_refs_from_list(value, id_key="id"))
    evals = config.get("eval")
    if isinstance(evals, list):
        refs.extend(_environment_refs_from_list(evals, id_key="env_id"))
    return [ref for ref in refs if ref.env_id]


def _environment_refs_from_list(
    values: list[Any],
    *,
    id_key: str,
) -> list[_ConfigEnvironmentRef]:
    refs: list[_ConfigEnvironmentRef] = []
    for value in values:
        if isinstance(value, str):
            env_id, version = _split_env_ref(value)
            refs.append(_ConfigEnvironmentRef(env_id=env_id, version=version))
            continue
        if isinstance(value, dict):
            env_id = str(value.get(id_key) or value.get("id") or value.get("name") or "")
            refs.append(
                _ConfigEnvironmentRef(
                    env_id=env_id,
                    version=str(value.get("version") or ""),
                )
            )
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
    missing_metadata: list[str] = []
    missing_readmes: list[str] = []
    for env_dir in sorted(path for path in envs_dir.iterdir() if path.is_dir()):
        if not (env_dir / "pyproject.toml").is_file():
            missing_metadata.append(env_dir.name)
        if not _has_readme(env_dir):
            missing_readmes.append(env_dir.name)
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
    if missing_metadata:
        return LabDoctorCheck(
            name="Environment source hygiene",
            status="WARN",
            message="Missing pyproject.toml in " + ", ".join(missing_metadata[:5]),
            remediation="Add package metadata before publishing environments.",
        )
    if missing_readmes:
        return LabDoctorCheck(
            name="Environment source hygiene",
            status="WARN",
            message="Missing README in " + ", ".join(missing_readmes[:5]),
            remediation="Add README.md so local and Hub views have useful context.",
        )
    env_count = len([path for path in envs_dir.iterdir() if path.is_dir()])
    return LabDoctorCheck(
        name="Environment source hygiene",
        status="PASS",
        message=f"{env_count} local environment source dir(s) look clean.",
    )


def _has_readme(path: Path) -> bool:
    return any(candidate.is_file() for candidate in (path / "README.md", path / "readme.md"))


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


def _sync_prime_skills(workspace: Path, emit: Emit) -> None:
    for skill_name in LAB_SKILLS:
        src = (
            f"https://raw.githubusercontent.com/{VERIFIERS_REPO}/refs/heads/{VERIFIERS_REF}"
            f"/skills/{skill_name}/SKILL.md"
        )
        dst = workspace / ".prime" / "skills" / skill_name / "SKILL.md"
        _download_file(src, dst, emit)


def _prepare_agent_skill_dirs(workspace: Path, agents: tuple[str, ...], emit: Emit) -> None:
    for agent in agents:
        skills_dir = workspace / AGENT_SKILLS_DIR_MAP.get(agent, f".{agent}/skills")
        skills_dir.mkdir(parents=True, exist_ok=True)
        for skill_name in LAB_SKILLS:
            source_dir = workspace / ".prime" / "skills" / skill_name
            if not source_dir.exists():
                continue
            target_name = AGENT_SKILL_NAME_MAP.get(agent, {}).get(skill_name, skill_name)
            target_dir = skills_dir / target_name
            _safe_link_or_copy_skill_dir(source_dir, target_dir)
        emit(f"Prepared {skills_dir.relative_to(workspace)}\n")


def _safe_link_or_copy_skill_dir(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.symlink_to(os.path.relpath(source, start=target.parent), target_is_directory=True)
    except OSError:
        shutil.copytree(source, target, dirs_exist_ok=True)


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


def _download_configs(workspace: Path, configs: list[ConfigSpec], emit: Emit) -> None:
    for repo, source_path, dest_path in configs:
        ref = PRIME_RL_REF if repo == PRIME_RL_REPO else VERIFIERS_REF
        src = f"https://raw.githubusercontent.com/{repo}/refs/heads/{ref}/{source_path}"
        _download_file(src, workspace / dest_path, emit)


def _download_file(url: str, dest: Path, emit: Emit, *, force: bool = False) -> None:
    if dest.exists() and not force:
        emit(f"{dest.name} already exists\n")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = httpx.get(url, follow_redirects=True, timeout=60.0)
    response.raise_for_status()
    dest.write_bytes(response.content)
    emit(f"Downloaded {dest}\n")


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


def _dedupe_config_destinations(configs: list[ConfigSpec]) -> list[ConfigSpec]:
    deduped: list[ConfigSpec] = []
    seen: set[str] = set()
    for config in configs:
        if config[2] in seen:
            continue
        seen.add(config[2])
        deduped.append(config)
    return deduped


def _parse_agents(value: str | None) -> list[str]:
    raw_agents = value.split(",") if value else ["codex"]
    agents: list[str] = []
    seen: set[str] = set()
    for raw_agent in raw_agents:
        agent = raw_agent.strip().lower()
        if not agent:
            continue
        if agent not in SUPPORTED_AGENTS:
            raise ValueError(
                f"Unsupported coding agent '{raw_agent}'. Supported values: "
                + ", ".join(SUPPORTED_AGENTS)
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


def _remove_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


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
