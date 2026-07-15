"""Cheap Prime Lab workspace hygiene checks."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

LAB_AGENT_SKILL_DIRS = (
    ".agents/skills",
    ".claude/skills",
    ".cursor/skills",
    ".factory/skills",
    ".grok/skills",
    ".opencode/skills",
    ".pi/skills",
)
LAB_GITIGNORE_PATTERNS = (
    ".env",
    "/AGENTS.md",
    "/CLAUDE.md",
    "/CLAUDE.local.md",
    "/.prime/",
    *(f"/{skill_dir}/" for skill_dir in LAB_AGENT_SKILL_DIRS),
    "/outputs/",
    "/prime-rl/",
    "/environments/AGENTS.md",
    "/environments/*/outputs/",
    "/environments/*/dist/",
    "/environments/*/*.egg-info/",
    "/environments/*/__pycache__/",
    "__pycache__/",
    "*.py[cod]",
    ".pytest_cache/",
    ".ruff_cache/",
)
LAB_GITHUB_WORKFLOW_RELATIVE_PATH = Path(".github") / "workflows" / "prime-lab-hygiene.yml"
LAB_GITHUB_WORKFLOW = """\
name: Prime Lab Hygiene

on:
  pull_request:
  push:
    branches:
      - main

jobs:
  lab-git-hygiene:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - name: Check Lab git hygiene
        run: uvx --from prime prime lab hygiene
"""

LAB_TRACKED_EXACT_PATHS = {
    "AGENTS.md",
    "CLAUDE.md",
    "CLAUDE.local.md",
    "environments/AGENTS.md",
}
LAB_TRACKED_PREFIXES = (
    ".prime/",
    *(f"{skill_dir}/" for skill_dir in LAB_AGENT_SKILL_DIRS),
    "outputs/",
    "prime-rl/",
    ".pytest_cache/",
    ".ruff_cache/",
)
MAX_TRACKED_PATHS_TO_SHOW = 8

Emit = Callable[[str], None]


@dataclass(frozen=True)
class LabHygieneOptions:
    """Options for cheap Lab hygiene checks."""

    fix: bool = False
    fail_on_tracked: bool = False


@dataclass(frozen=True)
class LabHygieneResult:
    """Result from a cheap Lab hygiene check."""

    exit_code: int
    workspace: Path
    fixed: tuple[str, ...]
    warnings: tuple[str, ...]
    tracked_paths: tuple[str, ...]


def find_lab_workspace(path: Path) -> Path | None:
    """Find the nearest Lab workspace marker at or above a path."""

    current = path.expanduser().resolve(strict=False)
    if not current.is_dir():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / ".prime" / "lab.json").is_file():
            return candidate
    return None


def run_lab_hygiene_preflight(
    options: LabHygieneOptions,
    *,
    workspace: Path,
    emit: Emit | None = None,
) -> LabHygieneResult:
    """Run cheap local Lab hygiene checks with optional safe fixes."""

    workspace = workspace.expanduser().resolve(strict=False)
    fixed: list[str] = []
    warnings: list[str] = []

    if options.fix:
        if not (workspace / "configs").is_dir():
            (workspace / "configs").mkdir(parents=True, exist_ok=True)
            fixed.append("created configs/")
        if not (workspace / "environments").is_dir():
            (workspace / "environments").mkdir(parents=True, exist_ok=True)
            fixed.append("created environments/")
        missing = append_lab_gitignore(workspace)
        if missing:
            fixed.append("added standard .gitignore entries")
        tracked_paths = tracked_lab_hygiene_paths(workspace)
        if tracked_paths:
            untracked_paths = _untrack_lab_hygiene_paths(workspace, tracked_paths)
            if untracked_paths:
                fixed.append(
                    "untracked generated Lab files: " + _tracked_paths_summary(untracked_paths)
                )
    else:
        if not (workspace / "configs").is_dir():
            warnings.append("missing configs/; run prime lab hygiene --fix")
        if not (workspace / "environments").is_dir():
            warnings.append("missing environments/; run prime lab hygiene --fix")
        missing = missing_lab_gitignore_patterns(_read_gitignore(workspace))
        if missing:
            warnings.append("missing standard .gitignore entries; run prime lab hygiene --fix")

    tracked_paths = tracked_lab_hygiene_paths(workspace)
    if tracked_paths:
        warnings.append(_tracked_paths_warning(tracked_paths))

    exit_code = 1 if options.fail_on_tracked and tracked_paths else 0
    result = LabHygieneResult(
        exit_code=exit_code,
        workspace=workspace,
        fixed=tuple(fixed),
        warnings=tuple(warnings),
        tracked_paths=tracked_paths,
    )
    if emit is not None:
        _emit_hygiene_result(result, emit)
    return result


def write_lab_github_workflow(workspace: Path) -> Path:
    """Write the generated GitHub Actions workflow for Lab git hygiene."""

    path = workspace.expanduser().resolve(strict=False) / LAB_GITHUB_WORKFLOW_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(LAB_GITHUB_WORKFLOW, encoding="utf-8")
    return path


def append_lab_gitignore(workspace: Path) -> tuple[str, ...]:
    """Append missing Lab-generated artifact patterns to the workspace gitignore."""

    path = workspace / ".gitignore"
    existing = _read_gitignore(workspace)
    missing = missing_lab_gitignore_patterns(existing)
    if missing:
        section = "\n# Lab generated artifacts\n" + "\n".join(missing) + "\n"
        path.write_text(existing.rstrip() + section + "\n", encoding="utf-8")
    return tuple(missing)


def missing_lab_gitignore_patterns(existing: str) -> list[str]:
    existing_patterns = {
        line.strip()
        for line in existing.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    return [pattern for pattern in LAB_GITIGNORE_PATTERNS if pattern not in existing_patterns]


def tracked_lab_hygiene_paths(workspace: Path) -> tuple[str, ...]:
    """Return tracked generated Lab guidance or output paths under a workspace."""

    workspace = workspace.expanduser().resolve(strict=False)
    git_root = _git_root(workspace)
    if git_root is None:
        return ()

    tracked: list[str] = []
    for git_relative_path in _git_tracked_paths(git_root):
        absolute_path = git_root / git_relative_path
        try:
            workspace_relative_path = absolute_path.relative_to(workspace)
        except ValueError:
            continue
        relative = workspace_relative_path.as_posix()
        if _is_tracked_lab_hygiene_path(relative):
            tracked.append(relative)
    return tuple(sorted(tracked))


def _is_tracked_lab_hygiene_path(relative: str) -> bool:
    if relative in LAB_TRACKED_EXACT_PATHS:
        return True
    if any(relative.startswith(prefix) for prefix in LAB_TRACKED_PREFIXES):
        return True
    if relative.endswith(".pyc") or "/__pycache__/" in relative:
        return True

    parts = relative.split("/")
    if len(parts) < 3 or parts[0] != "environments":
        return False
    nested = "/".join(parts[2:])
    return (
        nested.startswith(("outputs/", "dist/", "__pycache__/"))
        or ".egg-info/" in nested
        or nested.endswith(".pyc")
    )


def _read_gitignore(workspace: Path) -> str:
    path = workspace / ".gitignore"
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _git_root(workspace: Path) -> Path | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(workspace), "rev-parse", "--show-toplevel"],
            capture_output=True,
            check=False,
            text=True,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    root = result.stdout.strip()
    return Path(root).resolve(strict=False) if root else None


def _git_tracked_paths(git_root: Path) -> tuple[Path, ...]:
    try:
        result = subprocess.run(
            ["git", "-C", str(git_root), "ls-files", "-z"],
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return ()
    if result.returncode != 0:
        return ()
    raw_paths = result.stdout.split(b"\0")
    return tuple(Path(raw.decode("utf-8")) for raw in raw_paths if raw)


def _untrack_lab_hygiene_paths(workspace: Path, paths: tuple[str, ...]) -> tuple[str, ...]:
    try:
        result = subprocess.run(
            ["git", "-C", str(workspace), "rm", "--cached", "--force", "--quiet", "--", *paths],
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return ()
    return paths if result.returncode == 0 else ()


def _tracked_paths_summary(tracked_paths: tuple[str, ...]) -> str:
    shown = tracked_paths[:MAX_TRACKED_PATHS_TO_SHOW]
    suffix = (
        "" if len(tracked_paths) <= len(shown) else f" and {len(tracked_paths) - len(shown)} more"
    )
    return ", ".join(shown) + suffix


def _tracked_paths_warning(tracked_paths: tuple[str, ...]) -> str:
    summary = _tracked_paths_summary(tracked_paths)
    shown = tracked_paths[:MAX_TRACKED_PATHS_TO_SHOW]
    quoted_paths = " ".join(shlex.quote(path) for path in shown)
    return (
        "tracked generated Lab files: "
        + summary
        + f"; run `git rm --cached {quoted_paths}` to keep them local only"
    )


def _emit_hygiene_result(result: LabHygieneResult, emit: Emit) -> None:
    if result.fixed:
        emit("Lab hygiene: " + "; ".join(result.fixed))
    for warning in result.warnings:
        emit("Lab hygiene: " + warning)


__all__ = [
    "LAB_GITHUB_WORKFLOW",
    "LAB_GITHUB_WORKFLOW_RELATIVE_PATH",
    "LAB_GITIGNORE_PATTERNS",
    "LabHygieneOptions",
    "LabHygieneResult",
    "append_lab_gitignore",
    "find_lab_workspace",
    "missing_lab_gitignore_patterns",
    "run_lab_hygiene_preflight",
    "tracked_lab_hygiene_paths",
    "write_lab_github_workflow",
]
