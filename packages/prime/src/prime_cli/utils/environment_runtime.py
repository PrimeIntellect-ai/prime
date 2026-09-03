"""Classify which verifiers API an environment package targets (v0 vs v1).

Local mirror of the Hub's server-side classifier (platform
``backend/app/utils/environment_runtime.py``), applied to the environment
directory at push time. The result is sent as ``runtime_hint`` on wheel and
version uploads and the Hub stores it as authoritative, so the rules here must
stay consistent with the server:

* source evidence wins — ``verifiers.v1`` imports are an unambiguous v1 marker,
  ``def load_environment`` an unambiguous v0 one, ``[project].tags`` weak v0;
* otherwise the ``verifiers`` requirement decides — v1 shipped as verifiers
  0.2.0, so a lower bound at or above 0.2.0 is a v1 signal;
* neither signal → ``None``: the hint is omitted and the Hub classifies the
  package itself at unpack time.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version

VERIFIERS_V0 = "VERIFIERS_V0"
VERIFIERS_V1 = "VERIFIERS_V1"

# First verifiers release that shipped the v1 API (`verifiers.v1`).
VERIFIERS_V1_MIN_VERSION = Version("0.2.0")

# Same scan bounds as the server: markers live near the top of small modules.
MAX_SOURCE_FILE_BYTES = 256 * 1024
MAX_SOURCE_FILES = 40

# Skip tests and vendored code (directory components only, so `test_utils.py` still scans).
_SKIPPED_DIRS = {
    "tests",
    "test",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "site-packages",
    "outputs",
}

# Anchored to real import statements / the conventional `vf.` alias so prose
# mentions in docstrings are unlikely to match. A bare `class X(Taskset)` is
# deliberately not a marker: nothing ties that name to verifiers.
_V1_SOURCE_PATTERNS = (
    re.compile(r"^\s*(?:from|import)\s+verifiers\.v1\b", re.MULTILINE),
    re.compile(r"^\s*from\s+verifiers\s+import\s+(?:\([^)]*\b|[^\n]*\b)?v1\b", re.MULTILINE),
    re.compile(r"\bvf\.(?:Taskset|Harness|TasksetConfig|HarnessConfig)\b"),
)
_V0_SOURCE_PATTERNS = (
    re.compile(r"^\s*def\s+load_environment\s*\(", re.MULTILINE),
    re.compile(r"^\s*(?:from|import)\s+verifiers\.legacy\b", re.MULTILINE),
)


def find_verifiers_requirement(requirement_strings: Iterable[str]) -> Optional[Requirement]:
    """Effective ``verifiers`` requirement: unconditional entries beat marker-guarded
    ones, and among those the highest floor wins (repeated entries intersect)."""
    requirements: List[Requirement] = []
    for text in requirement_strings:
        try:
            requirement = Requirement(text.strip())
        except InvalidRequirement:
            continue
        if requirement.name.lower() == "verifiers":
            requirements.append(requirement)
    if not requirements:
        return None
    unconditional = [r for r in requirements if r.marker is None]
    return max(
        unconditional or requirements,
        key=lambda requirement: _verifiers_lower_bound(requirement) or Version("0"),
    )


def _verifiers_lower_bound(requirement: Requirement) -> Optional[Version]:
    """Highest floor set by ``>=``/``>``/``~=``/``==``/``===`` clauses (``==X.Y.*``
    counts as ``X.Y``; ``>`` as its version — a patch level is irrelevant at 0.2.0)."""
    floors: List[Version] = []
    for spec in requirement.specifier:
        if spec.operator not in {">=", ">", "==", "===", "~="}:
            continue
        version = spec.version
        if "*" in version:
            if spec.operator != "==" or not version.endswith(".*"):
                continue
            version = version[: -len(".*")]
        try:
            floors.append(Version(version))
        except InvalidVersion:
            continue
    return max(floors) if floors else None


def classify_runtime_from_metadata(requirement_strings: Iterable[str]) -> Optional[str]:
    """v1 for a verifiers floor >= 0.2.0, v0 for any other pin, ``None`` without one
    (a URL pin says nothing about the API)."""
    requirement = find_verifiers_requirement(requirement_strings)
    if requirement is None or requirement.url:
        return None
    floor = _verifiers_lower_bound(requirement)
    if floor is not None and floor >= VERIFIERS_V1_MIN_VERSION:
        return VERIFIERS_V1
    return VERIFIERS_V0


def classify_runtime_from_source(
    sources: Iterable[str], legacy_tags: bool = False
) -> Optional[str]:
    """v1 markers win: a v1 package may keep a ``load_environment`` shim for old
    callers, but nothing pre-v1 imports ``verifiers.v1``."""
    saw_v0 = legacy_tags
    for content in sources:
        if any(pattern.search(content) for pattern in _V1_SOURCE_PATTERNS):
            return VERIFIERS_V1
        if any(pattern.search(content) for pattern in _V0_SOURCE_PATTERNS):
            saw_v0 = True
    return VERIFIERS_V0 if saw_v0 else None


def _scan_order(rel_path: str) -> Tuple[int, str]:
    """Shallow modules first, ``__init__`` before its siblings."""
    depth = rel_path.count("/")
    if rel_path.endswith("__init__.py"):
        return (depth, f"{rel_path[: -len('__init__.py')]}!")
    return (depth, rel_path)


def collect_sources(
    env_path: Path, files: Iterable[Path], max_files: int = MAX_SOURCE_FILES
) -> List[str]:
    """Contents of the scannable modules among the push's archive files — the same
    set the Hub scans at unpack time, bounded the same way."""
    candidates: Dict[str, Path] = {}
    for file_path in files:
        rel_path = str(file_path.relative_to(env_path)).replace("\\", "/")
        if not rel_path.endswith(".py"):
            continue
        if any(part in _SKIPPED_DIRS for part in rel_path.split("/")[:-1]):
            continue
        try:
            if file_path.stat().st_size > MAX_SOURCE_FILE_BYTES:
                continue
        except OSError:
            continue
        candidates[rel_path] = file_path

    sources: List[str] = []
    for rel_path in sorted(candidates, key=_scan_order)[:max_files]:
        try:
            sources.append(candidates[rel_path].read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return sources


def detect_environment_runtime(
    env_path: Path,
    files: Iterable[Path],
    requires_dist: Optional[Iterable[str]] = None,
    dependencies: Optional[Iterable[str]] = None,
    legacy_tags: bool = False,
) -> Optional[str]:
    """The ``runtime_hint`` to send with a push, or ``None`` to omit it."""
    from_source = classify_runtime_from_source(collect_sources(env_path, files), legacy_tags)
    if from_source is not None:
        return from_source
    requirements = [r for r in (requires_dist or dependencies or []) if isinstance(r, str)]
    return classify_runtime_from_metadata(requirements)
