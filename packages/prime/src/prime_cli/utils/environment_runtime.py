"""Classify which verifiers API an environment package targets (v0 vs v1).

Local mirror of the Hub's server-side classifier (platform
``backend/app/utils/environment_runtime.py``), applied to the environment
directory at push time. The result is sent as the optional ``runtime_hint``
field on wheel and version uploads; the Hub stores it as authoritative, so the
rules here must stay consistent with the server:

* source evidence wins over the requirement heuristic — ``verifiers.v1``
  imports are an unambiguous v1 marker, ``def load_environment`` an
  unambiguous v0 one;
* otherwise the ``verifiers`` requirement decides — v1 shipped as verifiers
  0.2.0, so a lower bound at or above 0.2.0 is a v1 signal;
* a package with neither signal returns ``None`` and the hint is omitted, in
  which case the Hub classifies the package itself at unpack time.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

try:
    import tomllib
except ImportError:
    import tomli as tomllib
from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version

VERIFIERS_V0 = "VERIFIERS_V0"
VERIFIERS_V1 = "VERIFIERS_V1"

# First verifiers release that shipped the v1 API (`verifiers.v1`).
VERIFIERS_V1_MIN_VERSION = Version("0.2.0")

# Source files larger than this are skipped: markers live near the top of
# small package modules, not in bundled data files.
MAX_SOURCE_FILE_BYTES = 256 * 1024

# Upper bound on source files read per push. Markers live in the package's
# top-level modules, which `_source_scan_order` puts first.
MAX_SOURCE_FILES = 40

# Only scan Python modules and the project manifest; skip tests and vendored code.
_SCANNABLE_SUFFIXES = (".py", "pyproject.toml")
_SKIPPED_PATH_PARTS = {
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


def effective_requirement_strings(
    requires_dist: Optional[Iterable[str]],
    dependencies: Optional[Iterable[str]],
) -> List[str]:
    """Pick the richest requirement source: wheel metadata, then pyproject."""
    for value in (requires_dist, dependencies):
        if not value:
            continue
        entries = [entry for entry in value if isinstance(entry, str)]
        if entries:
            return entries
    return []


def _verifiers_requirements(requirement_strings: Iterable[str]) -> List[Requirement]:
    requirements: List[Requirement] = []
    for requirement_text in requirement_strings:
        try:
            requirement = Requirement(requirement_text.strip())
        except InvalidRequirement:
            continue
        if requirement.name.lower() == "verifiers":
            requirements.append(requirement)
    return requirements


def find_verifiers_requirement(
    requirement_strings: Iterable[str],
) -> Optional[Requirement]:
    """Return the package's effective ``verifiers`` requirement, if any.

    Unconditional entries take precedence over marker-guarded ones
    (``verifiers>=0.2; python_version >= "3.11"``). Within the applicable set,
    repeated entries intersect, so the one with the highest floor is the
    effective requirement — a package that admits v1 in any of its
    environments targets v1.
    """
    requirements = _verifiers_requirements(requirement_strings)
    if not requirements:
        return None
    unconditional = [r for r in requirements if r.marker is None]
    applicable = unconditional or requirements
    return max(
        applicable,
        key=lambda requirement: _verifiers_lower_bound(requirement) or Version("0"),
    )


def _verifiers_lower_bound(requirement: Requirement) -> Optional[Version]:
    """Effective lower bound of the specifier set, or None if it has no floor.

    ``>=``, ``>``, ``~=``, ``==``/``===`` and ``==X.Y.*`` prefix pins all set a
    floor; clauses intersect, so the highest floor is the effective one
    (``>=0.1,>=0.2`` admits nothing below 0.2). ``>`` is treated as its
    version — off by one patch level is irrelevant at the 0.2.0 cut.
    """
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


def classify_runtime_from_metadata(
    requirement_strings: Iterable[str],
) -> Optional[str]:
    """Heuristic classification from the verifiers requirement alone.

    * floor ``>= 0.2.0`` → v1 (v1 shipped as verifiers 0.2.0);
    * any other verifiers requirement (``>=0.1.x``, unbounded, ``<0.2``) → v0 —
      every package published before v1 existed looks like this, and v0
      packages that upgraded their pin are caught by the source scan;
    * no verifiers requirement → ``None`` (no opinion).
    """
    requirement = find_verifiers_requirement(requirement_strings)
    if requirement is None:
        return None
    if requirement.url:
        # `verifiers @ git+https://...` — can't tell which API from a URL pin.
        return None
    floor = _verifiers_lower_bound(requirement)
    if floor is not None and floor >= VERIFIERS_V1_MIN_VERSION:
        return VERIFIERS_V1
    return VERIFIERS_V0


def _is_scannable_source_path(path: str, size: Optional[int] = None) -> bool:
    """Whether a file is worth reading for classification."""
    if size is not None and size > MAX_SOURCE_FILE_BYTES:
        return False
    normalized = path.strip("/")
    if not normalized.endswith(_SCANNABLE_SUFFIXES):
        return False
    parts = normalized.split("/")
    # Only skip on *directory* components so a `test_utils.py` module still scans.
    return not any(part in _SKIPPED_PATH_PARTS for part in parts[:-1])


def _pyproject_declares_legacy_tags(content: str) -> bool:
    """True when ``[project].tags`` is set (only that table; ``[tool.*]`` tags
    are unrelated)."""
    try:
        data = tomllib.loads(content)
    except Exception:
        return False
    project = data.get("project")
    return isinstance(project, dict) and "tags" in project


def classify_runtime_from_source(sources: Mapping[str, str]) -> Optional[str]:
    """Classify from source contents (relative path → text).

    v1 markers win: a v1 package may keep a ``load_environment`` shim for old
    callers, but nothing pre-v1 imports ``verifiers.v1``.
    """
    saw_v0 = False
    for path, content in sources.items():
        if not content:
            continue
        if path.endswith("pyproject.toml"):
            # `[project].tags` is the non-PEP-621 field the old Actions CI
            # asserted on; v1 packages don't need it but may still carry it,
            # so it only counts as weak v0 evidence.
            if _pyproject_declares_legacy_tags(content):
                saw_v0 = True
            continue
        if any(pattern.search(content) for pattern in _V1_SOURCE_PATTERNS):
            return VERIFIERS_V1
        if any(pattern.search(content) for pattern in _V0_SOURCE_PATTERNS):
            saw_v0 = True
    return VERIFIERS_V0 if saw_v0 else None


def _source_scan_order(path: str) -> Tuple[int, int, str]:
    """Sort key: pyproject first, then shallow modules (``__init__`` before siblings)."""
    if path.endswith("pyproject.toml"):
        return (0, 0, path)
    depth = path.count("/")
    if path.endswith("__init__.py"):
        return (1, depth, f"{path[: -len('__init__.py')]}!")
    return (1, depth, path)


def collect_sources(
    env_path: Path,
    files: Iterable[Path],
    max_files: int = MAX_SOURCE_FILES,
) -> Dict[str, str]:
    """Read the scannable source files out of the push's archive file set.

    ``files`` is the same collection that goes into the source tarball, so the
    CLI scans exactly what the Hub would scan at unpack time. Reads stop early
    once a v1 marker is seen — it is conclusive.
    """
    by_rel_path: Dict[str, Path] = {}
    for file_path in files:
        rel_path = str(file_path.relative_to(env_path)).replace("\\", "/")
        try:
            size = file_path.stat().st_size
        except OSError:
            continue
        if _is_scannable_source_path(rel_path, size):
            by_rel_path[rel_path] = file_path

    sources: Dict[str, str] = {}
    for rel_path in sorted(by_rel_path, key=_source_scan_order)[:max_files]:
        try:
            sources[rel_path] = by_rel_path[rel_path].read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if classify_runtime_from_source(sources) == VERIFIERS_V1:
            break
    return sources


def detect_environment_runtime(
    env_path: Path,
    files: Iterable[Path],
    requires_dist: Optional[Iterable[str]] = None,
    dependencies: Optional[Iterable[str]] = None,
) -> Optional[str]:
    """The ``runtime_hint`` to send with a push, or ``None`` to omit it.

    Source markers beat the verifiers pin floor, matching the server-side
    classifier; with no signal at all the hint is omitted and the Hub
    classifies the package itself at unpack time.
    """
    from_source = classify_runtime_from_source(collect_sources(env_path, files))
    if from_source is not None:
        return from_source
    return classify_runtime_from_metadata(
        effective_requirement_strings(requires_dist, dependencies)
    )
