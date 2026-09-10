"""Which verifiers API an environment package targets (v0 vs v1), for ``prime env push``.

The Hub stores whatever the CLI declares (``--runtime v0|v1``) as the package's
runtime. Without the flag, the package's own ``verifiers`` requirement decides:
v1 shipped as verifiers 0.2.0, so a lower bound at or above 0.2.0 declares v1
and any other pin declares v0. Entries that only apply under an ``extra``
(``verifiers>=0.3; extra == "rl"``) are optional dependency groups a plain
install never resolves, so they say nothing about the runtime and are ignored.
No verifiers requirement (or a URL pin, which says nothing about the API) means
no hint is sent and the Hub lists the package as Unclassified until its owner
sets the runtime.

Mirrors the server's fallback (platform ``backend/app/utils/environment_runtime.py``).
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional

from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version

VERIFIERS_V0 = "VERIFIERS_V0"
VERIFIERS_V1 = "VERIFIERS_V1"

# First verifiers release that shipped the v1 API (`verifiers.v1`).
VERIFIERS_V1_MIN_VERSION = Version("0.2.0")

_RUNTIME_OPTIONS = {"v0": VERIFIERS_V0, "v1": VERIFIERS_V1}

# Markers that mention the ``extra`` variable are evaluated as a plain install
# sees them (no extra selected); ``packaging`` normalizes markers to a canonical
# string, so a word-boundary match finds the variable exactly.
_EXTRA_MARKER = re.compile(r"\bextra\b")


def parse_runtime_option(value: Optional[str]) -> Optional[str]:
    """``--runtime`` value → Hub runtime, or None when the flag was not given.

    Raises ``ValueError`` for anything other than ``v0`` / ``v1`` (case-insensitive).
    """
    if value is None:
        return None
    runtime = _RUNTIME_OPTIONS.get(value.strip().lower())
    if runtime is None:
        raise ValueError(f"--runtime must be 'v0' or 'v1', got {value!r}")
    return runtime


def _is_extra_guarded(requirement: Requirement) -> bool:
    """True when the requirement is not part of a plain (no-extra) install.

    Only markers that mention ``extra`` are judged, evaluated with no extra
    selected: ``extra == "rl"`` is false there and the entry is dropped, while
    ``extra != "rl"`` or ``extra == "rl" or python_version >= "3.10"`` hold and
    the entry counts. Other markers are left as best-effort floors.
    """
    marker = requirement.marker
    if marker is None or not _EXTRA_MARKER.search(str(marker)):
        return False
    try:
        return not marker.evaluate({"extra": ""})
    except Exception:  # pragma: no cover - defensive: unknown marker variable
        return True


def find_verifiers_requirement(requirement_strings: Iterable[str]) -> Optional[Requirement]:
    """Effective ``verifiers`` requirement: entries that only apply under an
    ``extra`` are ignored, unconditional entries beat marker-guarded ones, and
    among those the highest floor wins (repeated entries intersect)."""
    requirements: List[Requirement] = []
    for text in requirement_strings:
        try:
            requirement = Requirement(text.strip())
        except InvalidRequirement:
            continue
        if requirement.name.lower() == "verifiers" and not _is_extra_guarded(requirement):
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
