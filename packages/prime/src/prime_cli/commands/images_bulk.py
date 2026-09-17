import json
import re
import shutil
import sys
import tarfile
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import httpx
import typer
from gitignore_parser import parse_gitignore
from prime_sandboxes import (
    APIClient,
    APIError,
    Config,
    ImageVisibility,
    SourceImageBuildResult,
    UnauthorizedError,
)
from prime_sandboxes.image_references import is_docker_hub_reference
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from ..utils import get_console

console = get_console()

# Use a synthetic archive path to avoid collisions with Dockerfiles already in the context.
PACKAGED_DOCKERFILE_PATH = ".__prime_dockerfile__"

DEFAULT_MAX_IN_FLIGHT = 64
DEFAULT_BUILD_TIMEOUT_SECONDS = 1800
POLL_INTERVAL_SECONDS = 10.0
UPLOAD_TIMEOUT_SECONDS = 600.0

# Backoff schedule for rate-limit 429s
RATE_LIMIT_MAX_ATTEMPTS = 5
RATE_LIMIT_BACKOFF_INITIAL_SECONDS = 2.0
RATE_LIMIT_BACKOFF_MAX_SECONDS = 60.0

# Max consecutive deferrals without a single successful submission
MAX_CONSECUTIVE_SUBMIT_DEFERRALS = 20

FAILURE_TABLE_MAX_ROWS = 20

_TAG_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$")
_BUILD_MANIFEST_KEYS = {"image", "context", "dockerfile"}
_SOURCE_MANIFEST_KEY = "source"


class BulkPushValidationError(Exception):
    """Raised when the build list cannot be resolved; carries all problems."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


class QuotaExceededError(APIError):
    """A 429 caused by the wallet's image count/storage quota, not the rate limiter."""


class SubmitRateLimited(Exception):
    """Raised by a submit callable to defer the spec instead of failing it."""

    def __init__(self, message: str, retry_after: float):
        super().__init__(message)
        self.retry_after = retry_after


class BulkRunInterrupted(Exception):
    """Ctrl+C during a bulk run.

    Carries the outcomes recorded so far plus every never-submitted spec
    (marked SKIPPED), so the command can write a resume manifest. Jobs already
    in flight are not included — they keep running server-side.
    """

    def __init__(self, outcomes: list["BuildOutcome"]):
        super().__init__("bulk run interrupted")
        self.outcomes = outcomes


@dataclass
class BuildSpec:
    """One resolved build: where the context lives and what to call the image."""

    image_name: str
    image_tag: str
    context: Path
    dockerfile: Path
    source: str

    @property
    def image_ref(self) -> str:
        return f"{self.image_name}:{self.image_tag}"

    def to_manifest_line(self) -> dict[str, str]:
        """Serialize as a manifest entry (absolute paths, so it re-runs from any cwd)."""
        return {
            "image": self.image_ref,
            "context": str(self.context),
            "dockerfile": str(self.dockerfile),
        }


@dataclass
class BuildOutcome:
    """Terminal result for one spec.

    ``spec`` is any object with ``image_ref``, ``source`` and
    ``to_manifest_line()`` (BuildSpec here, TransferSpec for source-image builds).
    ``status`` is a backend terminal status (COMPLETED/FAILED/CANCELLED) or a
    client-side one: SUBMIT_FAILED (initiate/upload/start failed), TIMEOUT
    (no terminal status within --build-timeout), SKIPPED (never submitted
    because the quota was exhausted mid-run).
    """

    spec: Any
    status: str
    build_id: Optional[str] = None
    full_image_path: Optional[str] = None
    error: Optional[str] = None


def package_build_context(context_path: Path, dockerfile_path: Path) -> str:
    """Create a tar.gz of the build context with the Dockerfile packaged at
    ``PACKAGED_DOCKERFILE_PATH``.

    Returns the temp tar path; the caller must unlink it. A .dockerignore
    matcher is applied so ignored paths (e.g. local .venv, node_modules)
    aren't uploaded. BuildKit looks for <Dockerfile>.dockerignore next to the
    Dockerfile first and falls back to <context>/.dockerignore, so mirror that.
    """
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
        tar_path = tmp_file.name

    per_dockerfile_ignore = dockerfile_path.with_name(dockerfile_path.name + ".dockerignore")
    root_dockerignore = context_path / ".dockerignore"
    if per_dockerfile_ignore.is_file():
        dockerignore_path: Optional[Path] = per_dockerfile_ignore
    elif root_dockerignore.is_file():
        dockerignore_path = root_dockerignore
    else:
        dockerignore_path = None
    ignore_matcher = (
        parse_gitignore(str(dockerignore_path), base_dir=str(context_path))
        if dockerignore_path is not None
        else None
    )

    def tar_filter(tarinfo: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
        if ignore_matcher is None:
            return tarinfo
        rel = tarinfo.name
        if rel.startswith("./"):
            rel = rel[2:]
        if not rel or rel == ".":
            return tarinfo
        if ignore_matcher(str(context_path / rel)):
            return None
        return tarinfo

    try:
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(context_path, arcname=".", filter=tar_filter)
            tar.add(dockerfile_path, arcname=PACKAGED_DOCKERFILE_PATH)
    except Exception:
        Path(tar_path).unlink(missing_ok=True)
        raise
    return tar_path


# How long to pause new submissions after the server's source-build rate limiter
# rejects one.
SOURCE_RATE_LIMIT_PAUSE_SECONDS = 15.0

_SOURCE_MANIFEST_KEYS = {"source", "image"}

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass
class SourceBuildSpec:
    """One resolved source build: a registry image and its Prime destination.

    ``dest_name``/``dest_tag`` name the resulting VM image. When ``override``
    is False they are derived from the source the same way the server derives
    them and are only used for display and duplicate detection; when True the
    user chose them and they are sent with the request.
    """

    source_image: str
    dest_name: str
    dest_tag: str
    source: str
    override: bool = False

    @property
    def image_ref(self) -> str:
        return f"{self.dest_name}:{self.dest_tag}"

    @property
    def is_docker_hub(self) -> bool:
        return is_docker_hub_reference(self.source_image)

    def to_manifest_line(self) -> dict[str, str]:
        line = {"source": self.source_image}
        if self.override:
            line["image"] = self.image_ref
        return line


def derive_source_destination(source_ref: str, *, keep_namespace: bool = False) -> tuple[str, str]:
    """Derive a source build's destination when none is given.

    Docker Hub sources always keep their namespace because they become org-less
    platform images automatically. The implicit ``library/`` namespace is
    removed. ``keep_namespace`` applies the same platform naming rule to an
    explicit non-Docker-Hub registry source. Otherwise only the last path segment
    is kept for a personal or team destination.
    """
    ref = (source_ref or "").strip()
    if not ref:
        raise ValueError("empty image reference")
    # A comma is never valid inside an image reference.
    if "," in ref:
        raise ValueError("commas are not allowed; use one entry per image reference")

    digest: Optional[str] = None
    if "@" in ref:
        ref, _, digest = ref.rpartition("@")
        if not _DIGEST_RE.match(digest):
            raise ValueError("unsupported digest (only sha256 is supported)")

    first_segment = ref.split("/", 1)[0]
    has_registry = "/" in ref and (
        "." in first_segment or ":" in first_segment or first_segment == "localhost"
    )
    path_part = ref[len(first_segment) + 1 :] if has_registry else ref
    if not path_part:
        raise ValueError("missing repository")

    tag: Optional[str] = None
    if ":" in path_part:
        path_part, _, tag = path_part.rpartition(":")
        if not tag:
            raise ValueError("empty tag")
    elif digest is None:
        tag = "latest"

    repository = path_part.strip("/")
    if not repository:
        raise ValueError("missing repository")

    docker_hub = is_docker_hub_reference(source_ref)
    if docker_hub and repository.lower().startswith("library/"):
        repository = repository.split("/", 1)[1]
    name = repository.lower() if keep_namespace or docker_hub else repository.split("/")[-1].lower()
    if tag is None:
        assert digest is not None
        tag = "sha256-" + digest.split(":", 1)[1][:16]
    return name, tag


_DUPLICATE_DEST_HINT = (
    ' — build them via a JSONL manifest with distinct "image" destination overrides'
)


def parse_source_manifest_row(
    entry: dict[str, Any], where: str, *, platform_image: bool = False
) -> list[str] | SourceBuildSpec:
    """Validate one ``{"source": ..., "image"?: ...}`` manifest row.

    Returns the resolved spec, or a list of problems for the caller to append.
    """
    unknown = sorted(set(entry) - _SOURCE_MANIFEST_KEYS)
    if unknown:
        return [
            f"{where}: unknown key(s) {', '.join(unknown)} "
            f"(expected: {', '.join(sorted(_SOURCE_MANIFEST_KEYS))})"
        ]

    source = entry.get("source")
    if not source or not isinstance(source, str):
        return [f"{where}: 'source' is required"]
    source = source.strip()

    docker_hub = is_docker_hub_reference(source)
    try:
        derived_name, derived_tag = derive_source_destination(source, keep_namespace=platform_image)
    except ValueError as e:
        return [f"{where}: invalid source '{source}' ({e})"]

    image = entry.get("image")
    if image is not None and not isinstance(image, str):
        return [f"{where}: 'image' must be a string"]
    if image:
        if docker_hub:
            return [f"{where}: Docker Hub source builds do not accept a custom destination"]
        if ":" in image:
            dest_name, dest_tag = image.rsplit(":", 1)
        else:
            dest_name, dest_tag = image, "latest"
        # '/' separates owner from name in personal/team image paths, so
        # only platform images (org-less, stored under their source repository
        # namespace) may use a single-level namespace in the destination.
        segments = dest_name.split("/")
        if len(segments) > (2 if platform_image else 1) or not all(segments):
            hint = (
                "use 'name:tag' or a namespaced 'ns/name:tag'"
                if platform_image
                else "use simple names like 'myapp:v1'"
            )
            return [f"{where}: invalid destination '{image}'; {hint}"]
        if not _TAG_RE.match(dest_tag):
            return [f"{where}: invalid destination tag '{dest_tag}'"]
        override = True
    else:
        dest_name, dest_tag = derived_name, derived_tag
        override = False

    return SourceBuildSpec(
        source_image=source,
        dest_name=dest_name,
        dest_tag=dest_tag,
        source=where,
        override=override,
    )


def load_source_specs_from_manifest_entries(
    entries: list[tuple[int, dict[str, Any]]], manifest_name: str, *, platform_image: bool = False
) -> list[SourceBuildSpec]:
    """Resolve and fully validate source rows from a parsed JSONL manifest.

    ``entries`` pairs each row with its 1-based line number. Rows with problems
    raise BulkPushValidationError with every problem at once.
    """
    problems: list[str] = []
    specs: list[SourceBuildSpec] = []
    for lineno, entry in entries:
        result = parse_source_manifest_row(
            entry, f"{manifest_name}:{lineno}", platform_image=platform_image
        )
        if isinstance(result, list):
            problems.extend(result)
        else:
            specs.append(result)

    problems.extend(_duplicate_ref_problems(specs, hint=_DUPLICATE_DEST_HINT))
    if problems:
        raise BulkPushValidationError(problems)
    return specs


def load_hf_source_specs(
    dataset: str,
    *,
    config: Optional[str],
    split: str,
    column: Optional[str],
    platform_image: bool = False,
) -> tuple[list[SourceBuildSpec], list[str]]:
    """Resolve source-build specs from a Hugging Face dataset column.

    Pages the dataset through the datasets-server rows API (no local
    `datasets` dependency), dedupes identical references preserving order,
    and returns (specs, notes) where notes are informational messages.
    """
    # Imported here: images_hf imports BuildSpec helpers from this module.
    from .images_hf import (
        PARTIAL_DATASET_NOTE,
        check_split,
        iter_hf_rows,
        require_string_column,
        select_hf_config,
        warn_if_large,
    )

    ds = select_hf_config(dataset, config)
    dataset_id = ds.dataset_id
    notes: list[str] = []

    require_string_column(ds, column, reason="it cannot hold image references")
    check_split(ds, split)
    if ds.partial:
        notes.append(PARTIAL_DATASET_NOTE)
    warn_if_large(ds)

    sources: list[tuple[str, int]] = []  # (image ref, first row index), order-preserving
    seen: set[str] = set()
    scanned = 0
    empty_rows = 0
    for row_idx, row in iter_hf_rows(ds, split):
        value = row.get(column)
        if not isinstance(value, str) or not value.strip():
            empty_rows += 1
            continue
        scanned += 1
        value = value.strip()
        if value in seen:
            continue
        seen.add(value)
        sources.append((value, row_idx))

    if empty_rows:
        notes.append(f"Skipped {empty_rows} row(s) with an empty '{column}' value")
    duplicates = scanned - len(sources)
    if duplicates:
        notes.append(f"Collapsed {duplicates} duplicate image reference(s)")

    problems: list[str] = []
    specs: list[SourceBuildSpec] = []
    for ref, row_idx in sources:
        try:
            dest_name, dest_tag = derive_source_destination(ref, keep_namespace=platform_image)
        except ValueError as e:
            problems.append(f"{dataset_id} row {row_idx}: invalid image reference '{ref}' ({e})")
            continue
        specs.append(
            SourceBuildSpec(
                source_image=ref,
                dest_name=dest_name,
                dest_tag=dest_tag,
                source=f"row {row_idx}",
                override=False,
            )
        )

    problems.extend(_duplicate_ref_problems(specs, hint=_DUPLICATE_DEST_HINT))
    if not specs and not problems:
        problems.append(f"no image references found in '{dataset_id}' column '{column}'")
    if problems:
        raise BulkPushValidationError(problems)
    return specs, notes


def submit_source_build(
    client: APIClient,
    spec: SourceBuildSpec,
    *,
    team_id: Optional[str],
    visibility: Optional[ImageVisibility],
    owner_scope: Optional[str] = None,
) -> tuple[str, str]:
    """Queue one source-image VM build. Returns (build_id, full_image_path).

    Source builds are counted per image by the server's rate limiter
    (a rolling window), so a plain 429 means "later", not "failed": it is
    surfaced as SubmitRateLimited so the engine requeues the spec. Wallet
    quota 429s become QuotaExceededError and stop the run.
    """
    payload: dict[str, Any] = {
        "source_image": spec.source_image,
        "platform": "linux/amd64",
    }
    if spec.override:
        payload["image_name"] = spec.dest_name
        payload["image_tag"] = spec.dest_tag
    if team_id:
        payload["team_id"] = team_id
    if visibility is not None:
        payload["visibility"] = visibility.value
    if owner_scope is not None:
        payload["owner_scope"] = owner_scope

    try:
        response = client.request("POST", "/images/build", json=payload)
    except UnauthorizedError:
        raise
    except APIError as e:
        if _is_quota_429(e):
            raise QuotaExceededError(str(e)) from e
        if _is_http_429(e):
            raise SubmitRateLimited(str(e), retry_after=SOURCE_RATE_LIMIT_PAUSE_SECONDS) from e
        raise

    # Single-source builds return a top-level build_id today. Also accept the
    # bulk shape without silently dropping results if the server contract shifts.
    results = response.get("results")
    if isinstance(results, list):
        # Each spec holds one source, so any count other than one is invalid.
        if len(results) != 1 or not isinstance(results[0], dict):
            raise APIError(
                "invalid response from server "
                f"(expected one source-build result, got {len(results)})"
            )
        entry = SourceImageBuildResult.model_validate(results[0])
        if entry.build is None:
            error = entry.error or "invalid response from server (source build not queued)"
            if any(marker in error.lower() for marker in _QUOTA_DETAIL_MARKERS):
                raise QuotaExceededError(error)
            raise APIError(error)
        return entry.build.build_id, entry.build.full_image_path

    build_id = response.get("build_id") or response.get("buildId")
    if not build_id:
        raise APIError("invalid response from server (missing build_id)")
    return build_id, response.get("fullImagePath") or spec.image_ref


# ---------------------------------------------------------------------------
# Build-list resolution: JSONL manifest
# ---------------------------------------------------------------------------


def _duplicate_ref_problems(specs: list[Any], hint: str = "") -> list[str]:
    """Report specs (anything with image_ref/source) sharing a destination ref."""
    by_ref: dict[str, list[str]] = {}
    for spec in specs:
        by_ref.setdefault(spec.image_ref, []).append(spec.source)
    return [
        f"duplicate image reference '{ref}' ({', '.join(sources)}){hint}"
        for ref, sources in by_ref.items()
        if len(sources) > 1
    ]


def load_manifest(
    manifest_path: Path, *, platform_image: bool = False
) -> tuple[list[BuildSpec], list[SourceBuildSpec]]:
    """Parse and fully validate a JSONL manifest.

    Each row is either a Dockerfile build (``{"image", "context",
    "dockerfile"?}`` with paths relative to the manifest file) or a
    public-registry source build (``{"source", "image"?}``, where "image"
    optionally overrides the derived destination).
    """
    base = manifest_path.parent
    problems: list[str] = []
    specs: list[BuildSpec] = []
    source_entries: list[tuple[int, dict[str, Any]]] = []

    for lineno, raw_line in enumerate(manifest_path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        where = f"{manifest_path.name}:{lineno}"
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as e:
            problems.append(f"{where}: invalid JSON ({e})")
            continue
        if not isinstance(entry, dict):
            problems.append(f"{where}: expected a JSON object")
            continue

        if _SOURCE_MANIFEST_KEY in entry:
            source_entries.append((lineno, entry))
            continue

        unknown = sorted(set(entry) - _BUILD_MANIFEST_KEYS)
        if unknown:
            problems.append(
                f"{where}: unknown key(s) {', '.join(unknown)} "
                f"(expected: {', '.join(sorted(_BUILD_MANIFEST_KEYS))})"
            )
            continue

        image = entry.get("image")
        context = entry.get("context")
        if not image or not isinstance(image, str):
            problems.append(f"{where}: 'image' is required")
            continue
        if not context or not isinstance(context, str):
            problems.append(f"{where}: 'context' is required")
            continue
        if ":" in image:
            image_name, image_tag = image.rsplit(":", 1)
        else:
            image_name, image_tag = image, "latest"
        if not image_name or not image_tag:
            problems.append(f"{where}: invalid image reference '{image}'")
            continue
        if "/" in image_name:
            problems.append(
                f"{where}: image name cannot contain '/' ('{image_name}'); "
                "use simple names like 'myapp:v1'"
            )
            continue
        if not _TAG_RE.match(image_tag):
            problems.append(f"{where}: invalid image tag '{image_tag}'")
            continue

        dockerfile = entry.get("dockerfile")
        if dockerfile is not None and not isinstance(dockerfile, str):
            problems.append(f"{where}: 'dockerfile' must be a string")
            continue

        context_path = (base / context).resolve()
        dockerfile_path = (
            (base / dockerfile).resolve() if dockerfile else context_path / "Dockerfile"
        )
        if not context_path.is_dir():
            problems.append(f"{where}: build context is not a directory: {context_path}")
            continue
        if not dockerfile_path.is_file():
            problems.append(f"{where}: Dockerfile not found: {dockerfile_path}")
            continue

        specs.append(
            BuildSpec(
                image_name=image_name,
                image_tag=image_tag,
                context=context_path,
                dockerfile=dockerfile_path,
                source=where,
            )
        )

    try:
        source_specs = load_source_specs_from_manifest_entries(
            source_entries, manifest_path.name, platform_image=platform_image
        )
    except BulkPushValidationError as e:
        problems.extend(e.problems)
        source_specs = []

    problems.extend(_duplicate_ref_problems(specs + source_specs))
    if not specs and not source_specs and not problems:
        problems.append(f"{manifest_path.name}: manifest contains no builds")
    if problems:
        raise BulkPushValidationError(problems)
    return specs, source_specs


# ---------------------------------------------------------------------------
# Build-list resolution: Harbor task directories
# ---------------------------------------------------------------------------


def _is_harbor_task_dir(path: Path) -> bool:
    return (path / "task.toml").is_file() and (path / "environment").is_dir()


def discover_harbor_tasks(root: Path) -> list[Path]:
    """Return Harbor task directories under root (or root itself if it is a task)."""
    if _is_harbor_task_dir(root):
        return [root]
    return sorted(
        (p for p in root.iterdir() if p.is_dir() and _is_harbor_task_dir(p)),
        key=lambda p: p.name,
    )


def sanitize_image_name(raw: str) -> str:
    """Normalize to a valid image name: lowercase [a-z0-9._-], alphanumeric ends."""
    name = raw.strip().lower()
    name = re.sub(r"[^a-z0-9._-]+", "-", name)
    name = re.sub(r"^[^a-z0-9]+", "", name)
    name = re.sub(r"[^a-z0-9]+$", "", name)
    return name


def _render_name_template(template: str, *, task_dir_name: str, toml_name: Optional[str]) -> str:
    values = {"dir": task_dir_name, "name": toml_name or task_dir_name}
    try:
        rendered = template.format(**values)
    except (KeyError, IndexError):
        raise BulkPushValidationError(
            [
                f"invalid --name-template '{template}': "
                "supported placeholders are {dir} and {name}"
            ]
        )
    return sanitize_image_name(rendered)


def load_harbor_specs(
    root: Path, *, tag: str, name_template: str, platform_image: bool = False
) -> tuple[list[BuildSpec], list[SourceBuildSpec], list[tuple[str, str]]]:
    """Resolve build and source specs from a Harbor tasks directory.

    Tasks with a prebuilt ``[environment] docker_image`` become public-registry
    source builds; tasks with an ``environment/Dockerfile`` become Dockerfile
    builds. Returns (build_specs, source_specs, skipped) where skipped lists
    (task name, reason) for tasks with nothing to build (e.g. compose-only).
    Tasks sharing the same prebuilt image collapse into one source build.
    """
    tasks = discover_harbor_tasks(root)
    if not tasks:
        raise BulkPushValidationError(
            [
                f"no Harbor tasks found under {root} "
                "(a task directory contains task.toml and environment/)"
            ]
        )

    problems: list[str] = []
    specs: list[BuildSpec] = []
    source_specs: list[SourceBuildSpec] = []
    skipped: list[tuple[str, str]] = []
    first_task_by_source: dict[str, str] = {}
    for task_dir in tasks:
        try:
            with open(task_dir / "task.toml", "rb") as f:
                config = tomllib.load(f)
        except Exception as e:
            problems.append(f"{task_dir.name}: failed to parse task.toml ({e})")
            continue

        environment = config.get("environment") or {}
        docker_image = environment.get("docker_image") if isinstance(environment, dict) else None
        if docker_image and isinstance(docker_image, str):
            docker_image = docker_image.strip()
            first_task = first_task_by_source.get(docker_image)
            if first_task is not None:
                skipped.append((task_dir.name, f"same image as {first_task} ({docker_image})"))
                continue
            first_task_by_source[docker_image] = task_dir.name
            try:
                dest_name, dest_tag = derive_source_destination(
                    docker_image, keep_namespace=platform_image
                )
            except ValueError as e:
                problems.append(f"{task_dir.name}: invalid docker_image '{docker_image}' ({e})")
                continue
            source_specs.append(
                SourceBuildSpec(
                    source_image=docker_image,
                    dest_name=dest_name,
                    dest_tag=dest_tag,
                    source=task_dir.name,
                    override=False,
                )
            )
            continue

        dockerfile = task_dir / "environment" / "Dockerfile"
        if not dockerfile.is_file():
            skipped.append((task_dir.name, "no environment/Dockerfile"))
            continue

        task_section = config.get("task")
        toml_name = task_section.get("name") if isinstance(task_section, dict) else None
        image_name = _render_name_template(
            name_template, task_dir_name=task_dir.name, toml_name=toml_name
        )
        if not image_name:
            problems.append(f"{task_dir.name}: image name is empty after sanitization")
            continue

        specs.append(
            BuildSpec(
                image_name=image_name,
                image_tag=tag,
                context=task_dir / "environment",
                dockerfile=dockerfile,
                source=task_dir.name,
            )
        )

    problems.extend(_duplicate_ref_problems(specs, hint=" — use --name-template to disambiguate"))
    if not specs and not source_specs and not problems:
        problems.append(
            f"no buildable tasks under {root} "
            f"({len(skipped)} skipped: {', '.join(name for name, _ in skipped)})"
        )
    if problems:
        raise BulkPushValidationError(problems)
    return specs, source_specs, skipped


# ---------------------------------------------------------------------------
# Submission + polling
# ---------------------------------------------------------------------------

_QUOTA_DETAIL_MARKERS = ("image limit exceeded", "image storage limit exceeded")


def _is_http_429(error: APIError) -> bool:
    return "HTTP 429" in str(error)


def _is_quota_429(error: APIError) -> bool:
    message = str(error).lower()
    return _is_http_429(error) and any(marker in message for marker in _QUOTA_DETAIL_MARKERS)


def _request_with_rate_limit_retry(
    client: APIClient, method: str, path: str, *, json_body: dict[str, Any]
) -> dict[str, Any]:
    """client.request with exponential backoff on rate-limit 429s.

    Wallet-quota 429s are re-raised as QuotaExceededError immediately — they
    cannot succeed on retry and the caller must stop submitting new builds.
    """
    delay = RATE_LIMIT_BACKOFF_INITIAL_SECONDS
    for attempt in range(1, RATE_LIMIT_MAX_ATTEMPTS + 1):
        try:
            return client.request(method, path, json=json_body)
        except UnauthorizedError:
            raise
        except APIError as e:
            if _is_quota_429(e):
                raise QuotaExceededError(str(e)) from e
            if not _is_http_429(e) or attempt == RATE_LIMIT_MAX_ATTEMPTS:
                raise
            time.sleep(delay)
            delay = min(delay * 2, RATE_LIMIT_BACKOFF_MAX_SECONDS)
    raise AssertionError("unreachable")


def _submit_build(
    client: APIClient,
    spec: BuildSpec,
    *,
    team_id: Optional[str],
    visibility: Optional[ImageVisibility],
) -> tuple[str, str]:
    """Run the initiate -> upload context -> start flow for one build.

    Returns (build_id, full_image_path).
    """
    tar_path = package_build_context(spec.context, spec.dockerfile)
    try:
        payload: dict[str, Any] = {
            "image_name": spec.image_name,
            "image_tag": spec.image_tag,
            "dockerfile_path": PACKAGED_DOCKERFILE_PATH,
            "platform": "linux/amd64",
        }
        if team_id:
            payload["team_id"] = team_id
        if visibility is not None:
            payload["visibility"] = visibility.value

        response = _request_with_rate_limit_retry(
            client, "POST", "/images/build", json_body=payload
        )
        build_id = response.get("build_id")
        upload_url = response.get("upload_url")
        expires_in = response.get("expires_in")
        if not build_id or not upload_url or expires_in is None:
            raise APIError(
                "invalid response from server (missing build_id, upload_url, or expires_in)"
            )
        full_image_path = response.get("fullImagePath") or spec.image_ref

        with open(tar_path, "rb") as f:
            upload_response = httpx.put(
                upload_url,
                content=f,
                headers={"Content-Type": "application/octet-stream"},
                timeout=UPLOAD_TIMEOUT_SECONDS,
            )
            upload_response.raise_for_status()

        _request_with_rate_limit_retry(
            client,
            "POST",
            f"/images/build/{build_id}/start",
            json_body={"context_uploaded": True},
        )
        return build_id, full_image_path
    finally:
        Path(tar_path).unlink(missing_ok=True)


@dataclass
class _InFlightBuild:
    spec: Any
    full_image_path: str
    deadline: float


def run_bulk_jobs(
    client: APIClient,
    specs: list[Any],
    *,
    submit: Callable[[Any], tuple[str, str]],
    concurrency: int,
    build_timeout: int,
    progress_description: str = "Pushing images",
) -> list[BuildOutcome]:
    """Submit jobs with a sliding window and poll them to terminal status.

    ``submit(spec)`` starts one server-side build job and returns
    (build_id, full_image_path). A finished job immediately frees a slot for
    the next queued spec, so one slow job never blocks the rest of a "batch".
    Once the wallet quota is hit (QuotaExceededError), submission stops
    (remaining specs become SKIPPED) but jobs already in flight are still
    polled to completion. A submit that raises SubmitRateLimited requeues its
    spec and pauses new submissions for ``retry_after`` seconds while polling
    continues; after MAX_CONSECUTIVE_SUBMIT_DEFERRALS deferrals with no
    successful submission in between, the run stops submitting the same way
    the quota path does, so a persistently 429ing server cannot stall it
    forever.
    """
    pending: deque[Any] = deque(specs)
    in_flight: dict[str, _InFlightBuild] = {}
    outcomes: list[BuildOutcome] = []
    stop_skip_reason: Optional[str] = None
    submit_gate = 0.0
    consecutive_deferrals = 0
    rate_limit_notice_shown = False

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )
    with progress:
        task_id = progress.add_task(progress_description, total=len(specs))

        def record(outcome: BuildOutcome) -> None:
            outcomes.append(outcome)
            progress.advance(task_id)
            if outcome.status == "COMPLETED":
                progress.console.print(
                    f"[green]✓[/green] {outcome.full_image_path or outcome.spec.image_ref}"
                )
            elif outcome.status != "SKIPPED":  # skips are summarized once at the end
                progress.console.print(
                    f"[red]✗ {outcome.spec.image_ref}[/red] "
                    f"[dim]({outcome.status.lower()}: {outcome.error})[/dim]"
                )

        try:
            while pending or in_flight:
                while pending and stop_skip_reason is None and len(in_flight) < concurrency:
                    if time.monotonic() < submit_gate:
                        break
                    spec = pending.popleft()
                    try:
                        build_id, full_image_path = submit(spec)
                    except KeyboardInterrupt:
                        # The submit was cut short and may have half-queued the
                        # job; treating it as never submitted keeps it in the
                        # resume manifest.
                        pending.appendleft(spec)
                        raise
                    except SubmitRateLimited as e:
                        consecutive_deferrals += 1
                        if consecutive_deferrals > MAX_CONSECUTIVE_SUBMIT_DEFERRALS:
                            record(
                                BuildOutcome(
                                    spec=spec,
                                    status="SUBMIT_FAILED",
                                    error=(
                                        f"{e} (gave up after {MAX_CONSECUTIVE_SUBMIT_DEFERRALS} "
                                        "rate-limit deferrals with no successful submission)"
                                    ),
                                )
                            )
                            stop_skip_reason = (
                                "not submitted: the server kept rate-limiting submissions"
                            )
                            break
                        pending.appendleft(spec)
                        submit_gate = time.monotonic() + e.retry_after
                        if not rate_limit_notice_shown:
                            rate_limit_notice_shown = True
                            progress.console.print(
                                "[dim]Server rate limit reached; pacing submissions "
                                "(jobs already submitted keep running)...[/dim]"
                            )
                        break
                    except QuotaExceededError as e:
                        stop_skip_reason = "not submitted: image quota exceeded"
                        record(BuildOutcome(spec=spec, status="SUBMIT_FAILED", error=str(e)))
                    except UnauthorizedError:
                        raise
                    except (APIError, httpx.HTTPError, OSError) as e:
                        consecutive_deferrals = 0
                        record(BuildOutcome(spec=spec, status="SUBMIT_FAILED", error=str(e)))
                    else:
                        consecutive_deferrals = 0
                        in_flight[build_id] = _InFlightBuild(
                            spec=spec,
                            full_image_path=full_image_path,
                            deadline=time.monotonic() + build_timeout,
                        )

                if stop_skip_reason is not None and pending:
                    for spec in pending:
                        record(
                            BuildOutcome(
                                spec=spec,
                                status="SKIPPED",
                                error=stop_skip_reason,
                            )
                        )
                    pending.clear()

                if not pending and not in_flight:
                    break

                if not in_flight:
                    # Everything left is waiting on the submit gate.
                    time.sleep(max(0.0, submit_gate - time.monotonic()))
                    continue

                time.sleep(POLL_INTERVAL_SECONDS)

                for build_id, entry in list(in_flight.items()):
                    build_error: Optional[str] = None
                    try:
                        status_response = client.request("GET", f"/images/build/{build_id}")
                        build_status = str(status_response.get("status") or "")
                        error_message = status_response.get("errorMessage") or status_response.get(
                            "error_message"
                        )
                        if isinstance(error_message, str) and error_message.strip():
                            build_error = error_message.strip()
                    except UnauthorizedError:
                        raise
                    except APIError:
                        build_status = ""  # transient poll failure; the deadline still applies

                    if build_status in {"COMPLETED", "FAILED", "CANCELLED"}:
                        del in_flight[build_id]
                        failure_reason = f"build ended as {build_status}"
                        if build_error:
                            failure_reason += f": {build_error}"
                        record(
                            BuildOutcome(
                                spec=entry.spec,
                                status=build_status,
                                build_id=build_id,
                                full_image_path=entry.full_image_path,
                                error=None if build_status == "COMPLETED" else failure_reason,
                            )
                        )
                    elif time.monotonic() > entry.deadline:
                        del in_flight[build_id]
                        record(
                            BuildOutcome(
                                spec=entry.spec,
                                status="TIMEOUT",
                                build_id=build_id,
                                full_image_path=entry.full_image_path,
                                error=(
                                    f"no terminal status after {build_timeout}s "
                                    "(the build may still finish server-side)"
                                ),
                            )
                        )
        except KeyboardInterrupt:
            for spec in pending:
                outcomes.append(
                    BuildOutcome(
                        spec=spec, status="SKIPPED", error="not submitted: run interrupted"
                    )
                )
            raise BulkRunInterrupted(outcomes) from None

    return outcomes


def run_bulk_push(
    client: APIClient,
    build_specs: list[BuildSpec],
    source_specs: list[SourceBuildSpec],
    *,
    team_id: Optional[str],
    visibility: Optional[ImageVisibility],
    platform_image: bool,
    concurrency: int,
    build_timeout: int,
) -> list[BuildOutcome]:
    """Run Dockerfile and source builds through the shared engine.

    Dockerfile builds submit their packaged context; source builds queue a
    server-side public-registry VM build. Docker Hub sources (and explicit
    --platform-image sources) become public, org-less platform images: no
    team, forced PUBLIC visibility, platform owner scope.
    """
    specs: list[Any] = [*build_specs, *source_specs]

    def submit(spec: Any) -> tuple[str, str]:
        if isinstance(spec, BuildSpec):
            return _submit_build(client, spec, team_id=team_id, visibility=visibility)
        docker_hub = spec.is_docker_hub
        platform = platform_image or docker_hub
        return submit_source_build(
            client,
            spec,
            team_id=None if platform else team_id,
            visibility=ImageVisibility.PUBLIC if platform else visibility,
            owner_scope="platform" if platform else None,
        )

    return run_bulk_jobs(
        client,
        specs,
        submit=submit,
        concurrency=concurrency,
        build_timeout=build_timeout,
    )


def _write_failures_manifest(path: Path, failures: list[BuildOutcome]) -> None:
    lines = [json.dumps(outcome.spec.to_manifest_line()) for outcome in failures]
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


def push_bulk(
    manifest: Optional[Path] = typer.Option(
        None,
        "--manifest",
        "-m",
        help=(
            "JSONL manifest; each line is a Dockerfile build "
            '{"image": "name:tag", "context": "./dir", "dockerfile"?: "..."} '
            "with paths relative to the manifest file, or a public-registry source build "
            '{"source": "registry/repo:tag", "image"?: "name:tag"}'
        ),
    ),
    harbor: Optional[Path] = typer.Option(
        None,
        "--harbor",
        help=(
            "Harbor task directory (or directory of tasks); each task's "
            "environment/ folder is the build context, and tasks with a prebuilt "
            "[environment] docker_image build from that registry image instead"
        ),
    ),
    hf_dataset: Optional[str] = typer.Option(
        None,
        "--hf",
        help=(
            "Hugging Face dataset id or URL (e.g. 'org/dataset'); builds every "
            "Dockerfile stored in the dataset, or every registry image referenced "
            "in a column (--column)"
        ),
    ),
    hf_split: str = typer.Option("train", "--hf-split", help="Dataset split for --hf mode"),
    hf_config: Optional[str] = typer.Option(
        None,
        "--hf-config",
        help="Dataset config (called 'subset' in the HF viewer) for --hf mode",
        show_default="the dataset's only config",
    ),
    dockerfile_column: Optional[str] = typer.Option(
        None,
        "--dockerfile-column",
        help=(
            "Dataset column holding Dockerfile contents (required for --hf mode "
            "without --column; e.g. 'dockerfile')"
        ),
    ),
    name_column: Optional[str] = typer.Option(
        None,
        "--name-column",
        help=(
            "Dataset column naming each image (required for --hf mode without "
            "--column; e.g. 'instance_id'; values are sanitized to valid image names)"
        ),
    ),
    source_column: Optional[str] = typer.Option(
        None,
        "--column",
        help=(
            "Dataset column holding registry image references (required for --hf "
            "mode without --dockerfile-column; e.g. 'docker_image')"
        ),
    ),
    tag: str = typer.Option(
        "latest", "--tag", help="Image tag for Harbor and --hf Dockerfile modes"
    ),
    name_template: str = typer.Option(
        "{dir}",
        "--name-template",
        help=(
            "Image name template for Harbor and --hf Dockerfile modes; placeholders: "
            "{dir} (task directory name) and {name} (task.toml [task].name; in --hf "
            "mode both are the --name-column value). The result is sanitized to a "
            "valid image name"
        ),
    ),
    platform_image: bool = typer.Option(
        False,
        "--platform-image",
        help=(
            "Build explicit non-Docker-Hub registry sources as org-less platform VM "
            "images (admins only; implies --public)"
        ),
    ),
    public: bool = typer.Option(
        False, "--public", help="Make the images public when the builds complete"
    ),
    private: bool = typer.Option(
        False, "--private", help="Make non-Docker Hub images private when the builds complete"
    ),
    concurrency: int = typer.Option(
        DEFAULT_MAX_IN_FLIGHT, "--concurrency", help="Maximum builds in flight at once"
    ),
    build_timeout: int = typer.Option(
        DEFAULT_BUILD_TIMEOUT_SECONDS,
        "--build-timeout",
        help="Seconds to wait for a single build before giving up on it",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Resolve and print the build list without pushing"
    ),
    failures_out: Path = typer.Option(
        Path("push-bulk-failures.jsonl"),
        "--failures-out",
        help="Where to write a re-runnable manifest of failed builds",
    ),
):
    """
    Build and push many image artifacts in one command.

    Reads builds from a JSONL manifest (--manifest), a Harbor tasks directory
    (--harbor), or a Hugging Face dataset (--hf), validates everything up
    front, then keeps up to --concurrency builds running server-side, starting
    the next build as soon as one finishes. Failed builds are written to a
    manifest you can re-run.

    Dockerfile builds read a context directory (manifest rows and Harbor
    environment/ folders) or a dataset column of Dockerfile contents
    (--dockerfile-column with --name-column). Public-registry source builds
    read manifest {"source": ...} rows, Harbor tasks with a prebuilt
    [environment] docker_image, or a dataset column of image references
    (--column). Allowed source registries: Docker Hub, ghcr.io, quay.io,
    public.ecr.aws, registry.k8s.io, and mcr.microsoft.com. Google-hosted
    registries are rejected.

    Docker Hub sources always become public, org-less platform images. Their
    namespace-preserving destination is derived automatically, with the
    implicit library/ namespace removed. They do not accept "image" overrides
    or --private, and configured team context is ignored for those items.
    Admins can pass --platform-image to apply org-less public platform
    ownership to explicit non-Docker-Hub registry sources too. Those explicit
    sources may use namespaced manifest "image" overrides ('ns/name:tag').

    \b
    Examples:
        prime images push-bulk --manifest builds.jsonl
        prime images push-bulk --harbor ./tasks --tag v1
        prime images push-bulk --harbor ./tasks --name-template "swe-{dir}" --dry-run
        prime images push-bulk --hf org/dataset --dockerfile-column dockerfile \
            --name-column instance_id
        prime images push-bulk --hf org/dataset --column docker_image
        prime images push-bulk --manifest sources.jsonl --platform-image
        prime images push-bulk --manifest push-bulk-failures.jsonl
    """
    hf_context_root: Optional[Path] = None
    keep_hf_context = False
    try:
        modes = [m for m in (manifest, harbor, hf_dataset) if m is not None]
        if len(modes) != 1:
            console.print("[red]Error: Provide exactly one of --manifest, --harbor or --hf[/red]")
            raise typer.Exit(1)
        if hf_dataset is None and (
            hf_split != "train" or hf_config or dockerfile_column or name_column or source_column
        ):
            console.print(
                "[red]Error: --hf-split, --hf-config, --dockerfile-column, --name-column "
                "and --column only apply to --hf mode[/red]"
            )
            raise typer.Exit(1)
        if hf_dataset is not None and bool(dockerfile_column or name_column) == bool(source_column):
            console.print(
                "[red]Error: --hf mode needs exactly one of --dockerfile-column "
                "(with --name-column) or --column[/red]"
            )
            raise typer.Exit(1)
        if public and private:
            console.print("[red]Error: --public and --private cannot be used together[/red]")
            raise typer.Exit(1)
        if platform_image and private:
            console.print("[red]Error: Platform images must be public[/red]")
            raise typer.Exit(1)
        if concurrency < 1:
            console.print("[red]Error: --concurrency must be at least 1[/red]")
            raise typer.Exit(1)
        if build_timeout < 1:
            console.print("[red]Error: --build-timeout must be at least 1[/red]")
            raise typer.Exit(1)
        if manifest is not None and (tag != "latest" or name_template != "{dir}"):
            console.print(
                "[red]Error: --tag and --name-template only apply to --harbor and "
                "--hf Dockerfile modes[/red]"
            )
            raise typer.Exit(1)
        if source_column is not None and (tag != "latest" or name_template != "{dir}"):
            console.print(
                "[red]Error: --tag and --name-template only apply to Dockerfile builds[/red]"
            )
            raise typer.Exit(1)

        skipped: list[tuple[str, str]] = []
        notes: list[str] = []
        build_specs: list[BuildSpec] = []
        source_specs: list[SourceBuildSpec] = []
        source_desc = ""
        try:
            if manifest is not None:
                manifest_path = manifest.resolve()
                if not manifest_path.is_file():
                    raise BulkPushValidationError([f"manifest not found: {manifest_path}"])
                source_desc = f"manifest {manifest_path}"
                build_specs, source_specs = load_manifest(
                    manifest_path, platform_image=platform_image
                )
            elif harbor is not None:
                harbor_root = harbor.resolve()
                if not harbor_root.is_dir():
                    raise BulkPushValidationError(
                        [f"Harbor task directory not found: {harbor_root}"]
                    )
                if not _TAG_RE.match(tag):
                    raise BulkPushValidationError([f"invalid image tag '{tag}'"])
                source_desc = f"Harbor tasks in {harbor_root}"
                build_specs, source_specs, skipped = load_harbor_specs(
                    harbor_root,
                    tag=tag,
                    name_template=name_template,
                    platform_image=platform_image,
                )
            else:
                assert hf_dataset is not None
                # Imported here: images_hf imports BuildSpec helpers from this module.
                from .images_hf import normalize_hf_dataset_id

                source_desc = f"Hugging Face dataset {normalize_hf_dataset_id(hf_dataset)}"
                if source_column is not None:
                    console.print(f"[cyan]Reading image references from {source_desc}...[/cyan]")
                    source_specs, notes = load_hf_source_specs(
                        hf_dataset,
                        config=hf_config,
                        split=hf_split,
                        column=source_column,
                        platform_image=platform_image,
                    )
                else:
                    if not _TAG_RE.match(tag):
                        raise BulkPushValidationError([f"invalid image tag '{tag}'"])
                    # Imported here: images_hf imports BuildSpec helpers from this module.
                    from .images_hf import load_hf_build_specs

                    console.print(f"[cyan]Reading Dockerfiles from {source_desc}...[/cyan]")
                    hf_context_root = Path(tempfile.mkdtemp(prefix="prime-push-bulk-hf-"))
                    build_specs, notes = load_hf_build_specs(
                        hf_dataset,
                        config=hf_config,
                        split=hf_split,
                        dockerfile_column=dockerfile_column,
                        name_column=name_column,
                        tag=tag,
                        name_template=name_template,
                        context_root=hf_context_root,
                    )
        except BulkPushValidationError as e:
            console.print(
                f"[red]Error: cannot start bulk push ({len(e.problems)} problem(s)):[/red]"
            )
            for problem in e.problems:
                console.print(f"[red]  - {problem}[/red]")
            raise typer.Exit(1)

        if platform_image and not source_specs:
            console.print(
                "[red]Error: --platform-image only applies to public-registry source builds[/red]"
            )
            raise typer.Exit(1)

        docker_hub_specs = [spec for spec in source_specs if spec.is_docker_hub]
        if docker_hub_specs and private:
            console.print("[red]Error: Docker Hub source builds must be public[/red]")
            raise typer.Exit(1)

        for note in notes:
            console.print(f"[dim]{note}[/dim]")
        for task_name, reason in skipped:
            console.print(f"[yellow]Skipping {task_name}: {reason}[/yellow]")

        if dry_run:
            table = Table(title=f"Resolved {len(build_specs) + len(source_specs)} build(s)")
            table.add_column("Image", style="cyan", no_wrap=True)
            table.add_column("Kind", no_wrap=True)
            table.add_column("Source", overflow="fold")
            table.add_column("From", style="dim")
            for spec in build_specs:
                table.add_row(
                    spec.image_ref,
                    "dockerfile",
                    str(spec.context),
                    spec.source,
                )
            for spec in source_specs:
                table.add_row(
                    spec.image_ref,
                    "source",
                    spec.source_image,
                    spec.source,
                )
            console.print(table)
            console.print("[dim]Dry run only — re-run without --dry-run to push.[/dim]")
            return

        config = Config()
        visibility: Optional[ImageVisibility] = None
        if public:
            visibility = ImageVisibility.PUBLIC
        elif private:
            visibility = ImageVisibility.PRIVATE

        console.print(
            f"[bold blue]Bulk pushing {len(build_specs) + len(source_specs)} image(s)[/bold blue] "
            f"[dim]({source_desc})[/dim]"
        )
        non_docker_hub_specs = [spec for spec in source_specs if not spec.is_docker_hub]
        if platform_image:
            console.print("[dim]Source builds owner: Platform[/dim]")
            if config.team_id:
                console.print("[dim]Team context ignored: platform images are org-less[/dim]")
        elif docker_hub_specs:
            console.print("[dim]Docker Hub sources: Platform owner, PUBLIC visibility[/dim]")
            if config.team_id and not non_docker_hub_specs:
                console.print("[dim]Team context ignored for Docker Hub sources[/dim]")
        if not platform_image and non_docker_hub_specs and config.team_id:
            console.print(f"[dim]Other sources team: {config.team_id}[/dim]")
        if platform_image:
            console.print(f"[dim]Source builds visibility: {ImageVisibility.PUBLIC.value}[/dim]")
        elif non_docker_hub_specs and visibility is not None:
            console.print(f"[dim]Other sources visibility: {visibility.value}[/dim]")
        elif non_docker_hub_specs:
            console.print(
                "[dim]Other sources visibility: PRIVATE for new images "
                "(existing tags keep their current visibility)[/dim]"
            )
        if build_specs:
            if config.team_id:
                console.print(f"[dim]Dockerfile builds team: {config.team_id}[/dim]")
            if visibility is not None:
                console.print(f"[dim]Dockerfile builds visibility: {visibility.value}[/dim]")
            else:
                console.print(
                    "[dim]Dockerfile builds visibility: PRIVATE for new images "
                    "(existing tags keep their current visibility)[/dim]"
                )
        # Only explicit admin non-Docker-Hub platform builds skip the
        # source-build rate limit.
        if platform_image and docker_hub_specs:
            pacing_note = (
                " Docker Hub auto-platform builds remain rate-limited per account "
                "server-side, so large batches take a while to submit."
            )
        elif not platform_image and source_specs:
            pacing_note = (
                " Source-image builds are rate-limited per account server-side, "
                "so large batches take a while to submit."
            )
        else:
            pacing_note = ""
        console.print(
            f"[dim]Up to {concurrency} builds in flight; "
            f"polling every {int(POLL_INTERVAL_SECONDS)}s.{pacing_note}[/dim]"
        )
        console.print()

        client = APIClient()
        outcomes = run_bulk_push(
            client,
            build_specs,
            source_specs,
            team_id=config.team_id or None,
            visibility=visibility,
            platform_image=platform_image,
            concurrency=concurrency,
            build_timeout=build_timeout,
        )

        failures = [o for o in outcomes if o.status != "COMPLETED"]
        completed_count = len(outcomes) - len(failures)
        console.print()
        console.print(f"[bold]{completed_count}/{len(outcomes)} builds completed[/bold]")
        if not failures:
            console.print("[bold green]All images pushed successfully![/bold green]")
            console.print()
            console.print("[dim]Use them with: prime sandbox create <image reference>[/dim]")
            return

        failure_table = Table(title=f"{len(failures)} build(s) did not complete")
        failure_table.add_column("Image", style="cyan", no_wrap=True)
        failure_table.add_column("Status", no_wrap=True)
        failure_table.add_column("From", style="dim")
        failure_table.add_column("Error")
        for outcome in failures[:FAILURE_TABLE_MAX_ROWS]:
            failure_table.add_row(
                outcome.spec.image_ref, outcome.status, outcome.spec.source, outcome.error or ""
            )
        console.print(failure_table)
        if len(failures) > FAILURE_TABLE_MAX_ROWS:
            console.print(
                f"[dim]... and {len(failures) - FAILURE_TABLE_MAX_ROWS} more, "
                f"all included in {failures_out}[/dim]"
            )
        failure_errors = [(o.error or "").lower() for o in failures]
        if any(marker in error for error in failure_errors for marker in _QUOTA_DETAIL_MARKERS):
            console.print(
                "[red]Image quota reached — delete unused images (prime images delete) "
                "or request a higher limit, then retry.[/red]"
            )
        if any(
            # Matches the engine's skip reason and its give-up SUBMIT_FAILED error.
            "kept rate-limiting submissions" in (o.error or "")
            or "rate-limit deferrals" in (o.error or "")
            for o in failures
        ):
            console.print(
                "[yellow]The server kept rate-limiting submissions — wait a few minutes, "
                "then retry with the failures manifest.[/yellow]"
            )

        _write_failures_manifest(failures_out, failures)
        console.print()
        console.print(f"Wrote {len(failures)} failed build(s) to [bold]{failures_out}[/bold]")
        console.print(f"[dim]Retry with: prime images push-bulk --manifest {failures_out}[/dim]")
        if hf_context_root is not None:
            # The failures manifest points into the generated build contexts.
            keep_hf_context = True
            console.print(
                f"[dim]Keeping generated Dockerfiles in {hf_context_root} "
                "so the failures manifest can be re-run.[/dim]"
            )
        raise typer.Exit(1)

    except UnauthorizedError:
        console.print("[red]Error: Not authenticated. Please run 'prime login' first.[/red]")
        raise typer.Exit(1)
    except BulkRunInterrupted as e:
        console.print(
            "\n[yellow]Cancelled. Builds already started keep running server-side; "
            "check them with 'prime images list'.[/yellow]"
        )
        unfinished = [o for o in e.outcomes if o.status != "COMPLETED"]
        if unfinished:
            _write_failures_manifest(failures_out, unfinished)
            console.print(
                f"Wrote {len(unfinished)} unfinished build(s) to [bold]{failures_out}[/bold]"
            )
            console.print(
                f"[dim]Resume with: prime images push-bulk --manifest {failures_out}[/dim]"
            )
            if hf_context_root is not None:
                keep_hf_context = True
                console.print(
                    f"[dim]Keeping generated Dockerfiles in {hf_context_root} "
                    "so the manifest can be re-run.[/dim]"
                )
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print(
            "\n[yellow]Cancelled. Builds already started keep running server-side; "
            "check them with 'prime images list'.[/yellow]"
        )
        raise typer.Exit(1)
    finally:
        if hf_context_root is not None and not keep_hf_context:
            shutil.rmtree(hf_context_root, ignore_errors=True)
