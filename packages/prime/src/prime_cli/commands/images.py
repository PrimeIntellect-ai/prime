"""Commands for managing Docker images in Prime Intellect registry."""

import hashlib
import json
import re
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import click
import httpx
import typer
from gitignore_parser import parse_gitignore
from prime_sandboxes import APIClient, APIError, Config, UnauthorizedError
from rich.table import Table

from ..utils import (
    PlainTyper,
    get_console,
    json_output_help,
    output_data_as_json,
    validate_output_format,
)

app = PlainTyper(help="Manage Docker images in Prime Intellect registry", no_args_is_help=True)
console = get_console()
# Use a synthetic archive path to avoid collisions with Dockerfiles already in the context.
PACKAGED_DOCKERFILE_PATH = ".__prime_dockerfile__"
BATCH_BUILD_ENDPOINT = "/images/build-batches"
BATCH_BUILD_MANIFEST_VERSION = 1
HARBOR_COMPOSE_FILENAMES = (
    "docker-compose.yaml",
    "docker-compose.yml",
    "compose.yaml",
    "compose.yml",
)
HARBOR_CONTEXT_EXCLUDES = ("tests", "solution")
DEFAULT_BATCH_TAG_TEMPLATE = "{index}-{sha12}"

config = Config()


class ImageVisibility(str, Enum):
    PRIVATE = "PRIVATE"
    PUBLIC = "PUBLIC"


class BatchPushMode(str, Enum):
    DOCKERFILE = "dockerfile"
    HARBOR = "harbor"


LIST_IMAGES_JSON_HELP = json_output_help(
    "Raw API response is printed unchanged.",
    ".data[] = {displayRef?, fullImagePath?, imageName, imageTag, status, "
    "artifactType, ownerType, visibility, sizeBytes?, createdAt, pushedAt?}",
)

# ---------------------------------------------------------------------------
# Helpers for rendering `prime images list`
# ---------------------------------------------------------------------------

# Raw artifact row as returned by ``GET /v1/images``. The backend schema is
# documented in ``LIST_IMAGES_JSON_HELP`` above; we keep the dict shape loose
# here because the server may add new optional fields over time.
ImageRow = dict[str, Any]


@dataclass
class ArtifactPartition:
    """Per-artifact-type view of a grouped image.

    Holds the single most recently updated row for this artifact type. The
    status of that row is what gets rendered — we intentionally ignore older
    rows (including stale ``BUILDING`` / ``PENDING`` entries the backend may
    have orphaned) so the display reflects the current truth rather than a
    composite derived from history.
    """

    latest: Optional[ImageRow] = None

    def is_empty(self) -> bool:
        """True when no row exists for this artifact type."""
        return self.latest is None


# Mapping of artifact type (e.g. ``CONTAINER_IMAGE``) to its partition bucket.
PartitionMap = dict[str, ArtifactPartition]

# Timestamp priority used when picking the single latest row per artifact
# type. ``pushedAt`` wins for completed uploads; ``completedAt`` covers
# failed/cancelled terminal states; ``startedAt`` and ``createdAt`` are
# ultimate fallbacks for rows that never finished.
_LATEST_ROW_KEYS: tuple[str, ...] = ("pushedAt", "completedAt", "startedAt", "createdAt")


def _parse_ts(value: Any) -> Optional[datetime]:
    """Parse an ISO8601 timestamp (possibly ``Z`` suffixed) as a tz-aware UTC datetime.

    Naive timestamps (the backend emits ``createdAt`` as naive UTC, e.g. from
    ``datetime.utcnow()``) are treated as UTC so comparisons across the dataset
    are consistent. Returns ``None`` on failure.
    """
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _latest(rows: list[ImageRow], *keys: str) -> Optional[tuple[ImageRow, Optional[datetime]]]:
    """Return the row whose first parseable timestamp (across ``keys``) is newest.

    Each row is evaluated by walking ``keys`` in order and taking the first
    value that ``_parse_ts`` accepts. This means an unparseable-but-truthy
    value (e.g. a malformed date string) is skipped rather than short-circuiting
    selection — and the *same* parsed timestamp is returned alongside the row
    so callers don't re-read a potentially different field.

    The returned ``datetime`` is ``None`` only in the fallback case where no
    row had any parseable timestamp; we still return the first row so Size /
    Reference fields can be derived, but callers should treat "no timestamp"
    as a signal to fall through to a lower-priority tier.

    Returns ``None`` if ``rows`` is empty.
    """
    if not rows:
        return None
    best: Optional[ImageRow] = None
    best_ts: Optional[datetime] = None
    for row in rows:
        ts: Optional[datetime] = None
        for k in keys:
            ts = _parse_ts(row.get(k))
            if ts is not None:
                break
        if ts is None:
            continue
        if best_ts is None or ts > best_ts:
            best = row
            best_ts = ts
    if best is None:
        return rows[0], None
    return best, best_ts


def _coerce_artifact_type(value: Any) -> str:
    """Normalise an ``artifactType`` value into a usable string key.

    Defensive against a malformed backend payload (``null``, missing, or a
    non-string value) so that downstream ``sorted(partition)`` and label
    rendering never blow up on mixed key types.
    """
    if isinstance(value, str) and value:
        return value
    return "CONTAINER_IMAGE"


def _pick_row(rows: list[ImageRow], *keys: str) -> Optional[ImageRow]:
    """Return just the row component from ``_latest`` (drops the timestamp)."""
    result = _latest(rows, *keys)
    return result[0] if result is not None else None


def _partition_group(artifacts: list[ImageRow]) -> PartitionMap:
    """Group artifact rows by type, keeping only the most recent row per type.

    For a single ``imageName:imageTag`` the backend returns one row per
    (build, artifact type) plus any completed artifacts from the user images
    table. We pick the single newest row per artifact type (ordered by
    ``pushedAt → completedAt → startedAt → createdAt``) and render its raw
    ``status`` verbatim. Older rows — including stuck ``BUILDING`` zombies
    and stale failures — are simply not considered.
    """
    by_type: dict[str, list[ImageRow]] = {}
    for art in artifacts:
        t = _coerce_artifact_type(art.get("artifactType"))
        by_type.setdefault(t, []).append(art)

    result: PartitionMap = {}
    for art_type, rows in by_type.items():
        latest_row = _pick_row(rows, *_LATEST_ROW_KEYS)
        result[art_type] = ArtifactPartition(latest=latest_row)
    return result


_TYPE_LABELS: tuple[tuple[str, str], ...] = (
    ("CONTAINER_IMAGE", "[cyan]Container[/cyan]"),
    ("VM_SANDBOX", "[magenta]VM[/magenta]"),
)


def _ordered_present_types(partition: PartitionMap) -> list[tuple[str, str]]:
    """Return artifact types present in ``partition`` in display order.

    Container first, then VM, then any future types in sorted order. Types
    whose per-artifact partition bucket is completely empty (no completed,
    no active, no failed_only, no other) are skipped so we don't render
    dead slots.
    """
    ordered: list[tuple[str, str]] = []
    for art_type, label in _TYPE_LABELS:
        part = partition.get(art_type)
        if part is not None and not part.is_empty():
            ordered.append((art_type, label))
    for art_type in sorted(partition):
        if art_type in {"CONTAINER_IMAGE", "VM_SANDBOX"}:
            continue
        if not partition[art_type].is_empty():
            label = f"[white]{str(art_type).replace('_', ' ').title()}[/white]"
            ordered.append((art_type, label))
    return ordered


def _render_type_column(partition: PartitionMap) -> str:
    """Build the Type cell: ``Container / VM`` with color, only for types present."""
    parts = [label for _art_type, label in _ordered_present_types(partition)]
    return " / ".join(parts) if parts else "[dim]—[/dim]"


_STATUS_LABELS: dict[str, str] = {
    "COMPLETED": "[green]Ready[/green]",
    "BUILDING": "[yellow]Building[/yellow]",
    "UPLOADING": "[yellow]Uploading[/yellow]",
    "PENDING": "[blue]Pending[/blue]",
    "FAILED": "[red]Failed[/red]",
    "CANCELLED": "[dim]Cancelled[/dim]",
}


def _render_visibility(value: Any) -> str:
    try:
        visibility = ImageVisibility(str(value or ImageVisibility.PRIVATE.value).upper())
    except ValueError:
        visibility = ImageVisibility.PRIVATE

    if visibility == ImageVisibility.PUBLIC:
        return "[green]Public[/green]"
    return "[dim]Private[/dim]"


def _render_status_slot(part: Optional[ArtifactPartition]) -> str:
    """Render the raw status of the latest row for this artifact type.

    Unknown statuses (e.g. a future backend addition) are rendered as a
    dim title-cased label rather than being dropped.
    """
    if part is None or part.latest is None:
        return "[dim]—[/dim]"
    status = str(part.latest.get("status") or "UNKNOWN")
    return _STATUS_LABELS.get(status, f"[dim]{status.title()}[/dim]")


def _render_status_column(partition: PartitionMap) -> str:
    """Build the Status cell as positional slots aligned with the Type column.

    Example: if Type is ``Container / VM``, Status for ``rehl:latest`` with a
    container that's Ready and a VM that's Failed becomes ``Ready / Failed``.
    When an artifact has an active rebuild on top of a completed image, its
    slot is ``(rebuilding)``.
    """
    ordered = _ordered_present_types(partition)
    if not ordered:
        return "[dim]—[/dim]"
    return " / ".join(_render_status_slot(partition.get(art_type)) for art_type, _ in ordered)


def _render_image_reference(img: ImageRow, *, is_team_listing: bool) -> str:
    """Render the user-facing image reference.

    The owner prefix (``{userId}/`` or ``team-{teamId}/``) is **always**
    preserved so the full string is a valid reference that can be pasted
    directly into downstream commands (e.g. ``prime sandbox create``),
    which require the fully-qualified ``<userId>/<imageName>:<tag>`` form.

    Visual truncation (if any) is applied separately by
    :func:`_truncate_ref_left` so that when the terminal is too narrow the
    *owner prefix* is what gets clipped, not the image ``name:tag``.
    """
    del is_team_listing
    return (
        img.get("displayRef")
        or img.get("fullImagePath")
        or f"{img.get('imageName', 'unknown')}:{img.get('imageTag', 'latest')}"
    )


def _truncate_ref_left(ref: str, max_width: Optional[int]) -> str:
    """Left-ellipsize ``ref`` so the tail (``name:tag``) stays visible.

    If ``ref`` fits within ``max_width`` it's returned unchanged. Otherwise,
    the head of the string (the owner prefix) is dropped and replaced with a
    single ``…`` character. This matches the requested display style:

        cmkrcib4x00004kjyxq48…/nvidia-basic-dev:latest → …/nvidia-basic-dev:latest

    ``max_width`` must be at least 2; smaller values are clamped.
    """
    if max_width is None or max_width <= 0:
        return ref
    if len(ref) <= max_width:
        return ref
    max_width = max(max_width, 2)
    return "…" + ref[-(max_width - 1) :]


def _image_ref_column_width(console_width: int, is_team_listing: bool) -> int:
    """Compute a budget for the Image Reference column based on terminal width.

    The remaining columns have roughly fixed widths (worst-case labels):

        Type     ~14 chars ("Container / VM")
        Status   ~20 chars ("Uploading / Uploading")
        Visibility ~9 chars
        Size     ~10 chars
        Created  ~17 chars ("YYYY-MM-DD HH:MM ")
        Owner    ~10 chars (team listings only)
        borders + padding ~ 3 chars per column

    We subtract these from the terminal width and clamp to a reasonable range
    so the column is neither absurdly wide on large terminals nor unusable on
    tiny ones.
    """
    reserved = 14 + 20 + 9 + 10 + 17 + (10 if is_team_listing else 0)
    num_cols = 6 + (1 if is_team_listing else 0)
    reserved += 3 * num_cols
    budget = console_width - reserved
    return max(30, min(80, budget))


def _completed_size_mb(partition: PartitionMap) -> str:
    """Sum sizes of the latest COMPLETED rows per artifact type.

    Only completed artifacts carry a meaningful ``sizeBytes``; in-flight and
    failed rows contribute nothing, so summing across the latest-per-type
    gives the current on-disk footprint of the image.
    """
    total = 0
    for part in partition.values():
        row = part.latest
        if row is None or row.get("status") != "COMPLETED":
            continue
        total += row.get("sizeBytes") or 0
    if total <= 0:
        return "[dim]—[/dim]"
    return f"{total / 1024 / 1024:.1f} MB"


def _pick_display_datetime(partition: PartitionMap) -> Optional[datetime]:
    """Return the newest timestamp across the latest row of each artifact type."""
    latest_rows: list[ImageRow] = [
        part.latest for part in partition.values() if part.latest is not None
    ]
    if not latest_rows:
        return None
    result = _latest(latest_rows, *_LATEST_ROW_KEYS)
    if result is None:
        return None
    return result[1]


def _display_created(partition: PartitionMap) -> str:
    """Format the Created column value for a grouped image row."""
    ts = _pick_display_datetime(partition)
    return ts.strftime("%Y-%m-%d %H:%M") if ts is not None else ""


def _group_sort_key(partition: PartitionMap) -> datetime:
    """Key function to sort groups newest-first by their display timestamp."""
    ts = _pick_display_datetime(partition)
    return ts if ts is not None else datetime.min.replace(tzinfo=timezone.utc)


class BatchInputError(ValueError):
    """Raised when a batch input file or task tree cannot be converted to a manifest."""


@dataclass
class PackagedContext:
    path: Path
    size_bytes: int


@dataclass
class BatchBuildItem:
    source_id: str
    image_name: str
    image_tag: str
    source_type: str
    dockerfile_text: Optional[str] = None
    dockerfile_sha256: Optional[str] = None
    context_path: Optional[Path] = None
    dockerfile_path: Optional[Path] = None
    context_sha256: Optional[str] = None
    context_archive: Optional[PackagedContext] = None


_INVALID_TAG_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


def _validate_image_name(image_name: str) -> None:
    if not image_name:
        console.print("[red]Error: --image-name cannot be empty[/red]")
        raise typer.Exit(1)
    if "/" in image_name or ":" in image_name:
        console.print(
            "[red]Error: --image-name must be a simple image name without '/' or ':'.[/red]"
        )
        raise typer.Exit(1)


def _requested_visibility(public: bool, private: bool) -> Optional[ImageVisibility]:
    if public and private:
        console.print("[red]Error: --public and --private cannot be used together[/red]")
        raise typer.Exit(1)
    if public:
        return ImageVisibility.PUBLIC
    if private:
        return ImageVisibility.PRIVATE
    return None


def _resolve_build_paths(context: str, dockerfile: Optional[str]) -> tuple[Path, Path]:
    context_path = Path(context).resolve()
    dockerfile_path = Path(dockerfile).resolve() if dockerfile else context_path / "Dockerfile"

    if not context_path.exists():
        console.print(f"[red]Error: Build context not found at {context_path}[/red]")
        raise typer.Exit(1)

    if not context_path.is_dir():
        console.print(f"[red]Error: Build context must be a directory: {context_path}[/red]")
        raise typer.Exit(1)

    if not dockerfile_path.exists():
        console.print(f"[red]Error: Dockerfile not found at {dockerfile_path}[/red]")
        raise typer.Exit(1)

    if not dockerfile_path.is_file():
        console.print(f"[red]Error: Dockerfile must be a file: {dockerfile_path}[/red]")
        raise typer.Exit(1)

    return context_path, dockerfile_path


def _dockerignore_path(context_path: Path, dockerfile_path: Path) -> Optional[Path]:
    # BuildKit prefers <Dockerfile>.dockerignore next to the Dockerfile and
    # falls back to <context>/.dockerignore, so mirror that for uploaded tars.
    per_dockerfile_ignore = dockerfile_path.with_name(dockerfile_path.name + ".dockerignore")
    root_dockerignore = context_path / ".dockerignore"
    if per_dockerfile_ignore.is_file():
        return per_dockerfile_ignore
    if root_dockerignore.is_file():
        return root_dockerignore
    return None


def _context_rel_from_tar_name(name: str) -> Optional[Path]:
    rel = name[2:] if name.startswith("./") else name
    if not rel or rel == ".":
        return None
    return Path(rel)


def _is_excluded_context_rel(rel_path: Path, excluded_dirs: tuple[str, ...]) -> bool:
    return bool(rel_path.parts and rel_path.parts[0] in excluded_dirs)


def _make_context_tar_filter(
    context_path: Path,
    dockerfile_path: Path,
    *,
    excluded_dirs: tuple[str, ...] = (),
):
    dockerignore_path = _dockerignore_path(context_path, dockerfile_path)
    ignore_matcher = (
        parse_gitignore(str(dockerignore_path), base_dir=str(context_path))
        if dockerignore_path is not None
        else None
    )

    def tar_filter(tarinfo: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
        rel_path = _context_rel_from_tar_name(tarinfo.name)
        if rel_path is None:
            return tarinfo
        if _is_excluded_context_rel(rel_path, excluded_dirs):
            return None
        if ignore_matcher is not None and ignore_matcher(str(context_path / rel_path)):
            return None
        return tarinfo

    return tar_filter


def _package_build_context(
    context_path: Path,
    dockerfile_path: Path,
    *,
    excluded_dirs: tuple[str, ...] = (),
) -> PackagedContext:
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
        tar_path = Path(tmp_file.name)

    try:
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(
                context_path,
                arcname=".",
                filter=_make_context_tar_filter(
                    context_path,
                    dockerfile_path,
                    excluded_dirs=excluded_dirs,
                ),
            )
            tar.add(dockerfile_path, arcname=PACKAGED_DOCKERFILE_PATH)
    except Exception:
        try:
            tar_path.unlink()
        except Exception:
            pass
        raise

    return PackagedContext(path=tar_path, size_bytes=tar_path.stat().st_size)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_context(
    context_path: Path,
    dockerfile_path: Path,
    *,
    excluded_dirs: tuple[str, ...] = (),
) -> str:
    dockerignore_path = _dockerignore_path(context_path, dockerfile_path)
    ignore_matcher = (
        parse_gitignore(str(dockerignore_path), base_dir=str(context_path))
        if dockerignore_path is not None
        else None
    )
    digest = hashlib.sha256()
    for path in sorted(context_path.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(context_path)
        if _is_excluded_context_rel(rel_path, excluded_dirs):
            continue
        if ignore_matcher is not None and ignore_matcher(str(path)):
            continue
        digest.update(rel_path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _sanitize_image_tag(raw: str) -> str:
    tag = _INVALID_TAG_CHARS.sub("-", raw.strip())
    tag = re.sub(r"-{2,}", "-", tag).strip("-")
    if not tag:
        raise BatchInputError("Generated image tag is empty")
    if re.match(r"^[A-Za-z0-9_]", tag) is None:
        tag = f"t{tag}"
    return tag[:128]


def _format_batch_tag(
    tag_template: str,
    *,
    index: int,
    total: int,
    source_id: str,
    sha12: str,
) -> str:
    width = max(4, len(str(max(total, 1))))
    try:
        raw = tag_template.format(
            id=source_id,
            index=f"{index:0{width}d}",
            index0=f"{index - 1:0{width}d}",
            number=index,
            sha12=sha12,
        )
    except KeyError as exc:
        raise BatchInputError(f"Unknown tag template field: {exc}") from exc
    except ValueError as exc:
        raise BatchInputError(f"Invalid tag template: {exc}") from exc
    return _sanitize_image_tag(raw)


def _pick_jsonl_row_id(
    row: dict[str, Any],
    *,
    id_field: Optional[str],
    line_number: int,
) -> str:
    field_names = (id_field,) if id_field else ("id", "task_id")
    for field_name in field_names:
        if field_name and row.get(field_name) is not None:
            source_id = str(row[field_name]).strip()
            if source_id:
                return source_id
    expected = id_field or "id or task_id"
    raise BatchInputError(f"Line {line_number}: missing non-empty {expected} field")


def _read_jsonl_records(path: Path) -> list[tuple[int, dict[str, Any]]]:
    if not path.exists():
        raise BatchInputError(f"Input JSONL file not found: {path}")
    if not path.is_file():
        raise BatchInputError(f"Input path must be a JSONL file: {path}")

    records: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise BatchInputError(
                    f"Line {line_number}: invalid JSONL object: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise BatchInputError(f"Line {line_number}: expected a JSON object")
            records.append((line_number, value))

    if not records:
        raise BatchInputError(f"No JSONL rows found in {path}")
    return records


def _build_dockerfile_batch_items(
    source_path: Path,
    *,
    image_name: str,
    id_field: Optional[str],
    dockerfile_field: str,
    tag_template: str,
) -> list[BatchBuildItem]:
    raw_rows = _read_jsonl_records(source_path)
    prepared: list[tuple[str, str]] = []
    for line_number, row in raw_rows:
        source_id = _pick_jsonl_row_id(row, id_field=id_field, line_number=line_number)
        dockerfile_value = row.get(dockerfile_field)
        if not isinstance(dockerfile_value, str) or not dockerfile_value.strip():
            raise BatchInputError(f"Line {line_number}: missing non-empty {dockerfile_field} field")
        prepared.append((source_id, dockerfile_value))

    items: list[BatchBuildItem] = []
    total = len(prepared)
    for index, (source_id, dockerfile_text) in enumerate(prepared, start=1):
        dockerfile_sha256 = _sha256_text(dockerfile_text)
        image_tag = _format_batch_tag(
            tag_template,
            index=index,
            total=total,
            source_id=source_id,
            sha12=dockerfile_sha256[:12],
        )
        items.append(
            BatchBuildItem(
                source_id=source_id,
                image_name=image_name,
                image_tag=image_tag,
                source_type=BatchPushMode.DOCKERFILE.value,
                dockerfile_text=dockerfile_text,
                dockerfile_sha256=dockerfile_sha256,
            )
        )
    return items


def _find_harbor_compose_file(environment_path: Path) -> Optional[Path]:
    for filename in HARBOR_COMPOSE_FILENAMES:
        path = environment_path / filename
        if path.is_file():
            return path
    return None


def _build_harbor_batch_items(
    source_path: Path,
    *,
    image_name: str,
    tag_template: str,
    skip_unsupported_compose: bool,
) -> list[BatchBuildItem]:
    if not source_path.exists():
        raise BatchInputError(f"Harbor tasks directory not found: {source_path}")
    if not source_path.is_dir():
        raise BatchInputError(f"Harbor source must be a directory: {source_path}")

    task_dirs = sorted(
        path for path in source_path.iterdir() if path.is_dir() and not path.name.startswith(".")
    )
    if not task_dirs:
        raise BatchInputError(f"No task directories found in {source_path}")

    errors: list[str] = []
    prepared: list[tuple[str, Path, Path, str, str]] = []
    for task_dir in task_dirs:
        missing = [
            rel
            for rel in ("task.toml", "instruction.md", "environment/Dockerfile")
            if not (task_dir / rel).is_file()
        ]
        if missing:
            errors.append(f"{task_dir.name}: missing {', '.join(missing)}")
            continue

        environment_path = task_dir / "environment"
        compose_path = _find_harbor_compose_file(environment_path)
        if compose_path is not None:
            rel_compose = compose_path.relative_to(task_dir).as_posix()
            message = (
                f"{task_dir.name}: docker-compose tasks are not supported yet (found {rel_compose})"
            )
            if skip_unsupported_compose:
                console.print(f"[yellow]Skipping {message}[/yellow]")
                continue
            errors.append(message)
            continue

        dockerfile_path = environment_path / "Dockerfile"
        context_sha256 = _sha256_context(
            environment_path,
            dockerfile_path,
            excluded_dirs=HARBOR_CONTEXT_EXCLUDES,
        )
        prepared.append(
            (
                task_dir.name,
                environment_path,
                dockerfile_path,
                context_sha256,
                dockerfile_path.read_text(encoding="utf-8"),
            )
        )

    if errors:
        message = "\n  - ".join(errors)
        raise BatchInputError(f"Invalid Harbor task input:\n  - {message}")
    if not prepared:
        raise BatchInputError("No supported Harbor task environments found")

    items: list[BatchBuildItem] = []
    total = len(prepared)
    for index, (
        source_id,
        context_path,
        dockerfile_path,
        context_sha256,
        dockerfile_text,
    ) in enumerate(prepared, start=1):
        image_tag = _format_batch_tag(
            tag_template,
            index=index,
            total=total,
            source_id=source_id,
            sha12=context_sha256[:12],
        )
        items.append(
            BatchBuildItem(
                source_id=source_id,
                image_name=image_name,
                image_tag=image_tag,
                source_type=BatchPushMode.HARBOR.value,
                dockerfile_text=dockerfile_text,
                dockerfile_sha256=_sha256_text(dockerfile_text),
                context_path=context_path,
                dockerfile_path=dockerfile_path,
                context_sha256=context_sha256,
            )
        )
    return items


def _batch_manifest_record(item: BatchBuildItem) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": item.source_id,
        "image_name": item.image_name,
        "image_tag": item.image_tag,
        "source_type": item.source_type,
        "dockerfile_sha256": item.dockerfile_sha256,
    }
    if item.context_sha256 is not None:
        record["context_sha256"] = item.context_sha256
    if item.context_path is not None:
        record["context_path"] = str(item.context_path)
    if item.dockerfile_path is not None:
        record["dockerfile_path"] = str(item.dockerfile_path)
    return record


def _batch_payload_item(item: BatchBuildItem) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": item.source_id,
        "image_tag": item.image_tag,
        "source_type": item.source_type,
        "dockerfile_sha256": item.dockerfile_sha256,
    }
    if item.dockerfile_text is not None:
        payload["dockerfile"] = item.dockerfile_text
    if item.context_path is not None:
        payload["dockerfile_path"] = PACKAGED_DOCKERFILE_PATH
        payload["context_sha256"] = item.context_sha256
        payload["context_archive"] = {
            "content_type": "application/gzip",
            "size_bytes": (
                item.context_archive.size_bytes if item.context_archive is not None else None
            ),
        }
    else:
        payload["dockerfile_path"] = "Dockerfile"
    return payload


def _build_batch_payload(
    items: list[BatchBuildItem],
    *,
    mode: BatchPushMode,
    image_name: str,
    platform: str,
    visibility: Optional[ImageVisibility],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "manifest_version": BATCH_BUILD_MANIFEST_VERSION,
        "image_name": image_name,
        "mode": mode.value,
        "platform": platform,
        "items": [_batch_payload_item(item) for item in items],
    }
    if config.team_id:
        payload["team_id"] = config.team_id
    if visibility is not None:
        payload["visibility"] = visibility.value
    return payload


def _write_batch_manifest(path: Path, items: list[BatchBuildItem]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for item in items:
            file.write(json.dumps(_batch_manifest_record(item), sort_keys=True) + "\n")


def _batch_upload_urls(response: dict[str, Any]) -> dict[str, str]:
    urls: dict[str, str] = {}
    upload_urls = response.get("upload_urls")
    if isinstance(upload_urls, dict):
        for key, value in upload_urls.items():
            if isinstance(value, str):
                urls[str(key)] = value

    for key in ("items", "builds"):
        values = response.get(key)
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict):
                continue
            item_id = item.get("id") or item.get("source_id") or item.get("task_id")
            upload_url = item.get("upload_url")
            if item_id is not None and isinstance(upload_url, str):
                urls[str(item_id)] = upload_url
    return urls


def _batch_id(response: dict[str, Any]) -> Optional[str]:
    value = response.get("batch_id") or response.get("batchId") or response.get("id")
    return str(value) if value else None


def _upload_context_archive(upload_url: str, archive_path: Path) -> None:
    with archive_path.open("rb") as file:
        upload_response = httpx.put(
            upload_url,
            content=file,
            headers={"Content-Type": "application/octet-stream"},
            timeout=600.0,
        )
        upload_response.raise_for_status()


def _cleanup_batch_archives(items: list[BatchBuildItem]) -> None:
    for item in items:
        archive = item.context_archive
        if archive is None:
            continue
        try:
            archive.path.unlink()
        except Exception:
            pass


@app.command("push")
def push_image(
    image_reference: str = typer.Argument(
        ..., help="Image reference (e.g., 'myapp:v1.0.0' or 'myapp:latest')"
    ),
    context: str = typer.Option(".", "--context", "-c", help="Build context directory"),
    dockerfile: Optional[str] = typer.Option(
        None,
        "--dockerfile",
        "-f",
        help="Path to Dockerfile",
        show_default="<context>/Dockerfile",
    ),
    platform: str = typer.Option(
        "linux/amd64",
        "--platform",
        click_type=click.Choice(["linux/amd64", "linux/arm64"]),
        help="Target platform (defaults to linux/amd64 for Kubernetes compatibility)",
    ),
    public: bool = typer.Option(
        False,
        "--public",
        help="Make the image public when the build completes",
    ),
    private: bool = typer.Option(
        False,
        "--private",
        help="Make the image private when the build completes",
    ),
):
    """
    Build and push a Docker image to Prime Intellect registry.

    New image tags are private by default. Re-pushing an existing tag keeps
    its current visibility unless --public or --private is provided.

    \b
    Examples:
        prime images push myapp:v1.0.0
        prime images push myapp:latest --context ./app --dockerfile ../docker/Dockerfile.prod
        prime images push myapp:v1 --platform linux/arm64
        prime images push myapp:v1 --public
    """
    try:
        requested_visibility = _requested_visibility(public, private)

        # Parse image reference
        if ":" in image_reference:
            image_name, image_tag = image_reference.rsplit(":", 1)
        else:
            image_name = image_reference
            image_tag = "latest"

        # Validate image name doesn't contain slashes
        if "/" in image_name:
            console.print(
                "[red]Error: Image name cannot contain '/'. "
                "Use simple names like 'myapp:v1.0.0'.[/red]"
            )
            raise typer.Exit(1)

        console.print(
            f"[bold blue]Building and pushing image:[/bold blue] {image_name}:{image_tag}"
        )
        if config.team_id:
            console.print(f"[dim]Team: {config.team_id}[/dim]")
        console.print()

        # Initialize API client
        client = APIClient()

        context_path, dockerfile_path = _resolve_build_paths(context, dockerfile)

        # Create tar.gz of build context
        console.print("[cyan]Preparing build context...[/cyan]")
        packaged_context: Optional[PackagedContext] = None
        try:
            packaged_context = _package_build_context(context_path, dockerfile_path)

            tar_size_mb = packaged_context.size_bytes / (1024 * 1024)
            console.print(f"[green]✓[/green] Build context packaged ({tar_size_mb:.2f} MB)")
            console.print()

            # Initialize build
            console.print("[cyan]Initiating build...[/cyan]")
            try:
                build_payload = {
                    "image_name": image_name,
                    "image_tag": image_tag,
                    "dockerfile_path": PACKAGED_DOCKERFILE_PATH,
                    "platform": platform,
                }
                if config.team_id:
                    build_payload["team_id"] = config.team_id
                if requested_visibility is not None:
                    build_payload["visibility"] = requested_visibility.value

                build_response = client.request(
                    "POST",
                    "/images/build",
                    json=build_payload,
                )
            except UnauthorizedError:
                console.print(
                    "[red]Error: Not authenticated. Please run 'prime login' first.[/red]"
                )
                raise typer.Exit(1)
            except APIError as e:
                console.print(f"[red]Error: Failed to initiate build: {e}[/red]")
                raise typer.Exit(1)

            build_id = build_response.get("build_id")
            upload_url = build_response.get("upload_url")
            if not build_id or not upload_url:
                console.print(
                    "[red]Error: Invalid response from server "
                    "(missing build_id or upload_url)[/red]"
                )
                raise typer.Exit(1)
            full_image_path = build_response.get("fullImagePath") or f"{image_name}:{image_tag}"

            console.print("[green]✓[/green] Build initiated")
            console.print()

            # Upload build context to GCS
            console.print("[cyan]Uploading build context...[/cyan]")
            try:
                _upload_context_archive(upload_url, packaged_context.path)
            except httpx.HTTPError as e:
                console.print(f"[red]Upload failed: {e}[/red]")
                raise typer.Exit(1)

            console.print("[green]✓[/green] Build context uploaded")
            console.print()

            # Start the build
            console.print("[cyan]Starting build...[/cyan]")
            try:
                client.request(
                    "POST",
                    f"/images/build/{build_id}/start",
                    json={"context_uploaded": True},
                )
            except APIError as e:
                console.print(f"[red]Error: Failed to start build: {e}[/red]")
                raise typer.Exit(1)

            console.print("[green]✓[/green] Build started")
            console.print()

            console.print("[bold green]Build initiated successfully![/bold green]")
            console.print()
            console.print(f"[bold]Build ID:[/bold] {build_id}")
            console.print(f"[bold]Image:[/bold] {full_image_path}")
            if requested_visibility is not None:
                console.print(f"[bold]Visibility:[/bold] {requested_visibility.value}")
            else:
                console.print(
                    "[bold]Visibility:[/bold] PRIVATE for new images "
                    "(existing tags keep their current visibility)"
                )
            console.print()
            console.print("[cyan]Your image is being built.[/cyan]")
            console.print()
            console.print("[bold]Check build status:[/bold]")
            console.print("  prime images list")
            console.print()
            console.print(
                "[dim]The build typically takes a few minutes depending on image complexity.[/dim]"
            )
            console.print(
                "[dim]Once completed, you can use it with: "
                f"prime sandbox create {full_image_path}[/dim]"
            )
            console.print()

        finally:
            # Clean up temporary tar file
            if packaged_context is not None:
                try:
                    packaged_context.path.unlink()
                except Exception:
                    pass

    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        raise typer.Exit(1)


@app.command("push-batch")
def push_image_batch(
    source: str = typer.Argument(
        ...,
        help="JSONL file for dockerfile mode, or Harbor tasks directory for --mode harbor",
    ),
    image_name: str = typer.Option(
        ...,
        "--image-name",
        help="Image name to use for every generated tag (e.g. cligym)",
    ),
    mode: BatchPushMode = typer.Option(
        BatchPushMode.DOCKERFILE,
        "--mode",
        case_sensitive=False,
        help="Input mode: dockerfile JSONL (default) or harbor task directory scanner",
    ),
    id_field: Optional[str] = typer.Option(
        None,
        "--id-field",
        help="JSONL field to use as the original id. Defaults to id, then task_id.",
    ),
    dockerfile_field: str = typer.Option(
        "dockerfile",
        "--dockerfile-field",
        help="JSONL field containing raw Dockerfile text in dockerfile mode",
    ),
    tag_template: str = typer.Option(
        DEFAULT_BATCH_TAG_TEMPLATE,
        "--tag-template",
        help=(
            "Python format string for tags. Available fields: {index}, {index0}, "
            "{number}, {id}, {sha12}."
        ),
    ),
    platform: str = typer.Option(
        "linux/amd64",
        "--platform",
        click_type=click.Choice(["linux/amd64", "linux/arm64"]),
        help="Target platform for every image",
    ),
    public: bool = typer.Option(
        False,
        "--public",
        help="Make generated image tags public when builds complete",
    ),
    private: bool = typer.Option(
        False,
        "--private",
        help="Make generated image tags private when builds complete",
    ),
    manifest_output: Optional[str] = typer.Option(
        None,
        "--manifest-output",
        help="Write the generated client-side manifest as JSONL",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate inputs and print the manifest without calling the backend",
    ),
    skip_unsupported_compose: bool = typer.Option(
        False,
        "--skip-unsupported-compose",
        help="In Harbor mode, skip docker-compose tasks instead of failing the batch",
    ),
):
    """
    Build and push many Docker images in a single batch.

    The batch is created with POST /images/build-batches, followed by optional
    Harbor context uploads and POST /images/build-batches/{batch_id}/start. The
    platform fans each item out to its own build, capped by a per-batch
    concurrency limit, and bills each completed image as usual.

    Dockerfile mode is the default. It reads JSONL rows with id/task_id and
    dockerfile fields, where dockerfile is raw Dockerfile text.

    Harbor mode scans immediate task directories, requires task.toml,
    instruction.md, and environment/Dockerfile, then uses environment/ as the
    build context. docker-compose tasks are detected and rejected or skipped;
    compose support is intentionally deferred.

    \b
    Examples:
        prime images push-batch rows.jsonl --image-name cligym --public
        prime images push-batch rows.jsonl --image-name cligym --id-field task_id
        prime images push-batch ./tasks --mode harbor --image-name cligym --public
        prime images push-batch ./tasks --mode harbor --image-name cligym --dry-run
    """
    items: list[BatchBuildItem] = []
    try:
        visibility = _requested_visibility(public, private)
        _validate_image_name(image_name)
        source_path = Path(source).resolve()

        try:
            if mode == BatchPushMode.HARBOR:
                items = _build_harbor_batch_items(
                    source_path,
                    image_name=image_name,
                    tag_template=tag_template,
                    skip_unsupported_compose=skip_unsupported_compose,
                )
            else:
                items = _build_dockerfile_batch_items(
                    source_path,
                    image_name=image_name,
                    id_field=id_field,
                    dockerfile_field=dockerfile_field,
                    tag_template=tag_template,
                )
        except BatchInputError as exc:
            console.print(f"[red]Error: {exc}[/red]")
            raise typer.Exit(1) from exc

        if manifest_output:
            manifest_path = Path(manifest_output).resolve()
            _write_batch_manifest(manifest_path, items)
            console.print(f"[green]✓[/green] Wrote batch manifest: {manifest_path}")

        console.print(f"[bold blue]Prepared {len(items)} image build(s):[/bold blue] {image_name}")
        if config.team_id:
            console.print(f"[dim]Team: {config.team_id}[/dim]")
        if visibility is not None:
            console.print(f"[bold]Visibility:[/bold] {visibility.value}")
        else:
            console.print(
                "[bold]Visibility:[/bold] PRIVATE for new images "
                "(existing tags keep their current visibility)"
            )
        console.print(f"[bold]Mode:[/bold] {mode.value}")
        console.print()

        for item in items:
            console.print(f"  {item.source_id} -> {item.image_name}:{item.image_tag}")
        console.print()

        if dry_run:
            console.print("[yellow]Dry run: no backend request was sent.[/yellow]")
            console.print(f"[dim]Real runs submit to: POST {BATCH_BUILD_ENDPOINT}[/dim]")
            console.print()
            console.print("[bold]Generated manifest:[/bold]")
            for item in items:
                console.print(json.dumps(_batch_manifest_record(item), sort_keys=True))
            return

        context_items = [item for item in items if item.context_path is not None]
        if context_items:
            console.print("[cyan]Preparing Harbor build contexts...[/cyan]")
            for item in context_items:
                if item.context_path is None or item.dockerfile_path is None:
                    console.print(
                        f"[red]Error: Missing context metadata for {item.source_id}[/red]"
                    )
                    raise typer.Exit(1)
                item.context_archive = _package_build_context(
                    item.context_path,
                    item.dockerfile_path,
                    excluded_dirs=HARBOR_CONTEXT_EXCLUDES,
                )
            total_size = sum(item.context_archive.size_bytes for item in context_items)
            console.print(
                f"[green]✓[/green] Packaged {len(context_items)} context(s) "
                f"({total_size / (1024 * 1024):.2f} MB)"
            )
            console.print()

        client = APIClient()

        console.print(f"[cyan]Creating image build batch via {BATCH_BUILD_ENDPOINT}...[/cyan]")
        try:
            batch_response = client.request(
                "POST",
                BATCH_BUILD_ENDPOINT,
                json=_build_batch_payload(
                    items,
                    mode=mode,
                    image_name=image_name,
                    platform=platform,
                    visibility=visibility,
                ),
            )
        except UnauthorizedError:
            console.print("[red]Error: Not authenticated. Please run 'prime login' first.[/red]")
            raise typer.Exit(1)
        except APIError as exc:
            console.print(f"[red]Error: Failed to create image build batch: {exc}[/red]")
            raise typer.Exit(1) from exc

        batch_id = _batch_id(batch_response)
        if batch_id is None:
            console.print("[red]Error: Invalid response from server (missing batch_id)[/red]")
            raise typer.Exit(1)

        if context_items:
            upload_urls = _batch_upload_urls(batch_response)
            console.print("[cyan]Uploading Harbor build contexts...[/cyan]")
            for item in context_items:
                archive = item.context_archive
                upload_url = upload_urls.get(item.source_id)
                if archive is None or not upload_url:
                    console.print(
                        "[red]Error: Invalid response from server "
                        f"(missing upload_url for {item.source_id})[/red]"
                    )
                    raise typer.Exit(1)
                try:
                    _upload_context_archive(upload_url, archive.path)
                except httpx.HTTPError as exc:
                    console.print(f"[red]Upload failed for {item.source_id}: {exc}[/red]")
                    raise typer.Exit(1) from exc
            console.print("[green]✓[/green] Build contexts uploaded")
            console.print()

        console.print("[cyan]Starting image build batch...[/cyan]")
        try:
            client.request(
                "POST",
                f"{BATCH_BUILD_ENDPOINT}/{batch_id}/start",
                json={"contexts_uploaded": bool(context_items)},
            )
        except APIError as exc:
            console.print(f"[red]Error: Failed to start image build batch: {exc}[/red]")
            raise typer.Exit(1) from exc

        console.print("[green]✓[/green] Batch build started")
        console.print()
        console.print("[bold green]Image build batch initiated successfully![/bold green]")
        console.print(f"[bold]Batch ID:[/bold] {batch_id}")
        console.print(f"[bold]Images:[/bold] {len(items)}")
        console.print("[bold]Check build status:[/bold]")
        console.print("  prime images list")
        console.print()

    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        raise typer.Exit(1)
    finally:
        _cleanup_batch_archives(items)


@app.command("list", epilog=LIST_IMAGES_JSON_HELP)
def list_images(
    output: str = typer.Option("table", "--output", "-o", help="Output format (table or json)"),
    search: Optional[str] = typer.Option(
        None,
        "--search",
        "-q",
        help="Case-insensitive substring match on image name, tag, or reference",
    ),
    all_images: bool = typer.Option(
        False, "--all", "-a", help="[Deprecated] Show all accessible images (personal + team)"
    ),
    page: int = typer.Option(1, "--page", "-p", help="Page number"),
    num: int = typer.Option(50, "--num", "-n", help="Items per page (max 250)"),
):
    """
    List all images you've pushed to Prime Intellect registry.

    Shows personal images by default, or team images when a team is configured.

    The --search filter is applied server-side across all your images (not just
    the current page), and pagination reflects the filtered results.

    \b
    Examples:
        prime images list
        prime images list --search myapp
        prime images list -q nvidia
        prime images list --num 100
        prime images list --page 2
        prime images list --output json
    """
    validate_output_format(output, console)

    if num < 1 or page < 1:
        console.print("[red]Error:[/red] --num and --page must be at least 1")
        raise typer.Exit(1)
    if num > 250:
        console.print("[red]Error:[/red] --num cannot exceed 250")
        raise typer.Exit(1)

    if all_images and output != "json":
        console.print(
            "[yellow]Warning: --all flag is deprecated and will be removed in a future release. "
            "Images are now scoped to your current context (personal or team).[/yellow]"
        )
        console.print()
    try:
        client = APIClient()

        offset = (page - 1) * num

        # Build query params
        params: dict[str, str] = {"limit": str(num), "offset": str(offset)}
        if config.team_id:
            params["teamId"] = config.team_id
        if search:
            params["search"] = search

        response = client.request("GET", "/images", params=params)
        images: list[ImageRow] = response.get("data", [])
        has_total_count: bool = "totalCount" in response
        total_count: int = int(response.get("totalCount", offset + len(images)))

        if output == "json":
            output_data_as_json(response, console)
            return

        if not images:
            if has_total_count and total_count == 0:
                if search:
                    console.print(f"[yellow]No images match '{search}'.[/yellow]")
                    console.print(
                        "Try a different search term or run without [bold]--search[/bold]."
                    )
                else:
                    console.print("[yellow]No images or builds found.[/yellow]")
                    console.print("Push an image with: [bold]prime images push <name>:<tag>[/bold]")
            elif has_total_count:
                console.print(
                    f"[yellow]No images on page {page}. Total: {total_count} image(s).[/yellow]"
                )
                console.print("Try [bold]--page 1[/bold] to start from the beginning.")
            elif page > 1:
                console.print(f"[yellow]No images on page {page}.[/yellow]")
                console.print("Try [bold]--page 1[/bold] to start from the beginning.")
            elif search:
                console.print(f"[yellow]No images match '{search}'.[/yellow]")
                console.print("Try a different search term or run without [bold]--search[/bold].")
            else:
                console.print("[yellow]No images or builds found.[/yellow]")
                console.print("Push an image with: [bold]prime images push <name>:<tag>[/bold]")
            return

        # Table output
        is_team_listing: bool = bool(config.team_id)
        title: str
        if is_team_listing:
            title = f"Team Docker Images (team: {config.team_id})"
        else:
            title = "Personal Docker Images"

        grouped: dict[str, list[ImageRow]] = {}
        for img in images:
            owner_scope = img.get("teamId") or img.get("ownerType", "personal")
            key = f"{owner_scope}/{img.get('imageName', '')}:{img.get('imageTag', 'latest')}"
            grouped.setdefault(key, []).append(img)

        ref_max_width: int = _image_ref_column_width(
            console.size.width, is_team_listing=is_team_listing
        )

        table = Table(title=title)
        table.add_column(
            "Image Reference",
            style="cyan",
            no_wrap=True,
            overflow="crop",
            min_width=ref_max_width,
            max_width=ref_max_width,
        )
        table.add_column("Type", justify="center", no_wrap=True)
        if is_team_listing:
            table.add_column("Owner", justify="center")
        # Worst-case label is ``Cancelled / Cancelled`` (21 chars); pin to 21
        # so Rich never wraps the status text across two lines.
        table.add_column("Status", justify="center", no_wrap=True, min_width=21)
        table.add_column("Visibility", justify="center", no_wrap=True)
        table.add_column("Size", justify="right", no_wrap=True)
        table.add_column("Created", style="dim", no_wrap=True, min_width=16)

        sortable: list[tuple[datetime, list[ImageRow], PartitionMap]] = []
        for _key, artifacts in grouped.items():
            partition = _partition_group(artifacts)
            sortable.append((_group_sort_key(partition), artifacts, partition))
        sortable.sort(key=lambda item: item[0], reverse=True)

        for _ts, artifacts, partition in sortable:
            # Pick a representative row for the Image Reference / Owner
            # columns. Prefer the latest container artifact row so the
            # reference always resolves to something the user can copy-paste,
            # then fall back to the latest VM row, then to any raw artifact.
            container_latest = (partition.get("CONTAINER_IMAGE") or ArtifactPartition()).latest
            vm_latest = (partition.get("VM_SANDBOX") or ArtifactPartition()).latest
            preferred: ImageRow = container_latest or vm_latest or next(iter(artifacts), {})

            image_ref: str = _truncate_ref_left(
                _render_image_reference(preferred, is_team_listing=is_team_listing),
                ref_max_width,
            )
            type_display: str = _render_type_column(partition)
            status_display: str = _render_status_column(partition)
            visibility_display: str = _render_visibility(preferred.get("visibility"))
            size_mb: str = _completed_size_mb(partition)
            date_str: str = _display_created(partition)

            row: list[str] = [image_ref, type_display]
            if is_team_listing:
                owner_type = preferred.get(
                    "ownerType", "team" if preferred.get("teamId") else "personal"
                )
                owner_display: str = (
                    "[blue]Team[/blue]" if owner_type == "team" else "[dim]Personal[/dim]"
                )
                row.append(owner_display)
            row.extend([status_display, visibility_display, size_mb, date_str])
            table.add_row(*row)

        console.print()
        console.print(table)
        console.print()
        shown_groups = len(grouped)
        if has_total_count:
            has_next = offset + shown_groups < total_count
        else:
            has_next = shown_groups >= num
        if has_next or page > 1:
            start = offset + 1
            end = offset + shown_groups
            if has_total_count:
                console.print(
                    f"[dim]Page {page} • showing {start}-{end} of {total_count} image(s)[/dim]"
                )
            else:
                console.print(f"[dim]Page {page} • showing {start}-{end}[/dim]")
            if has_next:
                console.print(f"[dim]Use --page {page + 1} to see more.[/dim]")
        elif has_total_count:
            console.print(f"[dim]Total: {total_count} image(s)[/dim]")
        else:
            console.print(f"[dim]Total: {shown_groups} image(s)[/dim]")
        console.print()

    except UnauthorizedError:
        console.print("[red]Error: Not authenticated. Please run 'prime login' first.[/red]")
        raise typer.Exit(1)
    except APIError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


def _parse_mutable_image_reference(image_reference: str) -> tuple[str, str, Optional[str]]:
    """Parse refs accepted by mutating image commands.

    Returns ``(image_name, image_tag, team_id)``. Personal image refs may use
    either ``name:tag`` or ``{currentUserId}/name:tag``. Team image refs may
    use ``team-{teamId}/name:tag``.
    """
    team_id: Optional[str] = config.team_id
    if "/" in image_reference:
        namespace, rest = image_reference.split("/", 1)
        if namespace.startswith("team-"):
            extracted_team_id = namespace[5:]
            if not extracted_team_id:
                console.print(
                    "[red]Error: Invalid team image reference. "
                    "Expected format: team-{teamId}/imagename:tag[/red]"
                )
                raise typer.Exit(1)
            team_id = extracted_team_id
            image_reference = rest
        elif namespace == config.user_id:
            team_id = None
            image_reference = rest
        else:
            console.print(
                f"[red]Error: Unrecognized image namespace '{namespace}'. "
                "Use 'imagename:tag' for personal images, "
                "'{userId}/imagename:tag' with your current user ID, or "
                "'team-{teamId}/imagename:tag' for team images.[/red]"
            )
            raise typer.Exit(1)

    if ":" not in image_reference:
        console.print("[red]Error: Image reference must include a tag (e.g., myapp:latest)[/red]")
        raise typer.Exit(1)

    image_name, image_tag = image_reference.rsplit(":", 1)
    return image_name, image_tag, team_id


def _set_image_visibility(image_reference: str, visibility: ImageVisibility) -> None:
    image_name, image_tag, team_id = _parse_mutable_image_reference(image_reference)
    payload: dict[str, str] = {"visibility": visibility.value}
    if team_id:
        payload["teamId"] = team_id

    client = APIClient()
    client.request(
        "PATCH",
        f"/images/{image_name}/{image_tag}/visibility",
        json=payload,
    )
    context = f" (team: {team_id})" if team_id else ""
    console.print(
        f"[green]✓[/green] Updated {image_name}:{image_tag}{context} to {visibility.value}"
    )


@app.command("publish")
def publish_image(
    image_reference: str = typer.Argument(
        ...,
        help=(
            "Image reference to make public "
            "(e.g., 'myapp:v1.0.0', '<currentUserId>/myapp:v1.0.0', "
            "or 'team-{teamId}/myapp:v1.0.0')"
        ),
    ),
):
    """
    Make an image public so other Prime users can run it.

    \b
    Examples:
        prime images publish myapp:v1.0.0
        prime images publish cmk123/myapp:v1.0.0
        prime images publish team-abc123/myapp:v1.0.0
    """
    try:
        _set_image_visibility(image_reference, ImageVisibility.PUBLIC)
    except UnauthorizedError:
        console.print("[red]Error: Not authenticated. Please run 'prime login' first.[/red]")
        raise typer.Exit(1)
    except APIError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("unpublish")
def unpublish_image(
    image_reference: str = typer.Argument(
        ...,
        help=(
            "Image reference to make private "
            "(e.g., 'myapp:v1.0.0', '<currentUserId>/myapp:v1.0.0', "
            "or 'team-{teamId}/myapp:v1.0.0')"
        ),
    ),
):
    """
    Make a public image private again.

    \b
    Examples:
        prime images unpublish myapp:v1.0.0
        prime images unpublish cmk123/myapp:v1.0.0
        prime images unpublish team-abc123/myapp:v1.0.0
    """
    try:
        _set_image_visibility(image_reference, ImageVisibility.PRIVATE)
    except UnauthorizedError:
        console.print("[red]Error: Not authenticated. Please run 'prime login' first.[/red]")
        raise typer.Exit(1)
    except APIError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("delete")
def delete_image(
    image_reference: str = typer.Argument(
        ...,
        help=(
            "Image reference to delete "
            "(e.g., 'myapp:v1.0.0', '<currentUserId>/myapp:v1.0.0', "
            "or 'team-{teamId}/myapp:v1.0.0')"
        ),
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """
    Delete an image from your registry.

    For team images, you can use the team-prefixed format directly.
    Only the image creator or team admins can delete team images.

    \b
    Examples:
        prime images delete myapp:v1.0.0
        prime images delete myapp:latest --yes
        prime images delete cmk123/myapp:v1.0.0
        prime images delete team-abc123/myapp:v1.0.0
    """
    # Store original input for error messages
    original_reference = image_reference

    try:
        image_name, image_tag, team_id = _parse_mutable_image_reference(image_reference)

        context = f" (team: {team_id})" if team_id else ""
        if not yes:
            msg = f"Are you sure you want to delete {image_name}:{image_tag}{context}?"
            confirm = typer.confirm(msg)
            if not confirm:
                console.print("[yellow]Cancelled[/yellow]")
                raise typer.Exit(0)

        client = APIClient()

        params = {"teamId": team_id} if team_id else None

        client.request("DELETE", f"/images/{image_name}/{image_tag}", params=params)
        console.print(f"[green]✓[/green] Deleted {image_name}:{image_tag}{context}")

    except UnauthorizedError:
        console.print("[red]Error: Not authenticated. Please run 'prime login' first.[/red]")
        raise typer.Exit(1)
    except APIError as e:
        if "404" in str(e):
            console.print(f"[red]Error: Image {original_reference} not found[/red]")
        elif "403" in str(e):
            console.print(
                "[red]Error: You don't have permission to delete this image. "
                "Only the image creator or team admins can delete team images.[/red]"
            )
        else:
            console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
