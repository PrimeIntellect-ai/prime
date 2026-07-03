import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import click
import httpx
import typer
from prime_sandboxes import (
    APIClient,
    APIError,
    Config,
    ImageVisibility,
    UnauthorizedError,
)
from rich.table import Table

from ..utils import get_console
from .images_bulk import (
    _QUOTA_DETAIL_MARKERS,
    _TAG_RE,
    DEFAULT_BUILD_TIMEOUT_SECONDS,
    DEFAULT_MAX_IN_FLIGHT,
    FAILURE_TABLE_MAX_ROWS,
    POLL_INTERVAL_SECONDS,
    RATE_LIMIT_SKIP_REASON,
    SUPPORTED_PLATFORMS,
    BulkPushValidationError,
    QuotaExceededError,
    SubmitRateLimited,
    _duplicate_ref_problems,
    _is_http_429,
    _is_quota_429,
    _write_failures_manifest,
    discover_harbor_tasks,
    run_bulk_jobs,
)

console = get_console()

# How long to pause new submissions after the server's transfer rate limiter
# rejects one.
TRANSFER_RATE_LIMIT_PAUSE_SECONDS = 15.0

HF_DATASETS_SERVER_URL = "https://datasets-server.huggingface.co"
HF_ROWS_PAGE_LENGTH = 100
HF_IMAGE_COLUMN_CANDIDATES = ("docker_image", "image_name", "container_image", "image")
HF_REQUEST_TIMEOUT_SECONDS = 30.0
HF_REQUEST_MAX_ATTEMPTS = 6
HF_REQUEST_BACKOFF_SECONDS = 2.0
HF_REQUEST_BACKOFF_MAX_SECONDS = 60.0

_TRANSFER_MANIFEST_KEYS = {"source", "image", "platform"}

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass
class TransferSpec:
    """One resolved transfer: a source registry image and its Prime destination.

    ``dest_name``/``dest_tag`` are what the transfer will be called. When
    ``override`` is False they are derived from the source the same way the
    server derives them and are only used for display and duplicate detection;
    when True the user chose them and they are sent with the request.
    """

    source_image: str
    dest_name: str
    dest_tag: str
    platform: str
    source: str
    override: bool = False

    @property
    def image_ref(self) -> str:
        return f"{self.dest_name}:{self.dest_tag}"

    def to_manifest_line(self) -> dict[str, str]:
        line = {"source": self.source_image, "platform": self.platform}
        if self.override:
            line["image"] = self.image_ref
        return line


def derive_transfer_destination(source_ref: str) -> tuple[str, str]:
    """Derive the (name, tag) a transfer gets when no destination is given.

    Mirrors the server's parsing (last repository segment lowercased; tag kept,
    or ``sha256-<digest prefix>`` for digest refs). Used client-side for dry
    runs and duplicate-destination detection; the server stays authoritative.
    """
    ref = (source_ref or "").strip()
    if not ref:
        raise ValueError("empty image reference")
    # A comma is never valid inside an image reference. Reject it here rather
    # than letting the server split it into multiple transfers, which this
    # command tracks one-per-spec.
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

    name = repository.split("/")[-1].lower()
    if tag is None:
        assert digest is not None
        tag = "sha256-" + digest.split(":", 1)[1][:16]
    return name, tag


_DUPLICATE_DEST_HINT = (
    ' — transfer them via a JSONL manifest with distinct "image" destination overrides'
)


# ---------------------------------------------------------------------------
# Transfer-list resolution: JSONL manifest
# ---------------------------------------------------------------------------


def load_transfer_manifest(manifest_path: Path, default_platform: str) -> list[TransferSpec]:
    """Parse and fully validate a JSONL transfer manifest."""
    problems: list[str] = []
    specs: list[TransferSpec] = []

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
        unknown = sorted(set(entry) - _TRANSFER_MANIFEST_KEYS)
        if unknown:
            problems.append(
                f"{where}: unknown key(s) {', '.join(unknown)} "
                f"(expected: {', '.join(sorted(_TRANSFER_MANIFEST_KEYS))})"
            )
            continue

        source = entry.get("source")
        if not source or not isinstance(source, str):
            problems.append(f"{where}: 'source' is required")
            continue
        source = source.strip()

        platform = entry.get("platform") or default_platform
        if platform not in SUPPORTED_PLATFORMS:
            problems.append(
                f"{where}: unsupported platform '{platform}' "
                f"(supported: {', '.join(SUPPORTED_PLATFORMS)})"
            )
            continue

        try:
            derived_name, derived_tag = derive_transfer_destination(source)
        except ValueError as e:
            problems.append(f"{where}: invalid source '{source}' ({e})")
            continue

        image = entry.get("image")
        if image is not None and not isinstance(image, str):
            problems.append(f"{where}: 'image' must be a string")
            continue
        if image:
            if ":" in image:
                dest_name, dest_tag = image.rsplit(":", 1)
            else:
                dest_name, dest_tag = image, "latest"
            if not dest_name or "/" in dest_name:
                problems.append(
                    f"{where}: invalid destination '{image}'; use simple names like 'myapp:v1'"
                )
                continue
            if not _TAG_RE.match(dest_tag):
                problems.append(f"{where}: invalid destination tag '{dest_tag}'")
                continue
            override = True
        else:
            dest_name, dest_tag = derived_name, derived_tag
            override = False

        specs.append(
            TransferSpec(
                source_image=source,
                dest_name=dest_name,
                dest_tag=dest_tag,
                platform=platform,
                source=where,
                override=override,
            )
        )

    problems.extend(_duplicate_ref_problems(specs, hint=_DUPLICATE_DEST_HINT))
    if not specs and not problems:
        problems.append(f"{manifest_path.name}: manifest contains no transfers")
    if problems:
        raise BulkPushValidationError(problems)
    return specs


# ---------------------------------------------------------------------------
# Transfer-list resolution: Harbor task directories
# ---------------------------------------------------------------------------


def load_harbor_transfer_specs(
    root: Path, *, platform: str
) -> tuple[list[TransferSpec], list[tuple[str, str]]]:
    """Resolve transfer specs from a Harbor tasks directory.

    The complement of push-bulk's Harbor mode: it collects the prebuilt
    ``[environment] docker_image`` references that push-bulk skips. Returns
    (specs, skipped) where skipped lists tasks with nothing to transfer.
    Tasks sharing the same prebuilt image collapse into one transfer.
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
    specs: list[TransferSpec] = []
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
        if not docker_image or not isinstance(docker_image, str):
            skipped.append(
                (task_dir.name, "no prebuilt docker_image (build it with push-bulk instead)")
            )
            continue
        docker_image = docker_image.strip()

        first_task = first_task_by_source.get(docker_image)
        if first_task is not None:
            skipped.append((task_dir.name, f"same image as {first_task} ({docker_image})"))
            continue
        first_task_by_source[docker_image] = task_dir.name

        try:
            dest_name, dest_tag = derive_transfer_destination(docker_image)
        except ValueError as e:
            problems.append(f"{task_dir.name}: invalid docker_image '{docker_image}' ({e})")
            continue

        specs.append(
            TransferSpec(
                source_image=docker_image,
                dest_name=dest_name,
                dest_tag=dest_tag,
                platform=platform,
                source=task_dir.name,
                override=False,
            )
        )

    problems.extend(_duplicate_ref_problems(specs, hint=_DUPLICATE_DEST_HINT))
    if not specs and not problems:
        problems.append(
            f"no tasks with a prebuilt docker_image under {root} "
            f"({len(skipped)} skipped: {', '.join(name for name, _ in skipped)})"
        )
    if problems:
        raise BulkPushValidationError(problems)
    return specs, skipped


# ---------------------------------------------------------------------------
# Transfer-list resolution: Hugging Face datasets
# ---------------------------------------------------------------------------


def normalize_hf_dataset_id(raw: str) -> str:
    """Accept 'org/name', hf://datasets/... and huggingface.co dataset URLs."""
    dataset = (raw or "").strip()
    for prefix in (
        "hf://datasets/",
        "https://huggingface.co/datasets/",
        "http://huggingface.co/datasets/",
        "huggingface.co/datasets/",
    ):
        if dataset.startswith(prefix):
            dataset = dataset[len(prefix) :]
            dataset = dataset.split("?", 1)[0].split("#", 1)[0]
            # Drop trailing URL parts like /viewer/default/train.
            parts = [part for part in dataset.split("/") if part]
            dataset = "/".join(parts[:2])
            break
    dataset = dataset.strip("/")
    if not dataset or dataset.count("/") > 1:
        raise BulkPushValidationError(
            [f"invalid Hugging Face dataset '{raw}' (expected 'org/name' or a dataset URL)"]
        )
    return dataset


def _hf_get(path: str, params: dict[str, Any]) -> dict[str, Any]:
    """GET from the HF datasets-server with retries on transient failures.

    Uses HF_TOKEN from the environment for gated/private datasets. The
    datasets-server rate limit is per IP and clears within about a minute, so
    429/5xx retries back off long enough to ride out a full window, honoring
    Retry-After when the server sends one.
    """
    token = os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = f"{HF_DATASETS_SERVER_URL}{path}"
    delay = HF_REQUEST_BACKOFF_SECONDS
    for attempt in range(1, HF_REQUEST_MAX_ATTEMPTS + 1):
        retry_delay = delay
        delay = min(delay * 2, HF_REQUEST_BACKOFF_MAX_SECONDS)
        try:
            response = httpx.get(
                url, params=params, headers=headers, timeout=HF_REQUEST_TIMEOUT_SECONDS
            )
        except httpx.HTTPError as e:
            if attempt == HF_REQUEST_MAX_ATTEMPTS:
                raise BulkPushValidationError([f"Hugging Face request failed: {e}"])
            time.sleep(retry_delay)
            continue
        if response.status_code == 429 or response.status_code >= 500:
            if attempt == HF_REQUEST_MAX_ATTEMPTS:
                rate_limit_hint = (
                    " (rate limited — wait a minute and re-run)"
                    if response.status_code == 429
                    else ""
                )
                raise BulkPushValidationError(
                    [
                        f"Hugging Face request failed with HTTP "
                        f"{response.status_code}: {url}{rate_limit_hint}"
                    ]
                )
            retry_after = response.headers.get("retry-after")
            if retry_after:
                try:
                    retry_delay = max(retry_delay, float(retry_after))
                except ValueError:
                    pass
            time.sleep(retry_delay)
            continue
        if response.status_code != 200:
            try:
                detail = response.json().get("error") or response.text
            except ValueError:
                detail = response.text
            raise BulkPushValidationError(
                [
                    f"Hugging Face request failed with HTTP {response.status_code} "
                    f"({detail}). The dataset may be private/gated (set HF_TOKEN) or have "
                    "the dataset viewer disabled — as a fallback, extract the image "
                    "references yourself and use --manifest"
                ]
            )
        return response.json()
    raise AssertionError("unreachable")


def load_hf_specs(
    dataset: str,
    *,
    config: Optional[str],
    split: str,
    column: Optional[str],
    platform: str,
) -> tuple[list[TransferSpec], list[str]]:
    """Resolve transfer specs from a Hugging Face dataset column.

    Pages the dataset through the datasets-server rows API (no local
    `datasets` dependency), dedupes identical references preserving order,
    and returns (specs, notes) where notes are informational messages.
    """
    dataset_id = normalize_hf_dataset_id(dataset)
    notes: list[str] = []

    info = _hf_get("/info", {"dataset": dataset_id})
    configs = info.get("dataset_info")
    if not isinstance(configs, dict) or not configs:
        raise BulkPushValidationError([f"no dataset info available for '{dataset_id}'"])
    if config is None:
        if len(configs) > 1:
            raise BulkPushValidationError(
                [
                    f"dataset '{dataset_id}' has multiple configs "
                    f"({', '.join(sorted(configs))}); choose one with --hf-config"
                ]
            )
        config = next(iter(configs))
    elif config not in configs:
        raise BulkPushValidationError(
            [
                f"config '{config}' not found in dataset '{dataset_id}' "
                f"(available: {', '.join(sorted(configs))})"
            ]
        )
    config_info = configs.get(config) or {}

    features = config_info.get("features") or {}
    if column is None:
        column = next((c for c in HF_IMAGE_COLUMN_CANDIDATES if c in features), None)
        if column is None:
            raise BulkPushValidationError(
                [
                    f"could not auto-detect an image column in '{dataset_id}' "
                    f"(looked for: {', '.join(HF_IMAGE_COLUMN_CANDIDATES)}; "
                    f"dataset columns: {', '.join(sorted(features)) or 'unknown'}); "
                    "pass --column"
                ]
            )
        notes.append(f"Using column '{column}' (auto-detected)")
    elif features and column not in features:
        raise BulkPushValidationError(
            [
                f"column '{column}' not found in dataset '{dataset_id}' "
                f"(columns: {', '.join(sorted(features))})"
            ]
        )

    splits = config_info.get("splits")
    if isinstance(splits, dict) and splits and split not in splits:
        raise BulkPushValidationError(
            [
                f"split '{split}' not found in dataset '{dataset_id}' "
                f"(available: {', '.join(sorted(splits))})"
            ]
        )
    if info.get("partial"):
        notes.append(
            "Warning: the dataset viewer only covers part of this dataset; some rows may be missing"
        )

    sources: list[tuple[str, int]] = []  # (image ref, first row index), order-preserving
    seen: set[str] = set()
    scanned = 0
    empty_rows = 0
    offset = 0
    total: Optional[int] = None
    while True:
        page = _hf_get(
            "/rows",
            {
                "dataset": dataset_id,
                "config": config,
                "split": split,
                "offset": offset,
                "length": HF_ROWS_PAGE_LENGTH,
            },
        )
        rows = page.get("rows") or []
        if isinstance(page.get("num_rows_total"), int):
            total = page["num_rows_total"]
        for item in rows:
            row = item.get("row") or {}
            row_idx = item.get("row_idx", offset)
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
        offset += len(rows)
        if not rows or (total is not None and offset >= total):
            break

    if empty_rows:
        notes.append(f"Skipped {empty_rows} row(s) with an empty '{column}' value")
    duplicates = scanned - len(sources)
    if duplicates:
        notes.append(f"Collapsed {duplicates} duplicate image reference(s)")

    problems: list[str] = []
    specs: list[TransferSpec] = []
    for ref, row_idx in sources:
        try:
            dest_name, dest_tag = derive_transfer_destination(ref)
        except ValueError as e:
            problems.append(f"{dataset_id} row {row_idx}: invalid image reference '{ref}' ({e})")
            continue
        specs.append(
            TransferSpec(
                source_image=ref,
                dest_name=dest_name,
                dest_tag=dest_tag,
                platform=platform,
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


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------


def _submit_transfer(
    client: APIClient,
    spec: TransferSpec,
    *,
    team_id: Optional[str],
    visibility: Optional[ImageVisibility],
) -> tuple[str, str]:
    """Queue one transfer. Returns (build_id, full_image_path).

    Transfers are counted per image by the server's transfer rate limiter
    (a rolling window), so a plain 429 means "later", not "failed": it is
    surfaced as SubmitRateLimited so the engine requeues the spec. Wallet
    quota 429s become QuotaExceededError and stop the run.
    """
    payload: dict[str, Any] = {
        "source_image": spec.source_image,
        "platform": spec.platform,
    }
    if spec.override:
        payload["image_name"] = spec.dest_name
        payload["image_tag"] = spec.dest_tag
    if team_id:
        payload["team_id"] = team_id
    if visibility is not None:
        payload["visibility"] = visibility.value

    try:
        response = client.request("POST", "/images/build", json=payload)
    except UnauthorizedError:
        raise
    except APIError as e:
        if _is_quota_429(e):
            raise QuotaExceededError(str(e)) from e
        if _is_http_429(e):
            raise SubmitRateLimited(str(e), retry_after=TRANSFER_RATE_LIMIT_PAUSE_SECONDS) from e
        raise

    # Single-source transfers return a top-level build_id today; the endpoint
    # also has a per-source results shape (used for comma-separated sources),
    # so unwrap it like `prime images push --source-image` does in case the
    # contract shifts.
    results = response.get("results")
    if isinstance(results, list):
        # Specs are validated to hold exactly one image reference, so anything
        # other than one entry means we would silently drop transfers.
        if len(results) != 1 or not isinstance(results[0], dict):
            raise APIError(
                f"invalid response from server (expected one transfer result, got {len(results)})"
            )
        entry = results[0]
        if not entry.get("success") or not entry.get("buildId"):
            error = entry.get("error") or "invalid response from server (transfer not queued)"
            if any(marker in error.lower() for marker in _QUOTA_DETAIL_MARKERS):
                raise QuotaExceededError(error)
            raise APIError(error)
        return entry["buildId"], entry.get("fullImagePath") or spec.image_ref

    build_id = response.get("build_id") or response.get("buildId")
    if not build_id:
        raise APIError("invalid response from server (missing build_id)")
    return build_id, response.get("fullImagePath") or spec.image_ref


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


def transfer_bulk(
    manifest: Optional[Path] = typer.Option(
        None,
        "--manifest",
        "-m",
        help=(
            "JSONL manifest of transfers; each line is "
            '{"source": "registry/repo:tag", "image"?: "name:tag", "platform"?: "..."}'
        ),
    ),
    harbor: Optional[Path] = typer.Option(
        None,
        "--harbor",
        help=(
            "Harbor task directory (or directory of tasks); transfers each task's "
            "prebuilt [environment] docker_image"
        ),
    ),
    hf_dataset: Optional[str] = typer.Option(
        None,
        "--hf",
        help=(
            "Hugging Face dataset id or URL (e.g. 'R2E-Gym/R2E-Gym-Subset'); "
            "transfers every image referenced in the dataset"
        ),
    ),
    hf_split: str = typer.Option("train", "--hf-split", help="Dataset split for --hf mode"),
    hf_config: Optional[str] = typer.Option(
        None,
        "--hf-config",
        help="Dataset config for --hf mode",
        show_default="the dataset's only config",
    ),
    column: Optional[str] = typer.Option(
        None,
        "--column",
        help="Dataset column holding image references for --hf mode",
        show_default=f"auto-detect: {', '.join(HF_IMAGE_COLUMN_CANDIDATES)}",
    ),
    platform: str = typer.Option(
        "linux/amd64",
        "--platform",
        click_type=click.Choice(list(SUPPORTED_PLATFORMS)),
        help="Required platform for the transferred images (manifest lines may override)",
    ),
    public: bool = typer.Option(
        False, "--public", help="Make the images public when the transfers complete"
    ),
    private: bool = typer.Option(
        False, "--private", help="Make the images private when the transfers complete"
    ),
    concurrency: int = typer.Option(
        DEFAULT_MAX_IN_FLIGHT, "--concurrency", help="Maximum transfers in flight at once"
    ),
    build_timeout: int = typer.Option(
        DEFAULT_BUILD_TIMEOUT_SECONDS,
        "--build-timeout",
        help="Seconds to wait for a single transfer before giving up on it",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Resolve and print the transfer list without transferring"
    ),
    failures_out: Path = typer.Option(
        Path("transfer-bulk-failures.jsonl"),
        "--failures-out",
        help="Where to write a re-runnable manifest of failed transfers",
    ),
):
    """
    Copy many existing registry images into Prime in one command.

    Reads image references from a JSONL manifest (--manifest), a Harbor tasks
    directory (--harbor), or a Hugging Face dataset (--hf), validates
    everything up front, then queues transfers server-side, pacing submissions
    to the server's transfer rate limit and polling until every transfer
    finishes. Failed transfers are written to a manifest you can re-run.

    \b
    Manifest format (one JSON object per line; "image" overrides the
    destination, which otherwise derives from the source reference):
        {"source": "ghcr.io/org/app:v1"}
        {"source": "docker.io/other/app:v1", "image": "other-app:v1"}

    \b
    Harbor mode transfers the prebuilt docker_image of each task — exactly the
    tasks push-bulk skips, so the two commands together cover a task set.

    \b
    Hugging Face mode reads image references straight from a dataset column
    (auto-detected, e.g. docker_image) using the dataset viewer API — no local
    dataset download. Set HF_TOKEN for private or gated datasets.

    \b
    Examples:
        prime images transfer-bulk --manifest transfers.jsonl
        prime images transfer-bulk --harbor ./tasks
        prime images transfer-bulk --hf R2E-Gym/R2E-Gym-Subset --dry-run
        prime images transfer-bulk --hf org/dataset --hf-split test --column image_name
        prime images transfer-bulk --manifest transfer-bulk-failures.jsonl
    """
    try:
        modes = [m for m in (manifest, harbor, hf_dataset) if m is not None]
        if len(modes) != 1:
            console.print("[red]Error: Provide exactly one of --manifest, --harbor or --hf[/red]")
            raise typer.Exit(1)
        if hf_dataset is None and (hf_split != "train" or hf_config or column):
            console.print(
                "[red]Error: --hf-split, --hf-config and --column only apply to --hf mode[/red]"
            )
            raise typer.Exit(1)
        if public and private:
            console.print("[red]Error: --public and --private cannot be used together[/red]")
            raise typer.Exit(1)
        if concurrency < 1:
            console.print("[red]Error: --concurrency must be at least 1[/red]")
            raise typer.Exit(1)
        if build_timeout < 1:
            console.print("[red]Error: --build-timeout must be at least 1[/red]")
            raise typer.Exit(1)

        skipped: list[tuple[str, str]] = []
        notes: list[str] = []
        try:
            if manifest is not None:
                manifest_path = manifest.resolve()
                if not manifest_path.is_file():
                    raise BulkPushValidationError([f"manifest not found: {manifest_path}"])
                source_desc = f"manifest {manifest_path}"
                specs = load_transfer_manifest(manifest_path, default_platform=platform)
            elif harbor is not None:
                harbor_root = harbor.resolve()
                if not harbor_root.is_dir():
                    raise BulkPushValidationError(
                        [f"Harbor task directory not found: {harbor_root}"]
                    )
                source_desc = f"Harbor tasks in {harbor_root}"
                specs, skipped = load_harbor_transfer_specs(harbor_root, platform=platform)
            else:
                assert hf_dataset is not None
                source_desc = f"Hugging Face dataset {normalize_hf_dataset_id(hf_dataset)}"
                console.print(f"[cyan]Reading image references from {source_desc}...[/cyan]")
                specs, notes = load_hf_specs(
                    hf_dataset,
                    config=hf_config,
                    split=hf_split,
                    column=column,
                    platform=platform,
                )
        except BulkPushValidationError as e:
            console.print(
                f"[red]Error: cannot start bulk transfer ({len(e.problems)} problem(s)):[/red]"
            )
            for problem in e.problems:
                console.print(f"[red]  - {problem}[/red]")
            raise typer.Exit(1)

        for note in notes:
            console.print(f"[dim]{note}[/dim]")
        for task_name, reason in skipped:
            console.print(f"[yellow]Skipping {task_name}: {reason}[/yellow]")

        if dry_run:
            table = Table(title=f"Resolved {len(specs)} transfer(s)")
            table.add_column("Source", style="cyan", overflow="fold")
            table.add_column("Destination", overflow="fold")
            table.add_column("Platform", no_wrap=True)
            table.add_column("From", style="dim")
            for spec in specs:
                table.add_row(spec.source_image, spec.image_ref, spec.platform, spec.source)
            console.print(table)
            console.print("[dim]Dry run only — re-run without --dry-run to transfer.[/dim]")
            return

        config = Config()
        visibility: Optional[ImageVisibility] = None
        if public:
            visibility = ImageVisibility.PUBLIC
        elif private:
            visibility = ImageVisibility.PRIVATE

        console.print(
            f"[bold blue]Bulk transferring {len(specs)} image(s)[/bold blue] "
            f"[dim]({source_desc})[/dim]"
        )
        if config.team_id:
            console.print(f"[dim]Team: {config.team_id}[/dim]")
        if visibility is not None:
            console.print(f"[dim]Visibility: {visibility.value}[/dim]")
        else:
            console.print(
                "[dim]Visibility: PRIVATE for new images "
                "(existing tags keep their current visibility)[/dim]"
            )
        console.print(
            f"[dim]Up to {concurrency} transfers in flight; "
            f"polling every {int(POLL_INTERVAL_SECONDS)}s. Transfers are rate-limited "
            "per account server-side, so large batches take a while to submit.[/dim]"
        )
        console.print()

        client = APIClient()
        outcomes = run_bulk_jobs(
            client,
            specs,
            submit=lambda spec: _submit_transfer(
                client, spec, team_id=config.team_id or None, visibility=visibility
            ),
            concurrency=concurrency,
            build_timeout=build_timeout,
            progress_description="Transferring images",
        )

        failures = [o for o in outcomes if o.status != "COMPLETED"]
        completed_count = len(outcomes) - len(failures)
        console.print()
        console.print(f"[bold]{completed_count}/{len(outcomes)} transfers completed[/bold]")
        if not failures:
            console.print("[bold green]All images transferred successfully![/bold green]")
            console.print()
            console.print("[dim]Use them with: prime sandbox create <image reference>[/dim]")
            return

        failure_table = Table(title=f"{len(failures)} transfer(s) did not complete")
        failure_table.add_column("Source", style="cyan", overflow="fold")
        failure_table.add_column("Destination", overflow="fold")
        failure_table.add_column("Status", no_wrap=True)
        failure_table.add_column("From", style="dim")
        failure_table.add_column("Error")
        for outcome in failures[:FAILURE_TABLE_MAX_ROWS]:
            failure_table.add_row(
                outcome.spec.source_image,
                outcome.spec.image_ref,
                outcome.status,
                outcome.spec.source,
                outcome.error or "",
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
            (o.error or "") == RATE_LIMIT_SKIP_REASON or "rate-limit deferrals" in (o.error or "")
            for o in failures
        ):
            console.print(
                "[yellow]The server kept rate-limiting submissions — wait a few minutes, "
                "then retry with the failures manifest.[/yellow]"
            )

        _write_failures_manifest(failures_out, failures)
        console.print()
        console.print(f"Wrote {len(failures)} failed transfer(s) to [bold]{failures_out}[/bold]")
        console.print(
            f"[dim]Retry with: prime images transfer-bulk --manifest {failures_out}[/dim]"
        )
        raise typer.Exit(1)

    except UnauthorizedError:
        console.print("[red]Error: Not authenticated. Please run 'prime login' first.[/red]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print(
            "\n[yellow]Cancelled. Transfers already queued keep running server-side; "
            "check them with 'prime images list'.[/yellow]"
        )
        raise typer.Exit(1)
