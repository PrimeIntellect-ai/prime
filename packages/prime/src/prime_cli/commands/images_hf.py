import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import httpx
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from ..utils import get_console
from .images_bulk import (
    BuildSpec,
    BulkPushValidationError,
    _duplicate_ref_problems,
    _render_name_template,
)

console = get_console()

HF_DATASETS_SERVER_URL = "https://datasets-server.huggingface.co"
HF_ROWS_PAGE_LENGTH = 100
LARGE_DATASET_WARN_BYTES = 500_000_000
HF_REQUEST_TIMEOUT_SECONDS = 30.0
HF_REQUEST_MAX_ATTEMPTS = 9
HF_REQUEST_BACKOFF_SECONDS = 2.0
HF_REQUEST_BACKOFF_MAX_SECONDS = 60.0

PARTIAL_DATASET_NOTE = (
    "Warning: the dataset viewer only covers part of this dataset; some rows may be missing"
)


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


def hf_get(path: str, params: dict[str, Any]) -> dict[str, Any]:
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
            console.print(
                f"[dim]Hugging Face returned HTTP {response.status_code}; "
                f"retrying in {int(retry_delay)}s...[/dim]"
            )
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


@dataclass
class HFDataset:
    """One selected (dataset, config) pair with its /info metadata."""

    dataset_id: str
    config: str
    config_info: dict[str, Any]
    partial: bool

    @property
    def features(self) -> dict[str, Any]:
        features = self.config_info.get("features")
        return features if isinstance(features, dict) else {}


def select_hf_config(dataset: str, config: Optional[str]) -> HFDataset:
    """Fetch /info for the dataset and resolve which config to read."""
    dataset_id = normalize_hf_dataset_id(dataset)
    info = hf_get("/info", {"dataset": dataset_id})
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
    return HFDataset(
        dataset_id=dataset_id,
        config=config,
        config_info=configs.get(config) or {},
        partial=bool(info.get("partial")),
    )


def missing_column_error(flag: str, ds: HFDataset) -> BulkPushValidationError:
    """Error for a required column flag that was not given, listing the columns."""
    return BulkPushValidationError(
        [
            f"{flag} is required with --hf (columns in "
            f"'{ds.dataset_id}': {', '.join(sorted(ds.features)) or 'unknown'})"
        ]
    )


def require_string_column(ds: HFDataset, column: str, *, reason: str) -> None:
    """Reject columns that are missing or whose feature type cannot hold the
    needed strings (e.g. actual pictures, nested lists) before submitting
    anything."""
    features = ds.features
    if features and column not in features:
        raise BulkPushValidationError(
            [
                f"column '{column}' not found in dataset '{ds.dataset_id}' "
                f"(columns: {', '.join(sorted(features))})"
            ]
        )
    feature_spec = features.get(column)
    if feature_spec is not None and not (
        isinstance(feature_spec, dict)
        and feature_spec.get("_type") == "Value"
        and feature_spec.get("dtype") in ("string", "large_string")
    ):
        raise BulkPushValidationError(
            [f"column '{column}' in '{ds.dataset_id}' is not a string column, so {reason}"]
        )


def check_split(ds: HFDataset, split: str) -> None:
    splits = ds.config_info.get("splits")
    if isinstance(splits, dict) and splits and split not in splits:
        raise BulkPushValidationError(
            [
                f"split '{split}' not found in dataset '{ds.dataset_id}' "
                f"(available: {', '.join(sorted(splits))})"
            ]
        )


def warn_if_large(ds: HFDataset) -> None:
    # The rows API returns full rows (a columns filter is not honored), so
    # reading time scales with total dataset size, not just row count.
    dataset_size = ds.config_info.get("dataset_size")
    if isinstance(dataset_size, (int, float)) and dataset_size > LARGE_DATASET_WARN_BYTES:
        console.print(
            f"[dim]Dataset rows total ~{dataset_size / 1e9:.1f}GB and the viewer API "
            "streams full rows, so reading may take a few minutes.[/dim]"
        )


def iter_hf_rows(ds: HFDataset, split: str) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield (row_idx, row) for every row of the split, with a progress bar."""
    offset = 0
    total: Optional[int] = None
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )
    with progress:
        task_id = progress.add_task("Reading dataset rows", total=None)
        while True:
            page = hf_get(
                "/rows",
                {
                    "dataset": ds.dataset_id,
                    "config": ds.config,
                    "split": split,
                    "offset": offset,
                    "length": HF_ROWS_PAGE_LENGTH,
                },
            )
            rows = page.get("rows") or []
            if isinstance(page.get("num_rows_total"), int):
                total = page["num_rows_total"]
                progress.update(task_id, total=total)
            for item in rows:
                yield item.get("row_idx", offset), item.get("row") or {}
            offset += len(rows)
            progress.update(task_id, completed=offset)
            if not rows or (total is not None and offset >= total):
                break


# ---------------------------------------------------------------------------
# Build-list resolution for push-bulk: Hugging Face datasets of Dockerfiles
# ---------------------------------------------------------------------------


def load_hf_build_specs(
    dataset: str,
    *,
    config: Optional[str],
    split: str,
    dockerfile_column: Optional[str],
    name_column: Optional[str],
    tag: str,
    name_template: str,
    platform: str,
    context_root: Path,
) -> tuple[list[BuildSpec], list[str]]:
    """Resolve build specs from a Hugging Face dataset of Dockerfiles.

    The push-bulk complement of transfer-bulk's --hf mode: instead of image
    references, each row carries Dockerfile *contents* (--dockerfile-column)
    plus a name for the resulting image (--name-column). Each usable row's
    Dockerfile is written under ``context_root`` as a one-file build context.
    Rows with identical (name, Dockerfile) values collapse into one build.
    Returns (specs, notes) where notes are informational messages.
    """
    ds = select_hf_config(dataset, config)
    notes: list[str] = []
    if dockerfile_column is None:
        raise missing_column_error("--dockerfile-column", ds)
    if name_column is None:
        raise missing_column_error("--name-column", ds)
    require_string_column(ds, dockerfile_column, reason="it cannot hold Dockerfile contents")
    require_string_column(ds, name_column, reason="it cannot hold image names")
    check_split(ds, split)
    if ds.partial:
        notes.append(PARTIAL_DATASET_NOTE)
    warn_if_large(ds)

    rows: list[tuple[int, str, str]] = []  # (row idx, image name value, Dockerfile)
    seen: set[tuple[str, str]] = set()
    scanned = 0
    incomplete_rows = 0
    for row_idx, row in iter_hf_rows(ds, split):
        name_value = row.get(name_column)
        dockerfile = row.get(dockerfile_column)
        if (
            not isinstance(name_value, str)
            or not name_value.strip()
            or not isinstance(dockerfile, str)
            or not dockerfile.strip()
        ):
            incomplete_rows += 1
            continue
        scanned += 1
        key = (name_value.strip(), dockerfile)
        if key in seen:
            continue
        seen.add(key)
        rows.append((row_idx, name_value.strip(), dockerfile))

    if incomplete_rows:
        notes.append(
            f"Skipped {incomplete_rows} row(s) with an empty "
            f"'{name_column}' or '{dockerfile_column}' value"
        )
    duplicates = scanned - len(rows)
    if duplicates:
        notes.append(f"Collapsed {duplicates} duplicate row(s)")

    problems: list[str] = []
    specs: list[BuildSpec] = []
    for row_idx, name_value, dockerfile in rows:
        # In HF mode both {dir} and {name} render as the row's name value.
        image_name = _render_name_template(
            name_template, task_dir_name=name_value, toml_name=name_value
        )
        if not image_name:
            problems.append(
                f"{ds.dataset_id} row {row_idx}: image name is empty "
                f"after sanitizing '{name_value}'"
            )
            continue
        context_dir = context_root / f"row-{row_idx}"
        context_dir.mkdir(parents=True, exist_ok=True)
        dockerfile_path = context_dir / "Dockerfile"
        dockerfile_path.write_text(dockerfile)
        specs.append(
            BuildSpec(
                image_name=image_name,
                image_tag=tag,
                context=context_dir,
                dockerfile=dockerfile_path,
                platform=platform,
                source=f"row {row_idx}",
            )
        )

    problems.extend(_duplicate_ref_problems(specs, hint=" — use --name-template to disambiguate"))
    if not specs and not problems:
        problems.append(
            f"no builds found in '{ds.dataset_id}' (no rows with both "
            f"'{name_column}' and '{dockerfile_column}')"
        )
    if problems:
        raise BulkPushValidationError(problems)
    return specs, notes
