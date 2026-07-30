"""High-level client for Prime Traces.

Wraps the wire client with upload batching/retry and typed read paths.

The read surface (list/get envelopes) is provisional: the service PR defines
routes but no response models yet, so the page shape here is a proposal to
align on, not a settled contract.

Deferred to follow-up PRs once the service pins the corresponding responses:
exports (streaming ``GET /traces/export`` and the job API, which is 501 in
v0), episode reads, ``/search``, the ``environment_id`` filter (no populated
column behind it yet), and the dot-path query compiler (needs the
server-side field registry).
"""

import random
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Union

from .batching import (
    DEFAULT_TARGET_BATCH_BYTES,
    Batch,
    iter_batches,
    read_jsonl_lines,
)
from .core.client import TracesAPIClient
from .exceptions import RetryableAPIError
from .models import (
    BatchReceipt,
    LineFormat,
    TraceListPage,
    TraceSummary,
)

DEFAULT_MAX_ATTEMPTS = 5
_BACKOFF_BASE_SECONDS = 1.0
_BACKOFF_CAP_SECONDS = 30.0


def _build_params(
    pairs: Iterable[tuple], context: Optional[Dict[str, str]] = None
) -> Dict[str, object]:
    """Drop unset filters and expand the context map to ``context.<key>``."""
    params: Dict[str, object] = {key: value for key, value in pairs if value is not None}
    if context:
        for key, value in context.items():
            params[f"context.{key}"] = value
    return params


class TracesClient:
    """Client for the Prime Traces API."""

    def __init__(self, api_client: Optional[TracesAPIClient] = None, **client_kwargs):
        self.client = api_client or TracesAPIClient(**client_kwargs)

    # -- upload -------------------------------------------------------------

    def upload_file(
        self,
        path: Union[str, Path],
        *,
        line_format: LineFormat = LineFormat.TRACE,
        context: Optional[Dict[str, str]] = None,
        schema_version: int = 1,
        compress: bool = True,
        target_batch_bytes: int = DEFAULT_TARGET_BATCH_BYTES,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        on_batch: Optional[Callable[[Batch, BatchReceipt], None]] = None,
    ) -> List[BatchReceipt]:
        """Upload a completed JSONL file of traces (or episodes).

        Batches are content-addressed, so rerunning after a crash is safe:
        chunks whose bytes are reproduced resolve to the same idempotency key
        and the service replays the committed receipt without re-storing.

        A durable rejection (400) stops the upload and raises
        ``ValidationRejectedError`` — correct the file and rerun; already
        committed chunks replay for free. 429/503 retry the same bytes,
        honoring Retry-After.

        Batches are sent sequentially in v0. The contract allows 2–8 requests
        in flight per producer; add bounded concurrency here once the service
        is up and throughput is measured.
        """
        return self.upload_lines(
            read_jsonl_lines(path),
            line_format=line_format,
            context=context,
            schema_version=schema_version,
            compress=compress,
            target_batch_bytes=target_batch_bytes,
            max_attempts=max_attempts,
            on_batch=on_batch,
        )

    def upload_lines(
        self,
        lines: Iterable[bytes],
        *,
        line_format: LineFormat = LineFormat.TRACE,
        context: Optional[Dict[str, str]] = None,
        schema_version: int = 1,
        compress: bool = True,
        target_batch_bytes: int = DEFAULT_TARGET_BATCH_BYTES,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        on_batch: Optional[Callable[[Batch, BatchReceipt], None]] = None,
    ) -> List[BatchReceipt]:
        """Upload an iterable of raw JSONL lines. See ``upload_file``."""
        receipts: List[BatchReceipt] = []
        for batch in iter_batches(lines, target_bytes=target_batch_bytes):
            result = self._send_with_retry(
                batch,
                line_format=line_format,
                context=context,
                schema_version=schema_version,
                compress=compress,
                max_attempts=max_attempts,
            )
            receipt = BatchReceipt.model_validate(result)
            receipts.append(receipt)
            if on_batch is not None:
                on_batch(batch, receipt)
        return receipts

    def _send_with_retry(
        self,
        batch: Batch,
        *,
        line_format: LineFormat,
        context: Optional[Dict[str, str]],
        schema_version: int,
        compress: bool,
        max_attempts: int,
    ) -> dict:
        last_error: Optional[Exception] = None
        for attempt in range(max_attempts):
            try:
                return self.client.upload_batch(
                    batch.data,
                    batch.idempotency_key,
                    line_format=line_format,
                    schema_version=schema_version,
                    context=context,
                    compress=compress,
                )
            except RetryableAPIError as exc:
                last_error = exc
                if attempt == max_attempts - 1:
                    break
                delay = exc.retry_after
                if delay is None:
                    delay = min(
                        _BACKOFF_CAP_SECONDS,
                        _BACKOFF_BASE_SECONDS * (2**attempt),
                    ) * (0.5 + random.random())
                time.sleep(delay)
        assert last_error is not None
        raise last_error

    # -- traces: read -------------------------------------------------------

    def list(
        self,
        *,
        run_id: Optional[str] = None,
        environment_id: Optional[str] = None,
        model_id: Optional[str] = None,
        model_provider: Optional[str] = None,
        task_id: Optional[str] = None,
        reward_min: Optional[float] = None,
        reward_max: Optional[float] = None,
        outcome: Optional[str] = None,
        has_error: Optional[bool] = None,
        is_truncated: Optional[bool] = None,
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
        context: Optional[Dict[str, str]] = None,
        sort: Optional[str] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> TraceListPage:
        """List trace summaries, newest first (max 100 per page).

        ``created_after``/``created_before`` also prune storage partitions, so
        they are the cheapest filters available. ``context`` filters are
        equality-only against the batch-supplied map.
        """
        params = _build_params(
            (
                ("run_id", run_id),
                ("environment_id", environment_id),
                ("model_id", model_id),
                ("model_provider", model_provider),
                ("task_id", task_id),
                ("reward_min", reward_min),
                ("reward_max", reward_max),
                ("outcome", outcome),
                ("has_error", has_error),
                ("is_truncated", is_truncated),
                ("created_after", created_after),
                ("created_before", created_before),
                ("sort", sort),
                ("limit", limit),
                ("cursor", cursor),
            ),
            context,
        )
        return TraceListPage.model_validate(self.client.get_json("/traces", params=params))

    def iter(self, **filters) -> Iterator[TraceSummary]:
        """Iterate all matching trace summaries across pages."""
        cursor = filters.pop("cursor", None)
        while True:
            page = self.list(cursor=cursor, **filters)
            yield from page.traces
            if not page.next_cursor:
                return
            cursor = page.next_cursor

    def get(self, trace_id: str) -> TraceSummary:
        """Get one trace summary."""
        return TraceSummary.model_validate(self.client.get_json(f"/traces/{trace_id}"))

    def get_raw(self, trace_id: str) -> bytes:
        """Get the stored raw trace document, buffered in memory.

        A trace can be tens of MiB; prefer ``download_raw`` for large traces.
        """
        return b"".join(self.client.stream_bytes(f"/traces/{trace_id}", params={"raw": "true"}))

    def download_raw(self, trace_id: str, dest: Union[str, Path]) -> int:
        """Stream the raw trace document to ``dest``. Returns bytes written."""
        written = 0
        with open(dest, "wb") as f:
            for chunk in self.client.stream_bytes(f"/traces/{trace_id}", params={"raw": "true"}):
                f.write(chunk)
                written += len(chunk)
        return written

    # -- traces: delete -----------------------------------------------------

    def delete(self, trace_id: str, *, created_at: Optional[str] = None) -> None:
        """Delete every stored copy of one trace (202 Accepted).

        ``created_at`` is an optional performance hint that lets the service
        prune on its ordering-key prefix; correctness does not depend on it.
        """
        params = {"created_at": created_at} if created_at else None
        self.client.delete_json(f"/traces/{trace_id}", params=params)

    def delete_run(self, run_id: str) -> Optional[str]:
        """Delete every trace in a run. Returns the async job id, if any."""
        result = self.client.delete_json("/traces", params={"run_id": run_id})
        job_id = result.get("job_id")
        return str(job_id) if job_id is not None else None

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "TracesClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
