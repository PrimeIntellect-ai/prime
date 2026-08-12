"""High-level client for Prime Traces.

Wraps the wire client with upload batching/retry and typed read paths.

The read surface matches the service's pinned response models
(``prime-traces/src/traces/models.py`` in the platform repo): pages are
``{items, next_cursor}``, summaries nest ``model``/``score``/``execution``.

Deferred to follow-up PRs: exports (streaming ``GET /traces/export`` — its
filter vocabulary is not declared server-side yet — and the unimplemented job
API), episode reads, ``/search``, the ``environment_id`` filter (no populated
column behind it yet), and the dot-path query compiler (needs the
server-side field registry).
"""

import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Union
from urllib.parse import quote

from .batching import (
    DEFAULT_TARGET_BATCH_BYTES,
    Batch,
    iter_batches,
    read_jsonl_lines,
)
from .core.client import TracesAPIClient, retry_delay
from .exceptions import RetryableAPIError, TransportError
from .models import (
    LineFormat,
    TraceListPage,
    TraceSummary,
    UploadReceipt,
)

DEFAULT_MAX_ATTEMPTS = 5


def _build_params(
    pairs: Iterable[tuple], context: Optional[Dict[str, str]] = None
) -> Dict[str, object]:
    """Drop unset filters and expand the context map to ``context.<key>``."""
    params: Dict[str, object] = {key: value for key, value in pairs if value is not None}
    if context:
        for key, value in context.items():
            params[f"context.{key}"] = value
    return params


def _trace_endpoint(trace_id: str) -> str:
    """Build a trace endpoint with the ID encoded as one path segment."""
    return f"/traces/{quote(trace_id, safe='')}"


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
        on_batch: Optional[Callable[[Batch, UploadReceipt], None]] = None,
    ) -> List[UploadReceipt]:
        """Upload a completed JSONL file of traces (or episodes).

        Batches are content-addressed, so rerunning after a crash is safe:
        chunks whose bytes are reproduced resolve to the same idempotency key
        and the service replays the committed receipt without re-storing.

        A durable rejection (400) stops the upload and raises
        ``ValidationRejectedError`` — correct the file and rerun; already
        committed chunks replay for free. 429/503 retry the same bytes,
        honoring Retry-After. Transport failures (connection drops, timeouts,
        resets) also retry the same bytes: content addressing makes even an
        ambiguous failure safe, because a request that did land replays its
        receipt instead of storing twice.

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
        on_batch: Optional[Callable[[Batch, UploadReceipt], None]] = None,
    ) -> List[UploadReceipt]:
        """Upload an iterable of raw JSONL lines. See ``upload_file``."""
        receipts: List[UploadReceipt] = []
        for batch in iter_batches(lines, target_bytes=target_batch_bytes):
            result = self._send_with_retry(
                batch,
                line_format=line_format,
                context=context,
                schema_version=schema_version,
                compress=compress,
                max_attempts=max_attempts,
            )
            receipt = UploadReceipt.model_validate(result)
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
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
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
            except (RetryableAPIError, TransportError) as exc:
                last_error = exc
                if attempt == max_attempts - 1:
                    break
                time.sleep(retry_delay(exc, attempt))
        assert last_error is not None
        raise last_error

    # -- traces: read -------------------------------------------------------

    def list(
        self,
        *,
        run_id: Optional[str] = None,
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
        """Iterate all matching trace summaries across pages.

        Transient failures are retried inside the API client with a bounded
        budget; one that survives retries raises to the caller, who can
        resume from the last completed page by passing ``cursor=``.
        """
        cursor = filters.pop("cursor", None)
        while True:
            page = self.list(cursor=cursor, **filters)
            yield from page.items
            if not page.next_cursor:
                return
            cursor = page.next_cursor

    def get(self, trace_id: str) -> TraceSummary:
        """Get one trace summary."""
        return TraceSummary.model_validate(self.client.get_json(_trace_endpoint(trace_id)))

    def get_raw(self, trace_id: str) -> bytes:
        """Get the stored raw trace document, buffered in memory.

        A trace can be tens of MiB; prefer ``download_raw`` for large traces.
        """
        return b"".join(self.client.stream_bytes(_trace_endpoint(trace_id), params={"raw": "true"}))

    def download_raw(self, trace_id: str, dest: Union[str, Path]) -> int:
        """Stream the raw trace document to ``dest``. Returns bytes written."""
        return self._stream_to_file(_trace_endpoint(trace_id), {"raw": "true"}, dest)

    def _stream_to_file(
        self, endpoint: str, params: Optional[Dict[str, object]], dest: Union[str, Path]
    ) -> int:
        """Stream a response body to ``dest`` without clobbering it on failure.

        Bytes land in a sibling ``.partial`` file that replaces ``dest`` only
        after the stream ends cleanly, so a failed request — or a connection
        cut mid-stream — never truncates an existing file at ``dest``.
        """
        dest = Path(dest)
        partial = dest.with_name(dest.name + ".partial")
        written = 0
        try:
            with open(partial, "wb") as f:
                for chunk in self.client.stream_bytes(endpoint, params=params):
                    f.write(chunk)
                    written += len(chunk)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        partial.replace(dest)
        return written

    # -- traces: delete -----------------------------------------------------

    def delete(self, trace_id: str, *, created_at: Optional[str] = None) -> None:
        """Delete every stored copy of one trace (202 Accepted).

        ``created_at`` is an optional performance hint that lets the service
        prune on its ordering-key prefix; correctness does not depend on it,
        but a hint matching no stored copy is a 404 even when the trace exists
        under another timestamp. For that reason, a hinted delete also
        preserves a 404 received after an ambiguous first attempt: the client
        cannot prove whether that attempt deleted the trace or never reached
        the service, so reporting success could leave a trace behind.

        Raises ``NotFoundError`` when the owner has no such trace — including
        on a repeat of a delete that already succeeded. The design docs
        specify deletion as idempotent at the API level; the service checks
        existence first and answers 404 instead. Callers treating deletion as
        "make sure this is gone" should catch ``NotFoundError``.
        """
        params = {"created_at": created_at} if created_at else None
        self.client.delete(_trace_endpoint(trace_id), params=params)

    def delete_run(self, run_id: str) -> None:
        """Delete every trace in a run (202 Accepted).

        One mutation over the ``run_id`` predicate, not N per-trace calls, and
        synchronous: the service answers 202 with an empty body, so there is
        no job to poll. (The design docs specify ``202 { job_id }``; nothing
        server-side issues one, so no job handle is returned here rather than
        a permanent ``None``.)

        Episode rows are not touched. Raises ``NotFoundError`` when the run
        holds no traces for this owner — see ``delete`` on repeats.
        """
        self.client.delete("/traces", params={"run_id": run_id})

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "TracesClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
