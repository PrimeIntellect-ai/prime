"""High-level client for Prime Traces.

Wraps the wire client with upload batching/retry and typed read paths.

The read surface (list/get/episode envelopes, export params) is provisional:
the service defines these routes but has not pinned response models yet, so
the shapes here are a proposal to align on, not a settled contract.

Deliberately not implemented yet (open v0 contract decisions — do not freeze
them here): the exports *job* API (``POST /traces/exports`` is published as
501 in v0; the streaming ``GET /traces/export`` is what ``export`` wraps),
``/search``, the ``environment_id`` filter (no populated column behind it
yet), episode writes (episodes are read-only, written only as a side effect
of episode-grouped uploads), and the dot-path query compiler (needs the
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
from .exceptions import RetryableAPIError, TransportError
from .models import (
    EpisodeListPage,
    EpisodeSummary,
    LineFormat,
    TraceListPage,
    TraceSummary,
    UploadReceipt,
)

DEFAULT_MAX_ATTEMPTS = 5
_BACKOFF_BASE_SECONDS = 1.0
_BACKOFF_CAP_SECONDS = 30.0
# Retry-After is server-controlled input (and may come from a gateway's
# HTTP-date far in the future); honor it, but never let it park the uploader.
_RETRY_AFTER_CAP_SECONDS = 60.0


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
                delay = getattr(exc, "retry_after", None)
                if delay is None:
                    delay = min(
                        _BACKOFF_CAP_SECONDS,
                        _BACKOFF_BASE_SECONDS * (2**attempt),
                    ) * (0.5 + random.random())
                else:
                    delay = min(delay, _RETRY_AFTER_CAP_SECONDS)
                time.sleep(delay)
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

        Reads are not retried (only uploads are): a transient failure
        mid-iteration raises to the caller, who can resume from the last
        page by passing ``cursor=``.
        """
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
        return self._stream_to_file(f"/traces/{trace_id}", {"raw": "true"}, dest)

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
        prune on its ordering-key prefix; correctness does not depend on it.
        """
        params = {"created_at": created_at} if created_at else None
        self.client.delete_json(f"/traces/{trace_id}", params=params)

    def delete_run(self, run_id: str) -> Optional[str]:
        """Delete every trace in a run. Returns the async job id, if any."""
        result = self.client.delete_json("/traces", params={"run_id": run_id})
        job_id = result.get("job_id")
        return str(job_id) if job_id is not None else None

    # -- export -------------------------------------------------------------

    def export(
        self,
        dest: Union[str, Path],
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
    ) -> int:
        """Stream a filtered export (``GET /traces/export``) to ``dest``.

        Takes the same filter vocabulary as ``list`` — no pagination: the
        export is one file, resumable by re-running. Returns bytes written.
        Egress is metered on bytes actually sent, so a failed stream still
        costs for the bytes that made it; always stream to disk rather than
        buffering.

        A format parameter (raw JSONL vs. column projection) is not exposed
        yet — the service route does not define it; add it here when it lands.
        The exports *job* API is 501 in v0 and is deliberately not wrapped.
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
            ),
            context,
        )
        return self._stream_to_file("/traces/export", params, dest)

    # -- episodes (read-only in v0) -----------------------------------------

    def list_episodes(
        self,
        *,
        run_id: Optional[str] = None,
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> EpisodeListPage:
        params: Dict[str, object] = {}
        for key, value in (
            ("run_id", run_id),
            ("created_after", created_after),
            ("created_before", created_before),
            ("limit", limit),
            ("cursor", cursor),
        ):
            if value is not None:
                params[key] = value
        return EpisodeListPage.model_validate(self.client.get_json("/episodes", params=params))

    def get_episode(self, episode_id: str) -> EpisodeSummary:
        return EpisodeSummary.model_validate(self.client.get_json(f"/episodes/{episode_id}"))

    def list_episode_traces(
        self,
        episode_id: str,
        *,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> TraceListPage:
        params: Dict[str, object] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return TraceListPage.model_validate(
            self.client.get_json(f"/episodes/{episode_id}/traces", params=params)
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "TracesClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
