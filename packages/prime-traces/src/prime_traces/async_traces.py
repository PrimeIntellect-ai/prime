"""Async high-level client for Prime Traces.

The asyncio counterpart of ``traces.TracesClient``, with the same method
names, arguments, return types and failure semantics. Everything that decides
*what* a call does — filter encoding, ID encoding, record serialization,
batching, retry classification — is imported from the sync modules rather than
restated, so the two surfaces cannot answer the same question differently. The
async client owns only the awaiting, plus two things a sync client never has
to think about:

- Blocking work is moved off the event loop. Reading a JSONL file, hashing a
  batch, gzipping a body and writing a download to disk all happen in worker
  threads, so one 30 MiB upload does not stall every other task in the
  process.
- Producers may be async. ``upload_records`` and ``upload_lines`` accept an
  async iterable as readily as a sync one, so rollouts can stream straight
  from an async generator without being collected first.

Uploads still send one batch at a time. The contract allows 2–8 requests in
flight per producer, but that is a throughput decision to make against the
running service, and a concurrent version has to answer what a durable
rejection means for batches already in flight — so the ordering, the receipt
sequence and the stop-on-400 behaviour stay identical to the sync client for
now.
"""

import asyncio
import inspect
import tempfile
from collections.abc import AsyncIterable
from contextlib import aclosing
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Union,
)

from .batching import (
    DEFAULT_TARGET_BATCH_BYTES,
    Batch,
    aiter_batches,
    read_jsonl_lines,
)
from .core.async_client import AsyncTracesAPIClient, retry_sleep
from .core.client import retry_delay
from .exceptions import RetryableAPIError, TransportError
from .models import (
    EpisodeDetail,
    EpisodeListPage,
    LineFormat,
    TraceListPage,
    TraceSummary,
    UploadReceipt,
)
from .traces import (
    DEFAULT_MAX_ATTEMPTS,
    TraceRecord,
    _build_params,
    _encode_record,
    _episode_endpoint,
    _record_lines,
    _trace_endpoint,
)

#: An ``on_batch`` hook may be a plain callable or a coroutine function; the
#: uploader awaits whatever it returns if that turns out to be awaitable.
BatchCallback = Callable[[Batch, UploadReceipt], Union[None, Awaitable[None]]]


async def _arecord_lines(records: AsyncIterable[TraceRecord]) -> AsyncIterator[bytes]:
    """Serialize records from an async producer as they arrive."""
    record_number = 0
    iterator = aiter(records)
    try:
        async for record in iterator:
            record_number += 1
            yield await _run_sync_operation_safely(_encode_record, record, record_number)
    finally:
        close = getattr(iterator, "aclose", None)
        if close is not None:
            await close()


def _record_lines_for(
    records: Union[Iterable[TraceRecord], AsyncIterable[TraceRecord]],
) -> Union[Iterable[bytes], AsyncIterator[bytes]]:
    """Encode records, preserving whether the source is sync or async.

    ``aiter_batches`` consumes a synchronous source entirely in a worker
    thread, so handing it the sync generator — rather than adapting it to an
    async one here — is what keeps ``json.dumps`` over large records off the
    event loop.
    """
    if isinstance(records, AsyncIterable):
        return _arecord_lines(records)
    return _record_lines(records)


def _open_partial_file(dest: Path):
    """Open the sibling temporary file a download is staged through."""
    return tempfile.NamedTemporaryFile(
        mode="wb",
        dir=dest.parent,
        prefix=".prime-traces-",
        suffix=".partial",
        delete=False,
    )


def _discard_partial_file(handle: Any, partial: Path) -> None:
    """Close a staged download and unlink it even when close/flush fails."""
    try:
        handle.close()
    finally:
        partial.unlink(missing_ok=True)


async def _open_partial_file_safely(dest: Path):
    """Open a partial file without leaking it when the caller is cancelled.

    Cancelling ``to_thread`` does not stop a worker that is already running.
    Shield the worker and, if cancellation arrives, wait until it relinquishes
    the handle before closing and unlinking the file it may have created.
    """
    open_task = asyncio.create_task(asyncio.to_thread(_open_partial_file, dest))
    try:
        return await asyncio.shield(open_task)
    except asyncio.CancelledError:
        # Repeated cancellation requests must not cancel ``open_task`` either;
        # without its result there is no path to the delete=False filename.
        while not open_task.done():
            try:
                await asyncio.shield(open_task)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break

        if not open_task.cancelled():
            try:
                handle = open_task.result()
            except BaseException:
                pass
            else:
                try:
                    _discard_partial_file(handle, Path(handle.name))
                except BaseException:
                    # Cancellation is the caller-visible outcome. Cleanup is
                    # best-effort, but close must never prevent the unlink.
                    pass
        raise


async def _run_sync_operation_safely(operation: Callable[..., Any], *args: Any) -> Any:
    """Run synchronous work in a worker without abandoning it on cancellation.

    Cancelling an await of ``to_thread`` does not stop a worker that has already
    started. Keep the worker task shielded and wait for it to finish before
    propagating cancellation, so producer or file cleanup never races with
    outstanding synchronous work. Waiting remains asynchronous, which keeps
    the event loop responsive while the worker finishes.
    """
    operation_task = asyncio.create_task(asyncio.to_thread(operation, *args))
    try:
        return await asyncio.shield(operation_task)
    except asyncio.CancelledError:
        # Repeated cancellation requests must not detach the worker from the
        # file it still owns. Preserve cancellation, but only after the worker
        # has completed and cleanup can safely use the handle.
        while not operation_task.done():
            try:
                await asyncio.shield(operation_task)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break

        if not operation_task.cancelled():
            try:
                operation_task.result()
            except BaseException:
                # Cancellation remains the caller-visible outcome. Retrieving
                # a concurrent operation error prevents an unobserved-task
                # warning while the outer cleanup releases owned resources.
                pass
        raise


class AsyncTracesClient:
    """Async client for the Prime Traces API."""

    def __init__(self, api_client: Optional[AsyncTracesAPIClient] = None, **client_kwargs):
        self.client = api_client or AsyncTracesAPIClient(**client_kwargs)

    # -- upload -------------------------------------------------------------

    async def upload_records(
        self,
        records: Union[Iterable[TraceRecord], AsyncIterable[TraceRecord]],
        *,
        line_format: LineFormat = LineFormat.TRACE,
        context: Optional[Dict[str, str]] = None,
        schema_version: int = 1,
        compress: bool = True,
        target_batch_bytes: int = DEFAULT_TARGET_BATCH_BYTES,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        on_batch: Optional[BatchCallback] = None,
    ) -> List[UploadReceipt]:
        """Upload trace or episode records directly from memory.

        Each input may be a JSON-compatible mapping or an object implementing
        ``to_record()``, including verifiers ``Trace`` / ``Episode`` and
        prime-rl ``Rollout`` objects. The records may arrive from an async
        producer: an async generator of rollouts uploads incrementally, in
        bounded batches, without ever being collected into a list.
        """
        return await self.upload_lines(
            _record_lines_for(records),
            line_format=line_format,
            context=context,
            schema_version=schema_version,
            compress=compress,
            target_batch_bytes=target_batch_bytes,
            max_attempts=max_attempts,
            on_batch=on_batch,
        )

    async def upload_file(
        self,
        path: Union[str, Path],
        *,
        line_format: LineFormat = LineFormat.TRACE,
        context: Optional[Dict[str, str]] = None,
        schema_version: int = 1,
        compress: bool = True,
        target_batch_bytes: int = DEFAULT_TARGET_BATCH_BYTES,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        on_batch: Optional[BatchCallback] = None,
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

        The file is read in a worker thread, so a large upload does not block
        the loop between requests.
        """
        return await self.upload_lines(
            read_jsonl_lines(path),
            line_format=line_format,
            context=context,
            schema_version=schema_version,
            compress=compress,
            target_batch_bytes=target_batch_bytes,
            max_attempts=max_attempts,
            on_batch=on_batch,
        )

    async def upload_lines(
        self,
        lines: Union[Iterable[bytes], AsyncIterable[bytes]],
        *,
        line_format: LineFormat = LineFormat.TRACE,
        context: Optional[Dict[str, str]] = None,
        schema_version: int = 1,
        compress: bool = True,
        target_batch_bytes: int = DEFAULT_TARGET_BATCH_BYTES,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        on_batch: Optional[BatchCallback] = None,
    ) -> List[UploadReceipt]:
        """Upload raw JSONL lines, sync or async. See ``upload_file``."""
        receipts: List[UploadReceipt] = []
        async with aclosing(aiter_batches(lines, target_bytes=target_batch_bytes)) as batches:
            async for batch in batches:
                result = await self._send_with_retry(
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
                    outcome = on_batch(batch, receipt)
                    if inspect.isawaitable(outcome):
                        await outcome
        return receipts

    async def _send_with_retry(
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
                return await self.client.upload_batch(
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
                await retry_sleep(retry_delay(exc, attempt))
        assert last_error is not None
        raise last_error

    # -- traces: read -------------------------------------------------------

    async def list(
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
        return TraceListPage.model_validate(await self.client.get_json("/traces", params=params))

    async def iter(self, **filters: Any) -> AsyncIterator[TraceSummary]:
        """Iterate all matching trace summaries across pages.

        Transient failures are retried inside the API client with a bounded
        budget; one that survives retries raises to the caller, who can
        resume from the last completed page by passing ``cursor=``.
        """
        cursor = filters.pop("cursor", None)
        while True:
            page = await self.list(cursor=cursor, **filters)
            for summary in page.items:
                yield summary
            if not page.next_cursor:
                return
            cursor = page.next_cursor

    async def get(self, trace_id: str) -> TraceSummary:
        """Get one trace summary."""
        return TraceSummary.model_validate(await self.client.get_json(_trace_endpoint(trace_id)))

    async def get_raw(self, trace_id: str) -> bytes:
        """Get the stored raw trace document, buffered in memory.

        A trace can be tens of MiB; prefer ``download_raw`` for large traces.
        """
        stream = self.client.stream_bytes(_trace_endpoint(trace_id), params={"raw": "true"})
        async with aclosing(stream) as chunks:
            buffered = [chunk async for chunk in chunks]
        return b"".join(buffered)

    async def download_raw(self, trace_id: str, dest: Union[str, Path]) -> int:
        """Stream the raw trace document to ``dest``. Returns bytes written."""
        return await self._stream_to_file(_trace_endpoint(trace_id), {"raw": "true"}, dest)

    async def _stream_to_file(
        self, endpoint: str, params: Optional[Dict[str, object]], dest: Union[str, Path]
    ) -> int:
        """Stream a response body to ``dest`` without clobbering it on failure.

        Bytes land in a uniquely named sibling temporary file that replaces
        ``dest`` only after the stream ends cleanly, so a failed request — or a
        connection cut mid-stream — never truncates an existing file at
        ``dest`` or another download's temporary file.

        Writes go to a worker thread so a large download does not block the
        loop between chunks. File cleanup stays synchronous on purpose: it also
        runs when the task is being cancelled, where awaiting again would
        abandon the partial file on disk. The response iterator is explicitly
        closed before that cleanup so it cannot retain a pooled connection.
        """
        dest = Path(dest)
        handle = await _open_partial_file_safely(dest)
        partial = Path(handle.name)
        written = 0
        try:
            stream = self.client.stream_bytes(endpoint, params=params)
            async with aclosing(stream) as chunks:
                async for chunk in chunks:
                    await _run_sync_operation_safely(handle.write, chunk)
                    written += len(chunk)
            await _run_sync_operation_safely(handle.close)
            # Replacing a sibling file is a fast, atomic metadata operation.
            # Keep this commit step on the event-loop thread so cancellation
            # cannot report failure after a detached worker replaced ``dest``.
            partial.replace(dest)
        except BaseException:
            try:
                _discard_partial_file(handle, partial)
            except BaseException:
                # Preserve the stream, filesystem or cancellation failure that
                # brought us here; cleanup has still attempted both operations.
                pass
            raise
        return written

    # -- traces: delete -----------------------------------------------------

    async def delete(self, trace_id: str, *, created_at: Optional[str] = None) -> None:
        """Delete every stored copy of one trace (202 Accepted).

        ``created_at`` is an optional performance hint that lets the service
        prune on its ordering-key prefix; correctness does not depend on it,
        but a hint matching no stored copy is a 404 even when the trace exists
        under another timestamp.

        Ambiguous transport and gateway failures raise
        ``AmbiguousDeleteError`` without retrying: the first request may
        already have deleted this trace, and replaying it could delete a new
        copy uploaded between attempts.

        Raises ``NotFoundError`` when the owner has no such trace — including
        on a repeat of a delete that already succeeded. Callers treating
        deletion as "make sure this is gone" should catch ``NotFoundError``.
        """
        params = {"created_at": created_at} if created_at else None
        await self.client.delete(_trace_endpoint(trace_id), params=params)

    async def delete_run(self, run_id: str) -> None:
        """Delete every trace in a run (202 Accepted).

        One mutation over the ``run_id`` predicate, not N per-trace calls, and
        synchronous on the service side: it answers 202 with an empty body, so
        there is no job to poll.

        Episode rows are not touched. Raises ``NotFoundError`` when the run
        holds no traces for this owner — see ``delete`` on repeats. An
        ``AmbiguousDeleteError`` is not retried because a replay could delete
        traces added to the run after the first request.
        """
        await self.client.delete("/traces", params={"run_id": run_id})

    # -- episodes (read-only in v0) -----------------------------------------

    async def list_episodes(
        self,
        *,
        run_id: Optional[str] = None,
        environment_id: Optional[str] = None,
        outcome: Optional[str] = None,
        has_error: Optional[bool] = None,
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> EpisodeListPage:
        """List episode summaries using the server's complete filter set.

        ``environment_id`` is extracted from the canonical episode
        ``env.id``. Episodes carry no upload ``context`` map.
        """
        params = _build_params(
            (
                ("run_id", run_id),
                ("environment_id", environment_id),
                ("outcome", outcome),
                ("has_error", has_error),
                ("created_after", created_after),
                ("created_before", created_before),
                ("limit", limit),
                ("cursor", cursor),
            )
        )
        return EpisodeListPage.model_validate(
            await self.client.get_json("/episodes", params=params)
        )

    async def get_episode(self, episode_id: str) -> EpisodeDetail:
        """Episode-owned fields plus the read-time member-trace aggregate.

        The response carries the episode row's own ``has_error``/``error``
        alongside ``traces.any_trace_error``, so an environment-hook failure
        stays visible even when every individual trace succeeded.
        """
        return EpisodeDetail.model_validate(
            await self.client.get_json(_episode_endpoint(episode_id))
        )

    async def list_episode_traces(
        self,
        episode_id: str,
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
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> TraceListPage:
        """List an episode's member traces in upload order.

        The filter vocabulary matches the backend member-trace route and the
        top-level trace listing, except that member traces have no ``sort``
        option.
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
                ("limit", limit),
                ("cursor", cursor),
            ),
            context,
        )
        return TraceListPage.model_validate(
            await self.client.get_json(f"{_episode_endpoint(episode_id)}/traces", params=params)
        )

    async def aclose(self) -> None:
        await self.client.aclose()

    async def __aenter__(self) -> "AsyncTracesClient":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.aclose()
