"""Background uploader: one daemon thread draining a budgeted queue into sinks.

Backpressure: submitting never waits. A producer that outruns the uploader
queues up to a record budget and an estimated byte budget, then drops (counted).
One write takes about one request's worth of records, so a slow service costs a
long drain rather than a stalled producer. Fork safety: a child starts over with
an empty queue; the queued records belong to the parent. Containment:
record-specific failures drop only their batch, transient failures get a few
consecutive strikes before the sink is retired (for good, or until
``retire_cooldown`` has passed), and any other failure retires the sink at once.
The thread never dies on one bad batch.
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from prime_traces.batching import DEFAULT_TARGET_BATCH_BYTES

from . import _fork
from ._http import encode_json
from .exceptions import is_record_rejection, is_transient
from .sinks.base import Sink, SinkWriteError, is_episode, to_mapping

logger = logging.getLogger(__name__)

DEFAULT_MAX_QUEUED_RECORDS = 65_536
#: Estimated encoded bytes, a proxy for the memory the queued records hold.
DEFAULT_MAX_QUEUED_BYTES = 4 * 1024**3
DEFAULT_MAX_WRITE_BYTES = DEFAULT_TARGET_BATCH_BYTES
#: Share of the newest sample in the running estimate of a record's size.
RECORD_BYTES_SMOOTHING = 0.25
#: Seconds between warnings about dropped records; a burst can drop hundreds.
DROP_WARNING_INTERVAL = 10.0
#: Consecutive transient failures before a sink is retired.
TRANSIENT_FAILURE_LIMIT = 3


@dataclass
class _Flush:
    """A barrier the caller waits on."""

    event: threading.Event = field(default_factory=threading.Event)


#: "Nothing pulled ahead"; ``None`` is the stop sentinel.
_NOTHING: Any = object()


def deadline_after(timeout: Optional[float]) -> Optional[float]:
    return None if timeout is None else time.monotonic() + max(0.0, timeout)


def time_left(deadline: Optional[float]) -> Optional[float]:
    return None if deadline is None else max(0.0, deadline - time.monotonic())


class UploadWorker:
    """Drains a budgeted queue of record batches into a list of sinks."""

    def __init__(
        self,
        sinks: Sequence[Sink],
        *,
        max_queued_records: int = DEFAULT_MAX_QUEUED_RECORDS,
        max_queued_bytes: int = DEFAULT_MAX_QUEUED_BYTES,
        max_write_bytes: int = DEFAULT_MAX_WRITE_BYTES,
        on_error: Optional[Callable[[str, Exception], None]] = None,
        retire_cooldown: Optional[float] = None,
    ) -> None:
        self.sinks = sinks
        self.max_queued_records = max_queued_records
        self.max_queued_bytes = max_queued_bytes
        self.max_write_bytes = max_write_bytes
        self._on_error = on_error
        #: Seconds after which a sink retired on transient failures is tried
        #: again; ``None`` retires it for the rest of the run.
        self._retire_cooldown = retire_cooldown
        self._queue: "queue.Queue[Any]" = queue.Queue()
        #: Records submitted and not yet taken into a write.
        self._queued_records = 0
        self._queued_lock = threading.Lock()
        #: Running estimate of one record's encoded size, sampled by the uploader
        #: so that submitting never pays for encoding. ``None`` until the first sample.
        self._record_bytes: Optional[float] = None
        self._last_drop_warning: Optional[float] = None
        self._thread: Optional[threading.Thread] = None
        self._stopping = threading.Event()
        self._lock = threading.Lock()
        #: Records never handed to any sink because the queue was full.
        self.dropped = 0
        #: Records a sink could not store, by sink name. Kept apart from
        #: ``dropped``: another sink may well have stored them.
        self.failed_records: dict = {}
        self._transient_failures: dict = {}
        #: Sinks retired by this worker. Records that skip one are counted as
        #: lost to it; a sink that switched itself off quietly is not in here.
        self._retired: set = set()
        #: Sink name -> monotonic time at which a paused sink is tried again.
        self._cooldowns: dict = {}
        _fork.register(self)

    def start(self) -> None:
        with self._lock:
            self._start_locked()

    def _start_locked(self) -> None:
        if self._stopping.is_set():  # a closed worker is one-shot
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="prime-runs-uploader", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        # An item pulled off the queue while coalescing that could not join the
        # batch; it is processed next, ahead of anything queued after it.
        pending: Any = _NOTHING
        while True:
            item = pending if pending is not _NOTHING else self._queue.get()
            pending = _NOTHING
            try:
                if item is None:
                    return
                if isinstance(item, _Flush):
                    self._flush_sinks()
                    item.event.set()
                else:
                    batch, pending = self._coalesce(item)
                    with self._queued_lock:
                        self._queued_records -= len(batch)
                    self._dispatch(batch)
            except Exception as exc:  # noqa: BLE001 - the thread must outlive one bad batch
                logger.debug("Uploader iteration failed: %s", exc)

    def _coalesce(self, first: Sequence[Any]) -> "tuple[list, Any]":
        """Merge what is already queued behind ``first`` into one batch of about
        ``max_write_bytes``, so the request rate tracks upload latency rather
        than rollout throughput, and one slow or failed write stays small.
        Stops at a control item (a flush barrier must not be reordered past the
        records before it) and at a batch of the other record kind (the traces
        sink infers the line format from the first record). Records past the
        limit are processed next, ahead of anything queued after them."""
        batch = list(first)
        kind = is_episode(batch[0]) if batch else None
        if batch:
            self._sample_record_bytes(batch[0])
        limit = self._write_limit()
        while True:
            if len(batch) >= limit:
                return batch[:limit], (batch[limit:] or _NOTHING)
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return batch, _NOTHING
            if item is None or isinstance(item, _Flush):
                return batch, item
            if item and (kind is None or is_episode(item[0]) != kind):
                return batch, item
            batch.extend(item)

    def _sample_record_bytes(self, record: Any) -> None:
        try:
            size = len(encode_json(to_mapping(record)))
        except Exception as exc:  # noqa: BLE001 - the sink decides what a bad record means
            logger.debug("Could not size a record: %s", exc)
            return
        if self._record_bytes is None:
            self._record_bytes = float(size)
        else:
            self._record_bytes += RECORD_BYTES_SMOOTHING * (size - self._record_bytes)

    def _write_limit(self) -> int:
        """Records per write. Without a size estimate, as many as may be queued."""
        if not self._record_bytes:
            return max(1, self.max_queued_records)
        return max(1, int(self.max_write_bytes / self._record_bytes))

    def _dispatch(self, records: Sequence[Any]) -> None:
        for sink in self.sinks:
            self._maybe_revive(sink)
            if not sink.enabled:
                if sink.name in self._retired:
                    count = self.failed_records.get(sink.name, 0)
                    self.failed_records[sink.name] = count + len(records)
                continue
            try:
                sink.write(records)
            except SinkWriteError as exc:
                self._fail_sink(sink, exc.cause, dropped=exc.failed_records)
            except Exception as exc:  # noqa: BLE001 - one sink failing must not stop the others
                self._fail_sink(sink, exc, dropped=len(records))
            else:
                self._transient_failures.pop(sink.name, None)

    def _maybe_revive(self, sink: Sink) -> None:
        until = self._cooldowns.get(sink.name)
        if until is None or time.monotonic() < until:
            return
        del self._cooldowns[sink.name]
        self._retired.discard(sink.name)
        self._transient_failures.pop(sink.name, None)
        sink.enabled = True
        logger.info("Sink %s re-enabled after its cooldown", sink.name)

    def _flush_sinks(self) -> None:
        for sink in self.sinks:
            if not sink.enabled:
                continue
            try:
                sink.flush()
            except Exception as exc:  # noqa: BLE001
                self._fail_sink(sink, exc)

    def _fail_sink(self, sink: Sink, exc: Exception, *, dropped: int = 0) -> None:
        name = sink.name
        if dropped:
            self.failed_records[name] = self.failed_records.get(name, 0) + dropped

        if dropped and is_record_rejection(exc):
            logger.warning(
                "Sink %s rejected a batch of %d record(s) (%s: %s); "
                "still enabled for later records.",
                name,
                dropped,
                type(exc).__name__,
                exc,
            )
            self._notify(name, exc)
            return

        if is_transient(exc):
            strikes = self._transient_failures.get(name, 0) + 1
            self._transient_failures[name] = strikes
            if strikes < TRANSIENT_FAILURE_LIMIT:
                logger.warning(
                    "Sink %s dropped a batch of %d record(s) (%s: %s); "
                    "strike %d of %d, still enabled.",
                    name,
                    dropped,
                    type(exc).__name__,
                    exc,
                    strikes,
                    TRANSIENT_FAILURE_LIMIT,
                )
                self._notify(name, exc)
                return
            sink.enabled = False
            self._retired.add(name)
            if self._retire_cooldown is None:
                logger.warning(
                    "Sink %s disabled after %d consecutive transient failures: %s: %s",
                    name,
                    strikes,
                    type(exc).__name__,
                    exc,
                )
            else:
                self._cooldowns[name] = time.monotonic() + self._retire_cooldown
                logger.warning(
                    "Sink %s paused for %.0fs after %d consecutive transient failures: %s: %s",
                    name,
                    self._retire_cooldown,
                    strikes,
                    type(exc).__name__,
                    exc,
                )
        else:
            sink.enabled = False
            self._retired.add(name)
            logger.warning("Sink %s disabled after an error: %s: %s", name, type(exc).__name__, exc)
        self._notify(name, exc)

    def _notify(self, name: str, exc: Exception) -> None:
        if self._on_error is None:
            return
        try:
            self._on_error(name, exc)
        except Exception:  # noqa: BLE001 - the handler is the caller's problem
            logger.debug("Error handler raised while reporting a sink failure", exc_info=True)

    def submit(self, records: Sequence[Any]) -> bool:
        """Queue a batch without waiting. ``False`` means it was dropped: the
        queue is over its budget, or the worker is closed."""
        if not self.sinks:  # a disabled run: nowhere for the records to go
            return True
        # Serialize with close(): an accepted batch is queued before the stop sentinel.
        with self._lock:
            if self._stopping.is_set():
                return False
            self._start_locked()
            count = len(records)
            with self._queued_lock:
                accepted = self._has_room(count)
                if accepted:
                    self._queued_records += count
            if accepted:
                self._queue.put_nowait(records)
                return True
            self.dropped += count
            self._warn_dropped(count)
            return False

    def _has_room(self, count: int) -> bool:
        """An empty queue takes any batch, so an oversized one is not dropped outright."""
        queued = self._queued_records
        if not queued:
            return True
        total = queued + count
        if total > self.max_queued_records:
            return False
        return self._record_bytes is None or total * self._record_bytes <= self.max_queued_bytes

    def _warn_dropped(self, count: int) -> None:
        now = time.monotonic()
        last = self._last_drop_warning
        if last is not None and now - last < DROP_WARNING_INTERVAL:
            return
        self._last_drop_warning = now
        logger.warning(
            "Upload queue over budget (%d record(s) queued); dropped %d item(s) (%d total). "
            "The producer is outrunning the uploader.",
            self._queued_records,
            count,
            self.dropped,
        )

    def flush(self, timeout: Optional[float] = None) -> bool:
        """Block until everything queued so far has been written."""
        if self._thread is None or not self._thread.is_alive():
            self._flush_sinks()
            return True
        barrier = _Flush()
        self._queue.put_nowait(barrier)
        return barrier.event.wait(timeout if timeout is None else max(0.0, timeout))

    def close(self, timeout: Optional[float] = 30.0) -> None:
        """Drain, stop the thread, and close every sink."""
        with self._lock:
            self._stopping.set()
            thread = self._thread
        if thread is not None and thread.is_alive():
            self._queue.put_nowait(None)
            thread.join(timeout if timeout is None else max(0.0, timeout))
            if thread.is_alive():
                # Closing the sinks now would pull a client out from under a request.
                logger.warning(
                    "Uploader still running after %ss with %d record(s) queued; leaving it "
                    "and its sinks open. They may not finish before the process exits.",
                    timeout,
                    self._queued_records,
                )
                return
        self._thread = None
        for sink in self.sinks:
            try:
                sink.close()
            except Exception as exc:  # noqa: BLE001 - teardown must not raise
                logger.debug("Error closing sink %s: %s", sink.name, exc)

    def reset_after_fork(self) -> None:
        self._queue = queue.Queue()
        self._queued_records = 0
        self._queued_lock = threading.Lock()
        self._thread = None
        self._stopping = threading.Event()
        self._lock = threading.Lock()
