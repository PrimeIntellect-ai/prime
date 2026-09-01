"""Background uploader: one daemon thread draining a bounded queue into sinks.

Backpressure: the queue is bounded, so a producer that outruns the uploader
blocks briefly and then drops (counted) rather than stalling the run.

Fork safety: a forked child inherits the queue's memory but not the thread.
The child starts over empty; the queued records belong to the parent.

Containment: record-specific failures drop only their batch. A sink-wide
failure disables the sink, while transient failures get a few consecutive
strikes first — and, with a ``retire_cooldown``, the sink is tried again once
the cooldown has passed, so an outage costs a long run a window of records
rather than the rest of the run. The thread never dies on one bad batch.
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from . import _fork
from .exceptions import is_record_rejection, is_transient
from .sinks.base import Sink, SinkWriteError, is_episode

logger = logging.getLogger(__name__)

DEFAULT_QUEUE_SIZE = 256
DEFAULT_PUT_TIMEOUT = 5.0
#: Consecutive transient failures before a sink is retired.
TRANSIENT_FAILURE_LIMIT = 3


@dataclass
class _Flush:
    """A barrier the caller waits on."""

    event: threading.Event = field(default_factory=threading.Event)


#: "Nothing pulled ahead" — distinct from ``None``, which is the stop sentinel.
_NOTHING: Any = object()


def _deadline(timeout: Optional[float]) -> Optional[float]:
    if timeout is None:
        return None
    return time.monotonic() + max(0.0, timeout)


def _remaining(deadline: Optional[float]) -> Optional[float]:
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


class UploadWorker:
    """Drains a bounded queue of record batches into a list of sinks."""

    def __init__(
        self,
        sinks: Sequence[Sink],
        *,
        max_queue_size: int = DEFAULT_QUEUE_SIZE,
        put_timeout: float = DEFAULT_PUT_TIMEOUT,
        on_error: Optional[Callable[[str, Exception], None]] = None,
        retire_cooldown: Optional[float] = None,
    ) -> None:
        self.sinks = sinks
        self.max_queue_size = max_queue_size
        self.put_timeout = put_timeout
        self._on_error = on_error
        #: Seconds after which a sink retired on transient failures is tried
        #: again; ``None`` retires it for the rest of the run.
        self._retire_cooldown = retire_cooldown
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=max_queue_size)
        self._thread: Optional[threading.Thread] = None
        self._stopping = threading.Event()
        self._lock = threading.Lock()
        #: Records never handed to any sink because the queue was full.
        self.dropped = 0
        #: Records a particular sink could not store, by sink name. Kept apart
        #: from ``dropped``: another sink may well have stored them.
        self.failed_records: dict = {}
        self._transient_failures: dict = {}
        #: Sinks this worker retired after a failure. Records that skip one of
        #: these are lost to it and counted; a sink that switched itself off
        #: without raising (nowhere for the records to go) is not in here.
        self._retired: set = set()
        #: Sink name -> monotonic time at which a paused sink is tried again.
        self._cooldowns: dict = {}
        _fork.register(self)

    # ----------------------------------------------------------------- thread

    def start(self) -> None:
        with self._lock:
            self._start_locked()

    def _start_locked(self) -> None:
        """Start while ``_lock`` is held; a closed worker is one-shot."""
        if self._stopping.is_set():
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="prime-runs-uploader", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        # An item pulled off the queue while coalescing that could not join the
        # batch (a control item — the stop sentinel is None, hence the marker —
        # or records of the other kind); it is the next thing to process, ahead
        # of anything queued after it.
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
                    self._dispatch(batch)
            except Exception as exc:  # noqa: BLE001 - the thread must outlive one bad batch
                logger.debug("Uploader iteration failed: %s", exc)
            finally:
                self._queue.task_done()

    def _coalesce(self, first: Sequence[Any]) -> "tuple[list, Any]":
        """Merge everything already queued behind ``first`` into one batch.

        A producer hands over one episode at a time as rollouts finish, and
        each batch costs every sink a request: one ``POST /samples`` per
        episode runs into the platform's per-minute limit on any fast eval,
        and the retries then back the queue up until records are dropped.
        Draining what has accumulated while the previous request was in
        flight makes the request rate track upload latency instead of rollout
        throughput. Sinks split a large batch by size themselves.

        Stops at a control item (a flush barrier must not be reordered past
        the records queued before it) and at a batch of the other record kind
        (the traces sink infers the line format from the first record).
        Returns the merged batch and the item that stopped it, if any.
        """
        batch = list(first)
        kind = is_episode(batch[0]) if batch else None
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return batch, _NOTHING
            if item is None or isinstance(item, _Flush):
                return batch, item
            if item and (kind is None or is_episode(item[0]) != kind):
                return batch, item
            batch.extend(item)
            self._queue.task_done()

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
        """Re-enable a sink whose cooldown has passed. Its strikes start over."""
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
        """Account for a failed operation and decide whether to retire its sink.

        Record-specific failures drop only the current batch. Sink-wide
        permanent failures retire immediately; transient failures retire after
        ``TRANSIENT_FAILURE_LIMIT`` consecutive strikes — for good, or until
        the worker's ``retire_cooldown`` has passed.
        """
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

    # ------------------------------------------------------------------ queue

    def submit(self, records: Sequence[Any]) -> bool:
        """Hand a batch to the uploader. ``False`` means it was dropped: the
        queue stayed full for the whole put timeout."""
        # No sinks (a disabled run): there is nowhere for the records to go, so
        # do not copy them into a queue or start a thread to find that out.
        # Retired sinks are not the same case — their losses are counted.
        if not self.sinks:
            return True
        # Serialize acceptance with close(): an accepted batch is queued before
        # the stop sentinel, while a submission after close is refused.
        with self._lock:
            if self._stopping.is_set():
                return False
            self._start_locked()
            try:
                self._queue.put(records, timeout=self.put_timeout)
                return True
            except queue.Full:
                count = len(records)
                self.dropped += count
                logger.warning(
                    "Upload queue full after %.1fs; dropped %d item(s) (%d total). "
                    "The producer is outrunning the uploader.",
                    self.put_timeout,
                    count,
                    self.dropped,
                )
                return False

    def flush(self, timeout: Optional[float] = None) -> bool:
        """Block until everything queued so far has been written."""
        if self._thread is None or not self._thread.is_alive():
            self._flush_sinks()
            return True
        deadline = _deadline(timeout)
        barrier = _Flush()
        try:
            # The barrier gets the caller's drain budget, not the short
            # producer put timeout: a full queue is when finish() most needs it.
            self._queue.put(barrier, timeout=_remaining(deadline))
        except queue.Full:
            logger.warning("Could not enqueue a flush barrier before the drain timeout")
            return False
        return barrier.event.wait(_remaining(deadline))

    def close(self, timeout: Optional[float] = 30.0) -> None:
        """Drain, stop the thread, and close every sink."""
        with self._lock:
            self._stopping.set()
            thread = self._thread
        if thread is not None and thread.is_alive():
            deadline = _deadline(timeout)
            try:
                self._queue.put(None, timeout=_remaining(deadline))
            except queue.Full:
                logger.warning("Upload queue remained saturated through the close timeout")
            thread.join(_remaining(deadline))
            if thread.is_alive():
                # Closing the sinks now would pull a client or file handle out
                # from under a request still running on that thread.
                logger.warning(
                    "Uploader still running after %ss; leaving it and its sinks open. "
                    "Records still in flight may not finish before the process exits.",
                    timeout,
                )
                return
        self._thread = None
        for sink in self.sinks:
            try:
                sink.close()
            except Exception as exc:  # noqa: BLE001 - teardown must not raise
                logger.debug("Error closing sink %s: %s", sink.name, exc)

    # ------------------------------------------------------------------- fork

    def reset_after_fork(self) -> None:
        """Give the child a clean uploader; what was queued belongs to the parent."""
        self._queue = queue.Queue(maxsize=self.max_queue_size)
        self._thread = None
        self._stopping = threading.Event()
        self._lock = threading.Lock()
