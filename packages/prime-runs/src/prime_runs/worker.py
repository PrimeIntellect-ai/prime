"""Background uploader: one daemon thread draining a bounded queue into sinks.

Backpressure: the queue is bounded, so a producer that outruns the uploader
blocks briefly and then drops (counted) rather than stalling the run.

Fork safety: a forked child inherits the queue's memory but not the thread.
The child starts over empty; the queued records belong to the parent.

Containment: a sink that raises is given a few consecutive transient strikes,
then disabled for the rest of the run. The thread never dies on one bad batch.
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence

from . import _fork
from .exceptions import is_transient

logger = logging.getLogger(__name__)

DEFAULT_QUEUE_SIZE = 256
DEFAULT_PUT_TIMEOUT = 5.0
#: Consecutive transient failures before a sink is retired.
TRANSIENT_FAILURE_LIMIT = 3


@dataclass
class _Flush:
    """A barrier the caller waits on."""

    event: threading.Event = field(default_factory=threading.Event)


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
        sinks: List[Any],
        *,
        max_queue_size: int = DEFAULT_QUEUE_SIZE,
        put_timeout: float = DEFAULT_PUT_TIMEOUT,
        on_error: Optional[Callable[[str, Exception], None]] = None,
    ) -> None:
        self.sinks = sinks
        self.max_queue_size = max_queue_size
        self.put_timeout = put_timeout
        self._on_error = on_error
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
        _fork.register(self)

    # ----------------------------------------------------------------- thread

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stopping.clear()
            self._thread = threading.Thread(
                target=self._run, name="prime-runs-uploader", daemon=True
            )
            self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                if isinstance(item, _Flush):
                    self._flush_sinks()
                    item.event.set()
                else:
                    self._dispatch(item)
            except Exception as exc:  # noqa: BLE001 - the thread must outlive one bad batch
                logger.debug("Uploader iteration failed: %s", exc)
            finally:
                self._queue.task_done()

    def _dispatch(self, records: Sequence[Any]) -> None:
        for sink in self.sinks:
            if not getattr(sink, "enabled", True):
                continue
            try:
                sink.write(records)
            except Exception as exc:  # noqa: BLE001 - one sink failing must not stop the others
                self._fail_sink(sink, exc, dropped=len(records))
            else:
                self._transient_failures.pop(getattr(sink, "name", id(sink)), None)

    def _flush_sinks(self) -> None:
        for sink in self.sinks:
            if not getattr(sink, "enabled", True):
                continue
            try:
                sink.flush()
            except Exception as exc:  # noqa: BLE001
                self._fail_sink(sink, exc)

    def _fail_sink(self, sink: Any, exc: Exception, *, dropped: int = 0) -> None:
        """The batch is gone (the transports already retried). Decide whether
        the sink is too: permanent failures retire it at once, transient ones
        after ``TRANSIENT_FAILURE_LIMIT`` consecutive strikes."""
        name = getattr(sink, "name", type(sink).__name__)
        if dropped:
            self.failed_records[name] = self.failed_records.get(name, 0) + dropped

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
            logger.warning(
                "Sink %s disabled after %d consecutive transient failures: %s: %s",
                name,
                strikes,
                type(exc).__name__,
                exc,
            )
        else:
            sink.enabled = False
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
        if self._stopping.is_set():
            return False
        thread = self._thread
        if thread is None or not thread.is_alive():
            self.start()
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
                logger.debug("Error closing sink %s: %s", getattr(sink, "name", sink), exc)

    # ------------------------------------------------------------------- fork

    def reset_after_fork(self) -> None:
        """Give the child a clean uploader; what was queued belongs to the parent."""
        self._queue = queue.Queue(maxsize=self.max_queue_size)
        self._thread = None
        self._stopping = threading.Event()
        self._lock = threading.Lock()
