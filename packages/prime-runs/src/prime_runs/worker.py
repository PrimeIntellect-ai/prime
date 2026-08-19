"""Background uploader: the thread that keeps the network off the rollout loop.

Three things a producer should never have to think about, handled once here:

**Backpressure.** The queue is bounded. Verifiers' uploader held every episode
of a run in memory and posted them all at the end, which is fine at a hundred
episodes and is an OOM at a hundred thousand. A bounded queue trades that for a
short block, and — past the block — a counted drop, because stalling a training
run to protect telemetry is the wrong trade in the other direction.

**Fork safety.** Hosted evals fork after the SDK is initialized. A forked child
inherits the queue's *memory* but not the thread that drains it, so anything
already queued would sit there forever and any lock held mid-write stays held.
The child therefore starts over with an empty queue and a fresh thread, and
drops what it inherited: those records belong to the parent, which is still
running and will upload them itself.

**Containment.** A sink that raises is retried once, then disabled for the rest
of the run with the error reported through the run's error handler. The upload
thread never propagates into the producer, and never dies quietly either.
"""

import logging
import os
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_QUEUE_SIZE = 256
DEFAULT_PUT_TIMEOUT = 5.0


@dataclass
class WriteItem:
    """One batch of records destined for every enabled sink."""

    records: Sequence[Any]
    line_format: Optional[str] = None
    step: Optional[int] = None


@dataclass
class MetricItem:
    """One ``log()`` call destined for a backend that stores a time series."""

    metrics: dict
    step: Optional[int] = None


@dataclass
class _Flush:
    """A barrier the caller waits on."""

    event: threading.Event = field(default_factory=threading.Event)


class UploadWorker:
    """Drains a bounded queue into a list of sinks on one daemon thread."""

    def __init__(
        self,
        sinks: List[Any],
        *,
        max_queue_size: int = DEFAULT_QUEUE_SIZE,
        put_timeout: float = DEFAULT_PUT_TIMEOUT,
        on_error: Optional[Callable[[str, Exception], None]] = None,
        metric_writer: Optional[Callable[[dict, Optional[int]], None]] = None,
    ) -> None:
        self.sinks = sinks
        self.max_queue_size = max_queue_size
        self.put_timeout = put_timeout
        self._on_error = on_error
        # Set when the backend stores a real time series. Metrics then ride the
        # same queue as records, so a per-step log() in a training loop costs a
        # queue put rather than an HTTP round trip.
        self._metric_writer = metric_writer
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=max_queue_size)
        self._thread: Optional[threading.Thread] = None
        self._stopping = threading.Event()
        self._lock = threading.Lock()
        self.dropped = 0
        self._pid = os.getpid()
        self._register_fork_hook()

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
                    continue
                if isinstance(item, MetricItem):
                    self._write_metrics(item)
                    continue
                self._dispatch(item)
            except Exception as exc:  # noqa: BLE001 - the thread must outlive one bad batch
                logger.debug("Uploader iteration failed: %s", exc)
            finally:
                self._queue.task_done()

    def _dispatch(self, item: WriteItem) -> None:
        for sink in self.sinks:
            if not getattr(sink, "enabled", True):
                continue
            try:
                sink.write(item.records, line_format=item.line_format, step=item.step)
            except Exception as exc:  # noqa: BLE001 - one sink failing must not stop the others
                self._fail_sink(sink, exc)

    def _write_metrics(self, item: MetricItem) -> None:
        if self._metric_writer is None:
            return
        try:
            self._metric_writer(item.metrics, item.step)
        except Exception as exc:  # noqa: BLE001 - metrics must not kill the uploader
            logger.warning(
                "Dropped metrics for step %s: %s: %s", item.step, type(exc).__name__, exc
            )
            if self._on_error is not None:
                try:
                    self._on_error("metrics", exc)
                except Exception:  # noqa: BLE001
                    logger.debug("Error handler raised while reporting metrics", exc_info=True)

    def _flush_sinks(self) -> None:
        for sink in self.sinks:
            if not getattr(sink, "enabled", True):
                continue
            try:
                sink.flush()
            except Exception as exc:  # noqa: BLE001
                self._fail_sink(sink, exc)

    def _fail_sink(self, sink: Any, exc: Exception) -> None:
        """Disable a sink that raised, and report it exactly once.

        Not retried here: the transports already retry internally (traces on
        content-addressed uploads, the platform client on 429/5xx), so an error
        that reaches this point has already exhausted its budget. Continuing to
        call a sink in that state produces one log line per batch for the rest
        of the run and hides whatever failed first.
        """
        name = getattr(sink, "name", type(sink).__name__)
        sink.enabled = False
        logger.warning("Sink %s disabled after an error: %s: %s", name, type(exc).__name__, exc)
        if self._on_error is not None:
            try:
                self._on_error(name, exc)
            except Exception:  # noqa: BLE001 - the handler is the caller's problem
                logger.debug("Error handler raised while reporting a sink failure", exc_info=True)

    # ------------------------------------------------------------------ queue

    def submit(self, item: Any) -> bool:
        """Hand a batch or a metric point to the uploader.

        ``False`` means it was dropped: the queue stayed full for the whole
        timeout, so the producer is durably outrunning the uploader. Blocking
        further would turn a telemetry backlog into a stalled training run.
        """
        if self._stopping.is_set():
            return False
        thread = self._thread
        if thread is None or not thread.is_alive():
            self.start()
        try:
            self._queue.put(item, timeout=self.put_timeout)
            return True
        except queue.Full:
            count = len(item.records) if isinstance(item, WriteItem) else 1
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
        barrier = _Flush()
        try:
            self._queue.put(barrier, timeout=self.put_timeout)
        except queue.Full:
            logger.warning("Could not enqueue a flush barrier; the queue is saturated")
            return False
        return barrier.event.wait(timeout)

    def close(self, timeout: Optional[float] = 30.0) -> None:
        """Drain, stop the thread, and close every sink."""
        self._stopping.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            try:
                self._queue.put(None, timeout=self.put_timeout)
            except queue.Full:
                logger.warning("Upload queue saturated at close; some records may be lost")
            thread.join(timeout)
            if thread.is_alive():
                logger.warning("Uploader did not stop within %ss; abandoning it", timeout)
        self._thread = None
        for sink in self.sinks:
            try:
                sink.close()
            except Exception as exc:  # noqa: BLE001 - teardown must not raise
                logger.debug("Error closing sink %s: %s", getattr(sink, "name", sink), exc)

    # ------------------------------------------------------------------- fork

    def _register_fork_hook(self) -> None:
        if not hasattr(os, "register_at_fork"):  # pragma: no cover - Windows
            return
        os.register_at_fork(after_in_child=self._reinit_after_fork)

    def _reinit_after_fork(self) -> None:
        """Give the child a clean uploader.

        Everything queued at fork time belongs to the parent, which still has a
        live thread and will send it. Inheriting that queue would upload each
        record twice; inheriting the lock could deadlock the child on its first
        write.
        """
        self._pid = os.getpid()
        self._queue = queue.Queue(maxsize=self.max_queue_size)
        self._thread = None
        self._stopping = threading.Event()
        self._lock = threading.Lock()
