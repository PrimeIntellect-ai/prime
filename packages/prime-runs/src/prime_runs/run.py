"""The run handle, and ``init()`` that produces one.

A run is a long-lived thing with a status, so it is an object, not three
stateless calls. That single change is what lets the SDK take on the work every
producer was doing privately: streaming instead of buffering, containing its
own errors, reporting a terminal status when the process dies, and behaving the
same on rank 3 of a training job as on a laptop.

The identity rule matters most, so it is worth stating once. ``init()`` is
called *before* rollouts start, and the ID it returns is *the* run ID
everywhere — including inside every trace document the producer writes, and
including the local archive. Nothing is re-stamped afterwards and no producer
record is rewritten. Verifiers already stamps ``EvalRunInfo(id=config.run.id)``
at rollout time; the only change is where that ID comes from. Offline runs get
a locally issued ID through the same path, so there is one code path, not two.
"""

import atexit
import logging
import math
import os
import signal
import threading
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from . import _fork
from ._http import DEFAULT_TIMEOUT, UPLOAD_TIMEOUT, PlatformClient
from .backends import Backend, EvalsBackend, OfflineBackend
from .config import Config
from .exceptions import ConfigurationError, RunFinishedError
from .models import EnvironmentRef, Mode, OnError, RunHandle, RunKind, RunSpec, RunStatus
from .sinks import EvalSamplesSink, OfflineSink, Sink, TracesSink
from .worker import MetricItem, RunUpdateItem, UploadWorker, WriteItem

logger = logging.getLogger(__name__)

RUN_ID_ENV = "PRIME_RUN_ID"
MODE_ENV = "PRIME_RUNS_MODE"
#: Rank variables, in the order prime-rl sets them. Rank 0 owns the lifecycle.
RANK_ENV_VARS = ("RANK", "DP_RANK", "LOCAL_RANK")
DEFAULT_SUMMARY_FLUSH_SECONDS = 10.0
#: How long ``finish()`` waits for queued records. Derived from the upload
#: timeout rather than picked: a single in-flight sample POST is allowed 300s,
#: so a shorter budget here would routinely abandon an upload that was about to
#: succeed and then finalize the run without it.
DEFAULT_FINISH_TIMEOUT = float(UPLOAD_TIMEOUT.read or 300.0)

#: Run IDs this process exported into ``PRIME_RUN_ID``, mapped to the PID that
#: exported them. The PID is the whole point: it is what distinguishes "my
#: parent opened this run and I should join it" from "I opened this run a moment
#: ago and the variable is still lying around". A forked child sees a different
#: PID and correctly treats the entry as inherited.
_exported_run_ids: Dict[str, int] = {}


class Run:
    """A live run: an ID, a URL, somewhere to put metrics, somewhere to put traces.

    Every method that touches the network is contained. With the default
    ``on_error="warn"`` nothing raised by the platform escapes into a producer's
    loop — a run that has been going for six hours does not get killed by a 502
    on a telemetry call. ``on_error="raise"`` inverts that for tests and CI,
    where a silent upload failure is the bug.
    """

    def __init__(
        self,
        *,
        backend: Backend,
        handle: RunHandle,
        spec: RunSpec,
        sinks: Optional[List[Sink]] = None,
        mode: Mode = "online",
        on_error: OnError = "warn",
        is_primary: bool = True,
        owns_lifecycle: bool = True,
        summary_flush_seconds: float = DEFAULT_SUMMARY_FLUSH_SECONDS,
        finish_timeout: float = DEFAULT_FINISH_TIMEOUT,
        queue_size: Optional[int] = None,
    ) -> None:
        self._backend = backend
        self._handle = handle
        self._spec = spec
        self._mode: Mode = mode
        self._on_error: OnError = on_error
        self._is_primary = is_primary
        # A non-primary rank shares the run but must not create or close it:
        # eight ranks racing to finalize produce seven confusing failures and
        # one winner.
        self._owns_lifecycle = owns_lifecycle and is_primary
        self._status = RunStatus.RUNNING

        self.config: Dict[str, Any] = dict(spec.config)
        self.summary: Dict[str, Any] = dict(spec.summary)
        self.errors: List[str] = []
        # Raised at the next synchronization point the caller controls. A sink
        # fails on the uploader thread, where raising reaches nobody — so under
        # on_error="raise" the exception is held and re-raised from flush() or
        # finish(), which is where a test or a CI job is actually looking.
        self._deferred_error: Optional[BaseException] = None

        self._summary_flush_seconds = summary_flush_seconds
        self._finish_timeout = finish_timeout
        self._last_summary_flush = time.monotonic()
        self._summary_dirty = False
        self._config_dirty = False
        self._pending_metrics: Dict[str, Any] = {}
        self._pending_metric_step: Optional[int] = None
        self._finish_lock = threading.RLock()
        self._finish_condition = threading.Condition(self._finish_lock)
        self._finishing = False
        self._finishing_thread_id: Optional[int] = None
        self._finished = False
        # A Python signal handler can interrupt finish() on the same thread.
        # Re-entering teardown would duplicate finalization, while chaining the
        # signal immediately would kill the process before teardown completes.
        # Keep the first such signal and deliver it once the run is closed.
        self._pending_signal: Optional[tuple[int, Any, Any]] = None

        sinks = sinks or []
        worker_kwargs: Dict[str, Any] = {}
        if queue_size is not None:
            worker_kwargs["max_queue_size"] = queue_size
        self._worker = UploadWorker(
            sinks,
            on_error=self._record_sink_error,
            metric_writer=self._write_metrics if backend.supports_step_metrics else None,
            update_writer=self._write_run_update,
            **worker_kwargs,
        )
        context = _sink_context(spec, handle)
        for sink in sinks:
            try:
                sink.start(handle.id, context)
            except Exception as exc:  # noqa: BLE001 - a bad sink is not a bad run
                sink.enabled = False
                self._report(f"starting sink {getattr(sink, 'name', sink)}", exc)

        self._atexit_hook = self._on_process_exit
        atexit.register(self._atexit_hook)
        # Bound once and kept. ``self._handle_signal`` builds a *new* bound
        # method on every attribute access, so an ``is`` comparison against a
        # freshly-made one is always False — which is how handlers end up
        # installed forever, pinning a finished run and blocking the next run in
        # the process from installing its own.
        self._signal_handler = self._handle_signal
        self._previous_signal_handlers: Dict[int, Any] = {}
        # ``signal.signal`` can only run on the main thread. If finish() runs in
        # an executor, or this object is inherited across a fork, its handler
        # may remain as the process disposition until the main thread gets a
        # chance to replace it. Marking forked handlers as relinquishable lets
        # a child run take ownership without mistaking the inherited callback
        # for an application-installed handler.
        self._signal_handler_stale = False
        _fork.register(self)

    # -------------------------------------------------------------- identity

    @property
    def id(self) -> str:
        """The run ID. Stamp this onto every trace the run produces."""
        return self._handle.id

    @property
    def name(self) -> Optional[str]:
        return self._handle.name

    @property
    def url(self) -> Optional[str]:
        """Where to open this run — a dashboard URL, or a local path offline."""
        return self._handle.url

    @property
    def kind(self) -> RunKind:
        return self._spec.kind

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def status(self) -> RunStatus:
        return self._status

    @property
    def is_primary(self) -> bool:
        """Whether this process owns the run's lifecycle (rank 0, or single-process)."""
        return self._owns_lifecycle

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def dropped_records(self) -> int:
        """Records that reached no sink because the queue was full.

        Backpressure only: the producer durably outran the uploader. A record
        counted here was stored nowhere. Contrast ``failed_records``, which is
        per-sink and usually means the record is still safe in another sink.
        """
        return self._worker.dropped

    @property
    def failed_records(self) -> Dict[str, int]:
        """Records each sink could not store, by sink name.

        Not summed into one number and not merged into ``dropped_records``: with
        traces and the sample table both enabled, the same batch failing on one
        sink says nothing about whether the other stored it, so a single total
        would report data missing that is not actually gone.
        """
        return dict(self._worker.failed_records)

    def __repr__(self) -> str:
        return (
            f"<Run id={self.id!r} kind={self.kind!r} "
            f"mode={self._mode!r} status={self._status.value}>"
        )

    # ------------------------------------------------------------------- log

    def log(
        self,
        metrics: Mapping[str, Any],
        *,
        step: Optional[int] = None,
        commit: bool = True,
    ) -> None:
        """Record scalar metrics.

        Values always land in ``summary`` last-value-wins, the way W&B's
        implicit summary works. Whether they *also* become a time series
        depends on the backend: the training API stores one, the evaluations
        API stores a single metrics blob, and rather than making producers care,
        a backend without a time series simply keeps the summary — flushed on a
        timer so a tight loop does not turn into one PUT per step.

        ``commit=False`` stages values without scheduling a write, for callers
        assembling a step from several places.
        """
        self._require_live("log")
        cleaned = _clean_metrics(metrics)
        if not cleaned:
            return
        self.summary.update(cleaned)
        self._summary_dirty = True
        if not commit:
            if self._backend.supports_step_metrics:
                self._pending_metrics.update(cleaned)
                if step is not None:
                    self._pending_metric_step = step
            return
        if self._backend.supports_step_metrics:
            committed = {**self._pending_metrics, **cleaned}
            committed_step = step if step is not None else self._pending_metric_step
            self._pending_metrics.clear()
            self._pending_metric_step = None
            self._worker.submit(MetricItem(metrics=committed, step=committed_step))
            return
        self._maybe_flush_summary()

    def log_traces(
        self,
        records: Iterable[Any],
        *,
        line_format: Optional[str] = None,
        step: Optional[int] = None,
    ) -> None:
        """Hand traces or episodes to the sinks. Returns immediately.

        Accepts whatever the producer already has: verifiers ``Trace`` /
        ``Episode``, prime-rl ``Rollout``, or plain JSON mappings. Nothing is
        buffered until the end of the run — call this as rollouts complete and
        the dashboard fills in while the run is still going.
        """
        self._require_live("log_traces")
        batch = list(records)
        if not batch:
            return
        self._worker.submit(WriteItem(records=batch, line_format=line_format, step=step))

    # Producers that think in episodes rather than traces; same path.
    log_episodes = log_traces

    def log_samples(self, records: Iterable[Any], *, step: Optional[int] = None) -> None:
        """Alias for :meth:`log_traces`, matching prime-rl's ``Monitor`` vocabulary."""
        self.log_traces(records, step=step)

    def update_config(self, values: Mapping[str, Any]) -> None:
        """Merge into the run's config (its inputs). Flushed with the summary."""
        self._require_live("update_config")
        self.config.update(values)
        self._config_dirty = True

    # ---------------------------------------------------------------- finish

    def finish(
        self,
        summary: Optional[Mapping[str, Any]] = None,
        *,
        status: Union[RunStatus, str] = RunStatus.COMPLETED,
        error: Optional[str] = None,
    ) -> None:
        """Flush everything and close the run out. Idempotent.

        ``status`` must be one of the terminal :class:`RunStatus` values.
        Safe to call from ``__exit__``, an atexit hook and a signal handler at
        once — whichever gets there first reports the status, and the rest wait
        for that teardown to complete.
        """
        thread_id = threading.get_ident()
        with self._finish_condition:
            while self._finishing and not self._finished:
                # A signal handler can interrupt this very finish() call. It
                # cannot wait for itself, so _handle_signal defers chaining the
                # signal and this nested call simply yields to the active one.
                if self._finishing_thread_id == thread_id:
                    return
                self._finish_condition.wait()
            if self._finished:
                return
            resolved = RunStatus(status) if not isinstance(status, RunStatus) else status
            if not resolved.is_terminal():
                raise ValueError(f"finish() requires a terminal status, got {resolved.value!r}")
            self._finishing = True
            self._finishing_thread_id = thread_id

        try:
            self._finish_once(summary, resolved, error)
        finally:
            with self._finish_condition:
                self._finishing = False
                self._finishing_thread_id = None
                self._finished = True
                pending_signal = self._pending_signal
                self._pending_signal = None
                self._finish_condition.notify_all()

            if pending_signal is not None:
                self._chain_signal(*pending_signal)

    def _finish_once(
        self,
        summary: Optional[Mapping[str, Any]],
        resolved: RunStatus,
        error: Optional[str],
    ) -> None:
        """Perform the single teardown owned by the first ``finish()`` caller."""

        if summary:
            self.summary.update(_clean_metrics(summary))
        self._status = resolved

        finish_error: Optional[BaseException] = None
        deadline = time.monotonic() + max(0.0, self._finish_timeout)

        def remaining_finish_time() -> float:
            return max(0.0, deadline - time.monotonic())

        # Order matters: records first, so a dashboard that reacts to the
        # terminal status never sees a finished run with samples still landing.
        if not self._worker.flush(timeout=remaining_finish_time()):
            logger.warning(
                "Run %s: uploads did not drain within %ss; finalizing anyway. "
                "Some records may be missing from this run.",
                self.id,
                self._finish_timeout,
            )
        self._worker.close(timeout=remaining_finish_time())

        if self._owns_lifecycle:
            finish_error = self._finish_guarded(
                "updating the run",
                lambda: self._backend.update(
                    self.id,
                    config=self.config if (self._config_dirty or self.config) else None,
                    summary=self.summary or None,
                ),
                finish_error,
            )
            finish_error = self._finish_guarded(
                "finalizing the run",
                lambda: self._backend.finalize(
                    self.id,
                    status=resolved,
                    summary=self.summary or None,
                    error=error or (self.errors[0] if self.errors else None),
                    config=self.config or None,
                ),
                finish_error,
            )
        finish_error = self._finish_guarded(
            "closing the backend", self._backend.close, finish_error
        )

        atexit.unregister(self._atexit_hook)
        self._restore_signal_handlers()
        self._retract_run_id()

        if self._worker.dropped:
            logger.warning(
                "Run %s finished with %d record(s) that reached no sink; the producer "
                "outran the uploader.",
                self.id,
                self._worker.dropped,
            )
        for sink_name, count in self._worker.failed_records.items():
            # Deliberately phrased per sink: another sink may hold these records,
            # so claiming they are missing from the run would overstate the loss.
            logger.warning(
                "Run %s: the %s sink could not store %d record(s)", self.id, sink_name, count
            )
        # Last, so a run that failed to upload is still closed out properly
        # before the failure reaches the caller.
        if finish_error is not None:
            raise finish_error
        self._raise_deferred()

    def fail(self, error: Union[str, BaseException]) -> None:
        """Close the run out as failed."""
        self.finish(status=RunStatus.FAILED, error=_describe(error))

    def flush(self, timeout: Optional[float] = 30.0) -> bool:
        """Block until queued records have been written.

        Under ``on_error="raise"`` this is the first place an upload failure can
        surface, since the failure itself happened on the uploader thread.
        """
        update_queued = self._queue_run_update()
        flushed = self._worker.flush(timeout=timeout)
        # A periodic update can lose a race for the last queue slot. Once the
        # barrier drains that backlog, give the still-dirty snapshot one more
        # chance so an explicit flush keeps its persistence guarantee.
        if flushed and not update_queued and self._queue_run_update():
            flushed = self._worker.flush(timeout=timeout)
        self._raise_deferred()
        return flushed

    # -------------------------------------------------------- context manager

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if exc_type is None:
            self.finish()
            return False

        if isinstance(exc, KeyboardInterrupt):
            # An interrupt is a decision, not a fault, so it must not land in
            # the same bucket as broken ones. Matches the SIGINT handler, which
            # normally gets there first when signal handling is on.
            status = RunStatus.CRASHED
            error = "interrupted"
        else:
            status = RunStatus.FAILED
            error = _describe(exc)

        try:
            self.finish(status=status, error=error)
        except Exception as finish_error:
            # The producer exception is the reason this context is unwinding.
            # A telemetry teardown error must not replace it, even in strict
            # mode; finish() has already recorded the failure on the run.
            # Control-flow exceptions such as KeyboardInterrupt and SystemExit
            # deliberately bypass this handler so teardown cannot swallow them.
            logger.warning(
                "Run %s: finishing after %s also failed: %s: %s",
                self.id,
                exc_type.__name__,
                type(finish_error).__name__,
                finish_error,
                exc_info=True,
            )
        return False

    # ------------------------------------------------------------- internals

    def install_signal_handlers(self) -> None:
        """Report a terminal status when the process is killed.

        Only installed on the main thread, and only over a *default* handler or
        one relinquished by a finished/forked ``Run``. Replacing a handler the
        application chose would be worse than missing a status. The previous
        handler is always called afterwards, so SIGINT still raises
        ``KeyboardInterrupt`` and SIGTERM still terminates.
        """
        if threading.current_thread() is not threading.main_thread():
            return
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                current = signal.getsignal(signum)
            except (ValueError, OSError):  # pragma: no cover - platform dependent
                continue
            previous = current
            relinquished_owner: Optional[Run] = None
            if current not in (signal.SIG_DFL, signal.default_int_handler):
                owner = getattr(current, "__self__", None)
                if (
                    isinstance(owner, Run)
                    and current is owner._signal_handler
                    and (owner._finished or owner._signal_handler_stale)
                    and signum in owner._previous_signal_handlers
                ):
                    previous = owner._previous_signal_handlers[signum]
                    relinquished_owner = owner
                else:
                    continue
            try:
                signal.signal(signum, self._signal_handler)
            except (ValueError, OSError):  # pragma: no cover
                continue
            self._previous_signal_handlers[signum] = previous
            if relinquished_owner is not None:
                relinquished_owner._previous_signal_handlers.pop(signum, None)

    def _handle_signal(self, signum: int, frame: Any) -> None:
        name = signal.Signals(signum).name
        # Read the displaced handler *before* finishing: finish() restores and
        # then clears this table, so looking it up afterwards always yields
        # SIG_DFL — which re-raises the signal at its default disposition and
        # kills the process instead of running the handler the app installed.
        previous = self._previous_signal_handlers.get(signum, signal.SIG_DFL)
        with self._finish_condition:
            if self._finishing and self._finishing_thread_id == threading.get_ident():
                if self._pending_signal is None:
                    self._pending_signal = (signum, frame, previous)
                return
            finished = self._finished
        if not finished:
            # CRASHED, not FAILED: the producer never said the run failed, it was
            # stopped from outside its own control flow. Same bucket as the
            # atexit path, and deliberately not the bucket a broken eval lands in.
            try:
                self.finish(status=RunStatus.CRASHED, error=f"received {name}")
            except Exception as exc:  # noqa: BLE001 - the signal must still chain
                logger.warning("Run %s: reporting %s failed: %s", self.id, name, exc)
        self._chain_signal(signum, frame, previous)

    @staticmethod
    def _chain_signal(signum: int, frame: Any, previous: Any) -> None:
        """Restore and invoke the handler displaced by this run."""
        signal.signal(signum, previous)
        if callable(previous):
            previous(signum, frame)
        else:
            os.kill(os.getpid(), signum)

    def _restore_signal_handlers(self) -> None:
        remaining: Dict[int, Any] = {}
        for signum, previous in self._previous_signal_handlers.items():
            try:
                if signal.getsignal(signum) is self._signal_handler:
                    signal.signal(signum, previous)
            except (ValueError, OSError):  # pragma: no cover
                # Most commonly finish() was deliberately run in an executor.
                # Keep the displaced handler so the main thread can restore it
                # from the signal callback or hand it to the next Run.
                remaining[signum] = previous
        self._previous_signal_handlers = remaining

    def reset_after_fork(self) -> None:
        """Make an inherited handle safe to use in a forked child.

        The child may keep using this handle, but the process that created the
        run remains responsible for its lifecycle. In particular, the child's
        inherited signal and atexit callbacks must never finalize the parent's
        still-running run. The lock also has to be replaced because it may have
        been owned at fork time by a thread that no longer exists.
        """
        self._finish_lock = threading.RLock()
        self._finish_condition = threading.Condition(self._finish_lock)
        self._finishing = False
        self._finishing_thread_id = None
        self._pending_signal = None
        self._is_primary = False
        self._owns_lifecycle = False
        self._deferred_error = None
        self._signal_handler_stale = True

    def _on_process_exit(self) -> None:
        """Last resort: the process is exiting and nobody called ``finish()``.

        Reported as CRASHED rather than FAILED — the producer never said the run
        failed, it just stopped existing, and the distinction tells an operator
        whether to read the run's error or go look at the machine.
        """
        if self._finished:
            return
        logger.warning("Run %s was never finished; reporting it as crashed", self.id)
        # on_error="raise" must not turn interpreter shutdown into a traceback
        # from atexit; the run is already being reported as crashed.
        try:
            self.finish(status=RunStatus.CRASHED, error="process exited without finishing the run")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Run %s: reporting the crash failed: %s", self.id, exc)

    def _retract_run_id(self) -> None:
        """Stop advertising a finished run to processes started from here.

        Only retracts what this run published: if the value now points somewhere
        else, another run owns it and clearing it would orphan that one's
        children.
        """
        if not self._owns_lifecycle:
            return
        if os.environ.get(RUN_ID_ENV) == self.id:
            os.environ.pop(RUN_ID_ENV, None)
        _exported_run_ids.pop(self.id, None)

    def _require_live(self, operation: str) -> None:
        if self._finishing or self._finished:
            raise RunFinishedError(
                f"{operation}() was called on run {self.id}, which is already finished. "
                "The platform has closed this run out; start a new one."
            )

    def _write_metrics(self, metrics: Dict[str, Any], step: Optional[int]) -> None:
        self._backend.log_metrics(self.id, metrics, step)

    def _write_run_update(
        self,
        config: Optional[Dict[str, Any]],
        summary: Optional[Dict[str, Any]],
    ) -> None:
        self._backend.update(self.id, config=config, summary=summary)

    def _maybe_flush_summary(self) -> None:
        now = time.monotonic()
        if now - self._last_summary_flush < self._summary_flush_seconds:
            return
        self._queue_run_update()

    def _queue_run_update(self) -> bool:
        if not (self._summary_dirty or self._config_dirty) or not self._owns_lifecycle:
            return True
        config_dirty = self._config_dirty
        summary_dirty = self._summary_dirty
        item = RunUpdateItem(
            config=dict(self.config) if config_dirty else None,
            summary=dict(self.summary) if summary_dirty else None,
        )
        # Clear before the potentially blocking queue put. If another producer
        # thread logs while this one waits, its new dirty bit must survive.
        if config_dirty:
            self._config_dirty = False
        if summary_dirty:
            self._summary_dirty = False
        if not self._worker.submit(item):
            self._config_dirty = self._config_dirty or config_dirty
            self._summary_dirty = self._summary_dirty or summary_dirty
            return False
        self._last_summary_flush = time.monotonic()
        return True

    def _record_sink_error(self, sink_name: str, exc: Exception) -> None:
        """Called on the uploader thread when a sink gives up."""
        message = f"writing to the {sink_name} sink failed: {type(exc).__name__}: {exc}"
        self.errors.append(message)
        if self._on_error == "raise":
            if self._deferred_error is None:
                self._deferred_error = exc
            return
        logger.warning("Run %s: %s", self._handle.id, message)

    def _raise_deferred(self) -> None:
        """Re-raise the first upload failure, once."""
        exc = self._deferred_error
        if exc is None:
            return
        self._deferred_error = None
        raise exc

    def _finish_guarded(
        self,
        what: str,
        call: Any,
        first_error: Optional[BaseException],
    ) -> Optional[BaseException]:
        """Run one teardown step without letting it skip the steps after it."""
        try:
            call()
        except Exception as exc:  # noqa: BLE001 - teardown must continue
            message = f"{what} failed: {type(exc).__name__}: {exc}"
            self.errors.append(message)
            if self._on_error == "raise":
                return first_error or exc
            logger.warning("Run %s: %s", self._handle.id, message)
        return first_error

    def _report(self, what: str, exc: Exception) -> None:
        message = f"{what} failed: {type(exc).__name__}: {exc}"
        self.errors.append(message)
        if self._on_error == "raise":
            raise exc
        logger.warning("Run %s: %s", self._handle.id, message)


# --------------------------------------------------------------------- init


def init(
    *,
    name: Optional[str] = None,
    kind: RunKind = "eval",
    environments: Optional[Sequence[Any]] = None,
    model: Optional[str] = None,
    framework: Optional[str] = None,
    dataset: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    config: Optional[Mapping[str, Any]] = None,
    summary: Optional[Mapping[str, Any]] = None,
    id: Optional[str] = None,
    mode: Optional[Mode] = None,
    dir: Optional[str] = None,
    team_id: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    traces_url: Optional[str] = None,
    traces: bool = True,
    samples: bool = True,
    sinks: Optional[List[Sink]] = None,
    on_error: OnError = "warn",
    handle_signals: bool = True,
    queue_size: Optional[int] = None,
) -> Run:
    """Start a run and return a handle to it.

    Call this *before* the first rollout: the ID it returns is what every trace
    in the run should carry, and the URL it returns is what a producer prints so
    someone can watch the run fill in.

    ``mode`` selects where the run lives. Left unset it is read from
    ``$PRIME_RUNS_MODE``, and failing that inferred: online when there is an API
    key, offline when there is not. Offline is a real run with a real ID and a
    real directory, which is why producers no longer need a ``--no-push``
    branch — the call sites are identical either way.

    ``id`` attaches to an existing run instead of creating one, for resuming
    after a crash and for non-primary ranks joining a run rank 0 created.
    """
    resolved_config = Config()
    api_key = api_key if api_key is not None else resolved_config.api_key
    base_url = base_url or resolved_config.base_url
    team_id = team_id if team_id is not None else resolved_config.team_id

    spec = RunSpec(
        name=name,
        kind=kind,
        environments=[EnvironmentRef.coerce(entry) for entry in (environments or [])],
        model=model,
        framework=framework,
        dataset=dataset,
        description=description,
        tags=list(tags or []),
        team_id=team_id,
        config=dict(config or {}),
        summary=dict(summary or {}),
    )

    is_primary = _is_primary_rank()
    joined_id = _inherited_run_id()
    inherited_id = id or joined_id
    # Owning the lifecycle means "this call is responsible for creating and
    # finalizing the run". An explicit `id=` is a deliberate resume, so it owns.
    # An ID picked up from the environment belongs to whoever exported it, so it
    # does not. (A non-primary rank never owns either way — see Run.__init__.)
    owns_lifecycle = id is not None or joined_id is None
    resolved_mode = _resolve_mode(mode, api_key=api_key, is_primary=is_primary, run_id=inherited_id)

    if resolved_mode == "disabled":
        backend: Backend = _DisabledBackend()
        handle = RunHandle(id=inherited_id or _local_id(), name=name)
        return _build(
            spec, backend, handle, [], resolved_mode, on_error, is_primary, False, queue_size
        )

    if resolved_mode == "offline":
        offline = OfflineBackend(dir)
        handle = offline.attach(inherited_id) if inherited_id else offline.create(spec)
        run_sinks = sinks if sinks is not None else [OfflineSink(offline.directory)]
        run = _build(
            spec,
            offline,
            handle,
            run_sinks,
            resolved_mode,
            on_error,
            is_primary,
            owns_lifecycle,
            queue_size,
        )
        _announce(run, handle_signals)
        return run

    if kind != "eval":
        raise ConfigurationError(
            f"kind={kind!r} is not supported yet — training runs arrive with the RFT backend. "
            'Use kind="eval", or mode="offline" to record the run locally.'
        )
    if not api_key:
        raise ConfigurationError(
            'mode="online" needs an API key. Set PRIME_API_KEY, run `prime login`, '
            'or pass mode="offline".'
        )

    client = PlatformClient(api_key=api_key, base_url=base_url, timeout=DEFAULT_TIMEOUT)
    backend = EvalsBackend(client, frontend_url=resolved_config.frontend_url, team_id=team_id)
    handle = backend.attach(inherited_id) if inherited_id else backend.create(spec)

    if sinks is None:
        run_sinks = []
        if traces:
            run_sinks.append(TracesSink(api_key=api_key, traces_url=traces_url, team_id=team_id))
        if samples:
            # Both transports run during the transition: traces is the system of
            # record, the sample table is what today's viewer reads, and Prime
            # Traces is still gated to an account allowlist.
            samples_client = PlatformClient(
                api_key=api_key, base_url=base_url, timeout=DEFAULT_TIMEOUT
            )
            run_sinks.append(EvalSamplesSink(samples_client, close_client=True))
    else:
        run_sinks = list(sinks)

    run = _build(
        spec,
        backend,
        handle,
        run_sinks,
        resolved_mode,
        on_error,
        is_primary,
        owns_lifecycle,
        queue_size,
    )
    _announce(run, handle_signals)
    return run


def _build(
    spec: RunSpec,
    backend: Backend,
    handle: RunHandle,
    sinks: List[Sink],
    mode: Mode,
    on_error: OnError,
    is_primary: bool,
    owns_lifecycle: bool,
    queue_size: Optional[int],
) -> Run:
    return Run(
        backend=backend,
        handle=handle,
        spec=spec,
        sinks=sinks,
        mode=mode,
        on_error=on_error,
        is_primary=is_primary,
        owns_lifecycle=owns_lifecycle,
        queue_size=queue_size,
    )


def _announce(run: Run, handle_signals: bool) -> None:
    """Publish the run ID to child processes and arm crash reporting.

    Exporting ``PRIME_RUN_ID`` is how forked workers and subprocess launchers
    join the run their parent created instead of each opening their own — the
    same trick prime-rl's monitor used with ``RUN_ID``, generalized so every
    producer gets it. The PID is recorded alongside so that *this* process does
    not later mistake its own export for a parent's.
    """
    os.environ[RUN_ID_ENV] = run.id
    _exported_run_ids[run.id] = os.getpid()
    if handle_signals:
        run.install_signal_handlers()
    if run.url:
        logger.info("Run %s: %s", run.id, run.url)


def _inherited_run_id() -> Optional[str]:
    """A run ID this process should join, or ``None`` to open a fresh run.

    ``PRIME_RUN_ID`` set by an ancestor means "join that run". The same variable
    set by an earlier ``init()`` *in this process* means nothing of the sort —
    without this check, a second eval in one process would silently attach to
    the first, and would never create or finalize a run of its own.
    """
    value = os.getenv(RUN_ID_ENV)
    if not value:
        return None
    if _exported_run_ids.get(value) == os.getpid():
        return None
    return value


class _DisabledBackend:
    """No-op lifecycle, so ``mode="disabled"`` needs no branching upstream."""

    kind = "disabled"
    supports_step_metrics = False

    def create(self, spec: RunSpec) -> RunHandle:
        return RunHandle(id=_local_id())

    def attach(self, run_id: str) -> RunHandle:
        return RunHandle(id=run_id)

    def update(self, run_id: str, **kwargs: Any) -> None:
        return None

    def log_metrics(self, run_id: str, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        return None

    def finalize(self, run_id: str, **kwargs: Any) -> None:
        return None

    def close(self) -> None:
        return None


def _local_id() -> str:
    from .backends.offline import new_run_id

    return new_run_id()


def _is_primary_rank() -> bool:
    """Whether this process should own the run's lifecycle.

    Any rank variable set to something other than 0 means a peer process is
    rank 0 and owns creation and finalization. Non-primary ranks still upload
    their own records — the point is that eight processes contribute to one run
    rather than creating eight.
    """
    for name in RANK_ENV_VARS:
        value = os.getenv(name)
        if value and value.strip() not in ("0", ""):
            return False
    return True


def _resolve_mode(
    mode: Optional[Mode], *, api_key: str, is_primary: bool, run_id: Optional[str]
) -> Mode:
    if mode is None:
        env_mode = os.getenv(MODE_ENV)
        if env_mode:
            mode = env_mode.strip().lower()  # type: ignore[assignment]
    if mode not in (None, "online", "offline", "disabled"):
        raise ConfigurationError(f"mode={mode!r} is not one of 'online', 'offline' or 'disabled'")
    if mode is None:
        if api_key:
            mode = "online"
        else:
            logger.warning(
                "No API key found (set PRIME_API_KEY or run `prime login`); "
                "recording this run offline instead."
            )
            mode = "offline"
    if mode in ("online", "offline") and not is_primary and not run_id:
        # A non-primary rank with no run to join would create a second run for
        # the same job. Recording nothing is better than that.
        logger.debug("Non-primary rank with no %s; disabling this run handle", RUN_ID_ENV)
        return "disabled"
    return mode  # type: ignore[return-value]


def _sink_context(spec: RunSpec, handle: RunHandle) -> Dict[str, str]:
    """Upload-scoped provenance.

    Not the join key — that is ``run.id`` inside the trace document, which the
    ingestion service extracts into an indexed column. What goes here is what
    you would want when looking at an upload and asking where it came from.
    """
    context = {"source": "prime-runs", "run_kind": spec.kind}
    if spec.framework:
        context["framework"] = spec.framework
    if spec.model:
        context["model"] = spec.model
    return context


def _describe(error: Union[str, BaseException]) -> str:
    if isinstance(error, BaseException):
        return f"{type(error).__name__}: {error}"
    return str(error)


def _clean_metrics(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """Drop values JSON cannot carry.

    NaN and infinity are the ones that matter: a diverged loss serializes as
    JavaScript's bare ``NaN``, which strict JSON rejects, and the failure
    surfaces as an opaque 400 on a payload nobody can inspect. Dropping the key
    loses one point; sending it loses the request.
    """
    cleaned: Dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, float) and not math.isfinite(value):
            logger.debug("Dropping non-finite metric %s=%r", key, value)
            continue
        if isinstance(value, Mapping):
            nested = _clean_metrics(value)
            if nested:
                cleaned[key] = nested
            continue
        cleaned[key] = value
    return cleaned


__all__ = ["Run", "init", "RUN_ID_ENV", "MODE_ENV"]
