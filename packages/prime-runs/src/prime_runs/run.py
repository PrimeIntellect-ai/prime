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

from ._http import DEFAULT_TIMEOUT, UPLOAD_TIMEOUT, PlatformClient
from .backends import Backend, EvalsBackend, OfflineBackend
from .config import Config
from .exceptions import ConfigurationError, RunFinishedError
from .models import EnvironmentRef, Mode, OnError, RunHandle, RunKind, RunSpec, RunStatus
from .sinks import EvalSamplesSink, OfflineSink, Sink, TracesSink
from .worker import MetricItem, UploadWorker, WriteItem

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

        self._summary_flush_seconds = summary_flush_seconds
        self._finish_timeout = finish_timeout
        self._last_summary_flush = time.monotonic()
        self._summary_dirty = False
        self._config_dirty = False
        self._finish_lock = threading.RLock()
        self._finished = False

        sinks = sinks or []
        worker_kwargs: Dict[str, Any] = {}
        if queue_size is not None:
            worker_kwargs["max_queue_size"] = queue_size
        self._worker = UploadWorker(
            sinks,
            on_error=self._record_sink_error,
            metric_writer=self._write_metrics if backend.supports_step_metrics else None,
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
        self._previous_signal_handlers: Dict[int, Any] = {}

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
        return self._is_primary

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def dropped_records(self) -> int:
        """Records the uploader could not keep up with. Should be zero."""
        return self._worker.dropped

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
            return
        if self._backend.supports_step_metrics:
            self._worker.submit(MetricItem(metrics=cleaned, step=step))
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

        Safe to call from ``__exit__``, an atexit hook and a signal handler at
        once — whichever gets there first reports the status, and the rest
        return.
        """
        with self._finish_lock:
            if self._finished:
                return
            self._finished = True

        resolved = RunStatus(status) if not isinstance(status, RunStatus) else status
        if summary:
            self.summary.update(_clean_metrics(summary))
        self._status = resolved

        # Order matters: records first, so a dashboard that reacts to the
        # terminal status never sees a finished run with samples still landing.
        if not self._worker.flush(timeout=self._finish_timeout):
            logger.warning(
                "Run %s: uploads did not drain within %ss; finalizing anyway. "
                "Some records may be missing from this run.",
                self.id,
                self._finish_timeout,
            )
        self._worker.close(timeout=self._finish_timeout)

        if self._owns_lifecycle:
            self._report_guarded(
                "updating the run",
                lambda: self._backend.update(
                    self.id,
                    config=self.config if (self._config_dirty or self.config) else None,
                    summary=self.summary or None,
                ),
            )
            self._report_guarded(
                "finalizing the run",
                lambda: self._backend.finalize(
                    self.id,
                    status=resolved,
                    summary=self.summary or None,
                    error=error or (self.errors[0] if self.errors else None),
                    config=self.config or None,
                ),
            )
        self._report_guarded("closing the backend", self._backend.close)

        atexit.unregister(self._atexit_hook)
        self._restore_signal_handlers()
        self._retract_run_id()

        if self._worker.dropped:
            logger.warning(
                "Run %s finished with %d dropped record(s)", self.id, self._worker.dropped
            )

    def fail(self, error: Union[str, BaseException]) -> None:
        """Close the run out as failed."""
        self.finish(status=RunStatus.FAILED, error=_describe(error))

    def flush(self, timeout: Optional[float] = 30.0) -> bool:
        """Block until queued records have been written. Mostly for tests."""
        flushed = self._worker.flush(timeout=timeout)
        self._flush_summary()
        return flushed

    # -------------------------------------------------------- context manager

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if exc_type is None:
            self.finish()
        elif isinstance(exc, KeyboardInterrupt):
            # An interrupt is a decision, not a fault, so it must not land in
            # the same bucket as broken ones. Matches the SIGINT handler, which
            # normally gets there first when signal handling is on.
            self.finish(status=RunStatus.CRASHED, error="interrupted")
        else:
            self.finish(status=RunStatus.FAILED, error=_describe(exc))
        return False

    # ------------------------------------------------------------- internals

    def install_signal_handlers(self) -> None:
        """Report a terminal status when the process is killed.

        Only installed on the main thread, and only over a *default* handler:
        replacing a handler the application chose would be worse than missing a
        status. The previous handler is always called afterwards, so SIGINT
        still raises ``KeyboardInterrupt`` and SIGTERM still terminates.
        """
        if threading.current_thread() is not threading.main_thread():
            return
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                current = signal.getsignal(signum)
            except (ValueError, OSError):  # pragma: no cover - platform dependent
                continue
            if current not in (signal.SIG_DFL, signal.default_int_handler):
                continue
            try:
                signal.signal(signum, self._handle_signal)
            except (ValueError, OSError):  # pragma: no cover
                continue
            self._previous_signal_handlers[signum] = current

    def _handle_signal(self, signum: int, frame: Any) -> None:
        name = signal.Signals(signum).name
        # Read the displaced handler *before* finishing: finish() restores and
        # then clears this table, so looking it up afterwards always yields
        # SIG_DFL — which re-raises the signal at its default disposition and
        # kills the process instead of running the handler the app installed.
        previous = self._previous_signal_handlers.get(signum, signal.SIG_DFL)
        if not self._finished:
            # CRASHED, not FAILED: the producer never said the run failed, it was
            # stopped from outside its own control flow. Same bucket as the
            # atexit path, and deliberately not the bucket a broken eval lands in.
            self.finish(status=RunStatus.CRASHED, error=f"received {name}")
        signal.signal(signum, previous)
        if callable(previous):
            previous(signum, frame)
        else:
            os.kill(os.getpid(), signum)

    def _restore_signal_handlers(self) -> None:
        for signum, previous in self._previous_signal_handlers.items():
            try:
                if signal.getsignal(signum) is self._handle_signal:
                    signal.signal(signum, previous)
            except (ValueError, OSError):  # pragma: no cover
                continue
        self._previous_signal_handlers.clear()

    def _on_process_exit(self) -> None:
        """Last resort: the process is exiting and nobody called ``finish()``.

        Reported as CRASHED rather than FAILED — the producer never said the run
        failed, it just stopped existing, and the distinction tells an operator
        whether to read the run's error or go look at the machine.
        """
        if self._finished:
            return
        logger.warning("Run %s was never finished; reporting it as crashed", self.id)
        self.finish(status=RunStatus.CRASHED, error="process exited without finishing the run")

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
        if self._finished:
            raise RunFinishedError(
                f"{operation}() was called on run {self.id}, which is already finished. "
                "The platform has closed this run out; start a new one."
            )

    def _write_metrics(self, metrics: Dict[str, Any], step: Optional[int]) -> None:
        self._backend.log_metrics(self.id, metrics, step)

    def _maybe_flush_summary(self) -> None:
        now = time.monotonic()
        if now - self._last_summary_flush < self._summary_flush_seconds:
            return
        self._flush_summary()

    def _flush_summary(self) -> None:
        if not (self._summary_dirty or self._config_dirty) or not self._owns_lifecycle:
            return
        config = self.config if self._config_dirty else None
        summary = self.summary if self._summary_dirty else None
        self._last_summary_flush = time.monotonic()
        self._summary_dirty = False
        self._config_dirty = False
        self._report_guarded(
            "flushing run metrics",
            lambda: self._backend.update(self.id, config=config, summary=summary),
        )

    def _record_sink_error(self, sink_name: str, exc: Exception) -> None:
        self._report(f"writing to the {sink_name} sink", exc)

    def _report_guarded(self, what: str, call: Any) -> None:
        try:
            call()
        except Exception as exc:  # noqa: BLE001 - routed through the error policy
            self._report(what, exc)

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
            run_sinks.append(EvalSamplesSink(client))
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
    if mode == "online" and not is_primary and not run_id:
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
