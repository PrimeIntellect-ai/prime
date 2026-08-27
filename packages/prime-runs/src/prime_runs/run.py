"""The run handle, and ``init()`` that produces one.

A run is a long-lived thing with a status, so it is an object rather than three
stateless calls: records stream instead of buffering, errors are contained, and
a process that dies still reports a terminal status.

``init()`` is called *before* rollouts start, and the ID it returns is *the* run
ID everywhere — including inside every trace document the producer writes.
Nothing is re-stamped afterwards.
"""

import atexit
import logging
import math
import os
import threading
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from . import _fork
from ._http import DEFAULT_TIMEOUT, UPLOAD_TIMEOUT, PlatformClient
from .backend import Backend, DisabledBackend, EvalsBackend, disabled_run_id
from .config import Config
from .exceptions import ConfigurationError, RunFinishedError
from .models import (
    CONFIG_SOURCE_KEY,
    RUN_KIND,
    ConfigSource,
    EnvironmentRef,
    Mode,
    OnError,
    RunHandle,
    RunSpec,
    RunStatus,
)
from .sinks import EvalSamplesSink, Sink, TracesSink
from .worker import UploadWorker

logger = logging.getLogger(__name__)

MODE_ENV = "PRIME_RUNS_MODE"
#: How long ``finish()`` gives queued uploads to drain before finalizing anyway.
#: Derived from the upload timeout: a single in-flight sample POST may take
#: this long, and a shorter budget would abandon an upload about to succeed.
DEFAULT_FINISH_TIMEOUT = float(UPLOAD_TIMEOUT.read or 300.0)


class Run:
    """A live run: an ID, a URL, somewhere to put traces, a summary.

    With the default ``on_error="warn"`` nothing raised by the platform escapes
    into a producer's loop; ``on_error="raise"`` surfaces the first failure
    from :meth:`flush` or :meth:`finish`, for tests and CI.
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
    ) -> None:
        self._backend = backend
        self._handle = handle
        self._spec = spec
        self._mode: Mode = mode
        self._on_error: OnError = on_error
        self._status = RunStatus.RUNNING
        # A forked child inherits this handle but must not close the parent's run.
        self._owns_lifecycle = True

        self.config: Dict[str, Any] = dict(spec.config)
        self.summary: Dict[str, Any] = {}
        self.errors: List[str] = []
        # Under on_error="raise", a failure on the uploader thread is held here
        # and re-raised from flush() or finish(), where the caller is looking.
        self._deferred_error: Optional[BaseException] = None

        self._finish_timeout = DEFAULT_FINISH_TIMEOUT
        self._finish_lock = threading.Lock()
        self._finishing = False
        self._finished = False
        self._atexit_hook = self._on_process_exit
        _fork.register(self)

        sinks = sinks or []
        self._worker = UploadWorker(sinks, on_error=self._record_sink_error)
        context = _sink_context(spec)
        for sink in sinks:
            try:
                sink.start(handle.id, context)
            except Exception as exc:  # noqa: BLE001 - a bad sink is not a bad run
                sink.enabled = False
                self._note(f"starting sink {sink.name}", exc)
                if self._on_error == "raise":
                    # The backend may already have created a remote run; close
                    # it out before the failure reaches the caller. finish()
                    # re-raises the error noted above.
                    self.finish(status=RunStatus.FAILED, error=_describe(exc))
                    raise exc

        atexit.register(self._atexit_hook)

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
        """The dashboard URL; ``None`` when disabled."""
        return self._handle.url

    @property
    def config_source(self) -> Optional[ConfigSource]:
        """The config file this run was launched from, if one was given."""
        raw = self.config.get(CONFIG_SOURCE_KEY)
        return ConfigSource.from_mapping(raw) if isinstance(raw, Mapping) else None

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def status(self) -> RunStatus:
        return self._status

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def dropped_records(self) -> int:
        """Records that reached no sink because the queue was full."""
        return self._worker.dropped

    @property
    def failed_records(self) -> Dict[str, int]:
        """Records each sink could not store, by sink name. Per sink because
        another sink may still hold them."""
        return dict(self._worker.failed_records)

    def __repr__(self) -> str:
        return f"<Run id={self.id!r} mode={self._mode!r} status={self._status.value}>"

    # ------------------------------------------------------------------- log

    def log_traces(self, traces: Iterable[Any]) -> None:
        """Hand bare traces to the sinks. Returns immediately.

        Accepts verifiers ``Trace`` objects or plain JSON mappings. Both reach
        Prime Traces; the v0 sample table is projected from *episodes*, so a
        bare trace has no row there. A rollout that is an episode (a group of
        traces) goes through :meth:`log_episodes`. Call this as rollouts
        complete; nothing is buffered until the end.
        """
        self._submit("log_traces", traces)

    def log_episodes(self, episodes: Iterable[Any]) -> None:
        """Hand episodes — grouped traces — to the sinks. Returns immediately.

        Accepts verifiers ``Episode`` objects or plain JSON mappings with a
        ``traces`` list. The episode's ``run`` reaches every member trace.
        Both reach Prime Traces; the v0 sample table (what today's viewer
        reads) is projected from episode *objects* only — a JSON episode has
        no row there, which the samples sink warns about once.
        """
        self._submit("log_episodes", episodes)

    def _submit(self, operation: str, records: Iterable[Any]) -> None:
        # Finish holds the same lock through worker teardown. A log call either
        # queues its batch first or observes that the run is already closing;
        # it can never restart the worker between those two states.
        with self._finish_lock:
            self._require_live(operation)
            batch = list(records)
            if batch:
                self._worker.submit(batch)

    def update_summary(self, values: Mapping[str, Any]) -> None:
        """Merge run-level outputs into :attr:`summary` ahead of :meth:`finish`.

        Non-finite numbers are dropped here, as they are for
        ``finish(summary=...)``; writing to ``summary`` directly skips that.
        """
        with self._finish_lock:
            self._require_live("update_summary")
            self.summary.update(_clean_metrics(values))

    def flush(self, timeout: Optional[float] = 30.0) -> bool:
        """Block until queued records have been written. Under
        ``on_error="raise"`` this is the first place an upload failure surfaces."""
        flushed = self._worker.flush(timeout=timeout)
        self._raise_deferred()
        return flushed

    # ---------------------------------------------------------------- finish

    def finish(
        self,
        summary: Optional[Mapping[str, Any]] = None,
        *,
        status: Union[RunStatus, str] = RunStatus.COMPLETED,
        error: Optional[str] = None,
    ) -> None:
        """Flush everything and close the run out. Idempotent: the first caller
        reports the status, concurrent callers wait for that teardown."""
        with self._finish_lock:
            if self._finished:
                return
            resolved = RunStatus(status) if not isinstance(status, RunStatus) else status
            if not resolved.is_terminal():
                raise ValueError(f"finish() requires a terminal status, got {resolved.value!r}")
            self._finishing = True
            try:
                self._finish_once(summary, resolved, error)
            finally:
                self._finishing = False
                self._finished = True

    def _finish_once(
        self,
        summary: Optional[Mapping[str, Any]],
        resolved: RunStatus,
        error: Optional[str],
    ) -> None:
        if summary:
            self.summary.update(_clean_metrics(summary))
        self._status = resolved

        deadline = time.monotonic() + max(0.0, self._finish_timeout)

        def remaining() -> float:
            return max(0.0, deadline - time.monotonic())

        # Records first, so a dashboard reacting to the terminal status never
        # sees a finished run with samples still landing.
        if not self._worker.flush(timeout=remaining()):
            logger.warning(
                "Run %s: uploads did not drain within %ss; finalizing anyway. "
                "Some records may be missing from this run.",
                self.id,
                self._finish_timeout,
            )
        self._worker.close(timeout=remaining())

        if self._owns_lifecycle:
            self._teardown_step(
                "updating the run",
                lambda: self._backend.update(
                    self.id, config=self.config or None, summary=self.summary or None
                ),
            )
            self._teardown_step(
                "finalizing the run",
                lambda: self._backend.finalize(
                    self.id,
                    status=resolved,
                    summary=self.summary or None,
                    error=error or (self.errors[0] if self.errors else None),
                    config=self.config or None,
                ),
            )
        self._teardown_step("closing the backend", self._backend.close)
        atexit.unregister(self._atexit_hook)

        if self._worker.dropped:
            logger.warning(
                "Run %s finished with %d record(s) that reached no sink; the producer "
                "outran the uploader.",
                self.id,
                self._worker.dropped,
            )
        for sink_name, count in self._worker.failed_records.items():
            logger.warning(
                "Run %s: the %s sink could not store %d record(s)", self.id, sink_name, count
            )
        # Last, so a run whose teardown failed is still closed out first.
        self._raise_deferred()

    def fail(self, error: Union[str, BaseException]) -> None:
        """Close the run out as failed."""
        self.finish(status=RunStatus.FAILED, error=_describe(error))

    # -------------------------------------------------------- context manager

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if exc_type is None:
            self.finish()
            return False

        if isinstance(exc, KeyboardInterrupt):
            # An interrupt is a decision, not a fault.
            status, error = RunStatus.CRASHED, "interrupted"
        else:
            status, error = RunStatus.FAILED, _describe(exc)

        try:
            self.finish(status=status, error=error)
        except Exception as finish_error:
            # The producer's exception is why this block is unwinding; a
            # telemetry teardown error must not replace it. Control-flow
            # exceptions (KeyboardInterrupt, SystemExit) deliberately pass.
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

    def reset_after_fork(self) -> None:
        """Make an inherited handle safe in a forked child: fresh lock, and the
        parent keeps ownership of the lifecycle so the child's atexit hook
        cannot finalize a still-running run."""
        self._finish_lock = threading.Lock()
        self._finishing = False
        self._owns_lifecycle = False
        self._deferred_error = None

    def _on_process_exit(self) -> None:
        """The process is exiting and nobody called ``finish()``: report CRASHED
        rather than FAILED, since the producer never said the run failed."""
        if self._finished:
            return
        logger.warning("Run %s was never finished; reporting it as crashed", self.id)
        try:
            self.finish(status=RunStatus.CRASHED, error="process exited without finishing the run")
        except Exception as exc:  # noqa: BLE001 - never a traceback from atexit
            logger.warning("Run %s: reporting the crash failed: %s", self.id, exc)

    def _require_live(self, operation: str) -> None:
        if self._finishing or self._finished:
            raise RunFinishedError(
                f"{operation}() was called on run {self.id}, which is already finished. "
                "The platform has closed this run out; start a new one."
            )

    def _record_sink_error(self, sink_name: str, exc: Exception) -> None:
        """Called on the uploader thread when a sink gives up on a batch."""
        self._note(f"writing to the {sink_name} sink", exc)

    def _teardown_step(self, what: str, call: Any) -> None:
        """Run one teardown step without letting it skip the steps after it."""
        try:
            call()
        except Exception as exc:  # noqa: BLE001 - teardown must continue
            self._note(what, exc)

    def _note(self, what: str, exc: BaseException) -> None:
        """Record a contained failure: warn, or hold it for the next sync point."""
        message = f"{what} failed: {type(exc).__name__}: {exc}"
        self.errors.append(message)
        if self._on_error == "raise":
            if self._deferred_error is None:
                self._deferred_error = exc
        else:
            logger.warning("Run %s: %s", self.id, message)

    def _raise_deferred(self) -> None:
        """Re-raise the first held failure, once."""
        exc = self._deferred_error
        if exc is None:
            return
        self._deferred_error = None
        raise exc


# --------------------------------------------------------------------- init


def init(
    *,
    name: Optional[str] = None,
    environments: Optional[Sequence[Any]] = None,
    model: Optional[str] = None,
    framework: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    config: Optional[Any] = None,
    mode: Optional[Mode] = None,
    team_id: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    on_error: OnError = "warn",
) -> Run:
    """Start a run and return a handle to it.

    Call this *before* the first rollout: the ID it returns is what every trace
    in the run should carry, and the URL is what a producer prints.

    ``mode`` defaults to ``$PRIME_RUNS_MODE``, else online when there is an API
    key and disabled (with a warning) when there is not. ``config`` is what the
    run was configured with: the path to the file it was launched from (stored
    byte for byte under ``config_source``), or a mapping taken as given.
    """
    settings = Config()
    api_key = api_key if api_key is not None else settings.api_key
    base_url = base_url or settings.base_url
    team_id = team_id if team_id is not None else settings.team_id

    spec = RunSpec(
        name=name,
        environments=[EnvironmentRef.coerce(entry) for entry in (environments or [])],
        model=model,
        framework=framework,
        description=description,
        tags=list(tags or []),
        team_id=team_id,
        config=_normalize_config(config),
    )
    resolved_mode = _resolve_mode(mode, api_key=api_key)

    backend: Backend
    sinks: List[Sink]
    if resolved_mode == "disabled":
        backend = DisabledBackend()
        handle = RunHandle(id=disabled_run_id(), name=name)
        sinks = []
    else:
        if not api_key:
            raise ConfigurationError(
                'mode="online" needs an API key. Set PRIME_API_KEY, run `prime login`, '
                'or pass mode="disabled".'
            )
        client = PlatformClient(api_key=api_key, base_url=base_url, timeout=DEFAULT_TIMEOUT)
        backend = EvalsBackend(client, frontend_url=settings.frontend_url, team_id=team_id)
        try:
            handle = backend.create(spec)
        except BaseException:
            # Ownership has not reached a Run yet, so nothing else can release
            # the connection pool when environment resolution or creation fails.
            try:
                backend.close()
            except Exception as close_error:  # noqa: BLE001 - preserve the create failure
                logger.debug(
                    "Error closing the platform client after run creation failed: %s",
                    close_error,
                )
            raise
        # Both transports run during the transition: traces is the system of
        # record, the sample table is what today's viewer reads.
        sinks = [TracesSink(api_key=api_key, team_id=team_id), EvalSamplesSink(client)]

    run = Run(
        backend=backend,
        handle=handle,
        spec=spec,
        sinks=sinks,
        mode=resolved_mode,
        on_error=on_error,
    )
    if run.url:
        logger.info("Run %s: %s", run.id, run.url)
    return run


def _resolve_mode(mode: Optional[Mode], *, api_key: str) -> Mode:
    if mode is None:
        env_mode = os.getenv(MODE_ENV)
        if env_mode:
            mode = env_mode.strip().lower()  # type: ignore[assignment]
    if mode not in (None, "online", "disabled"):
        raise ConfigurationError(f"mode={mode!r} is not one of 'online' or 'disabled'")
    if mode is None:
        if api_key:
            mode = "online"
        else:
            logger.warning(
                "No API key found (set PRIME_API_KEY or run `prime login`); "
                "this run will not be tracked."
            )
            mode = "disabled"
    return mode  # type: ignore[return-value]


def _sink_context(spec: RunSpec) -> Dict[str, str]:
    """Upload-scoped provenance. Not the join key — that is ``run.id`` inside
    the trace document."""
    context = {"source": "prime-runs", "run_type": RUN_KIND}
    if spec.framework:
        context["framework"] = spec.framework
    if spec.model:
        context["model"] = spec.model
    return context


def _normalize_config(value: Any) -> Dict[str, Any]:
    """A producer's config as a plain dict.

    A path is the file the run was launched from, kept byte for byte under
    ``config_source`` (stored, not parsed — the platform can read TOML). A
    mapping is taken exactly as given.
    """
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (str, os.PathLike, ConfigSource)):
        source = ConfigSource.coerce(value)
        assert source is not None  # coerce only returns None for None
        return {CONFIG_SOURCE_KEY: source.to_dict()}
    raise TypeError(
        f"config must be a path to the run's config file or a mapping, got {type(value).__name__}"
    )


def _describe(error: Union[str, BaseException]) -> str:
    if isinstance(error, BaseException):
        return f"{type(error).__name__}: {error}"
    return str(error)


def _clean_metrics(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """Drop NaN/infinity: strict JSON rejects them, and the failure would
    surface as an opaque 400 on the whole request."""
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


__all__ = ["Run", "init", "MODE_ENV"]
