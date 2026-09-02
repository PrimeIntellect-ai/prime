"""The run handle, and ``init()`` that opens one.

``init()`` is called before the first rollout; the id it returns keys every
record the run uploads, whatever run id the producer recorded locally.
"""

import asyncio
import atexit
import logging
import math
import os
import threading
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from . import _fork
from ._http import DEFAULT_TIMEOUT, UPLOAD_TIMEOUT, PlatformClient
from .backend import Backend, DisabledBackend, EvalsBackend, RftBackend, disabled_run_id
from .config import Config
from .exceptions import ConfigurationError, RunFinishedError
from .models import (
    CONFIG_SOURCE_KEY,
    ConfigSource,
    EnvironmentRef,
    Mode,
    OnError,
    RunHandle,
    RunKind,
    RunSpec,
    RunStatus,
    TrainingSpec,
)
from .sinks import EvalSamplesSink, RftMetricsSink, RftSamplesSink, Sink, TracesSink
from .worker import UploadWorker, deadline_after, time_left

logger = logging.getLogger(__name__)

MODE_ENV = "PRIME_RUNS_MODE"
#: How long ``finish()`` lets queued uploads drain: one in-flight sample POST
#: may take this long, and a shorter budget would abandon it about to succeed.
DEFAULT_FINISH_TIMEOUT = float(UPLOAD_TIMEOUT.read or 300.0)
#: Training runs last days: a training sink that struck out on transient
#: failures is tried again after this long instead of retired for the run.
TRAIN_RETIRE_COOLDOWN = 300.0


class Run:
    """A live run: an id, a URL, somewhere to put records, a summary.

    With ``on_error="warn"`` (the default) nothing the platform raises escapes
    into the producer's loop; ``on_error="raise"`` surfaces the first failure
    from :meth:`flush` or :meth:`finish`.
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
        finish_timeout: Optional[float] = None,
        metrics_sinks: Optional[List[Sink]] = None,
        attached: bool = False,
        retire_cooldown: Optional[float] = None,
    ) -> None:
        self._backend = backend
        self._handle = handle
        self._spec = spec
        self._mode: Mode = mode
        self._on_error: OnError = on_error
        self._status = RunStatus.RUNNING
        # A forked child inherits the handle but must not close the parent's run.
        self._owns_lifecycle = True
        # A run a launcher created and handed over: the platform owns its
        # failure marking, this process only completes it.
        self._attached = attached

        self.config: Dict[str, Any] = dict(spec.config)
        self.summary: Dict[str, Any] = {}
        self.errors: List[str] = []
        # on_error="raise": the first uploader-thread failure, re-raised from
        # flush() or finish() where the caller is looking.
        self._deferred_error: Optional[BaseException] = None

        self._finish_timeout = (
            DEFAULT_FINISH_TIMEOUT if finish_timeout is None else max(0.0, float(finish_timeout))
        )
        # _state_lock guards admission only; _finish_lock is held through the
        # slow teardown so concurrent finish() callers wait for it.
        self._state_lock = threading.Lock()
        self._finish_lock = threading.Lock()
        self._finishing = False
        self._finished = False
        self._atexit_hook = self._on_process_exit
        _fork.register(self)

        sinks = sinks or []
        metrics_sinks = metrics_sinks or []
        # Separate uploaders: a slow sample upload must not hold metrics back.
        self._worker = UploadWorker(
            sinks, on_error=self._record_sink_error, retire_cooldown=retire_cooldown
        )
        self._metrics_worker = UploadWorker(
            metrics_sinks, on_error=self._record_sink_error, retire_cooldown=retire_cooldown
        )
        context = _sink_context(spec)
        for sink in [*sinks, *metrics_sinks]:
            try:
                sink.start(handle.id, context)
            except Exception as exc:  # noqa: BLE001 - a bad sink is not a bad run
                sink.enabled = False
                self._note(f"starting sink {sink.name}", exc)
                if self._on_error == "raise":
                    # Close the remote run before the failure reaches the caller.
                    self.finish(status=RunStatus.FAILED, error=_describe(exc))
                    raise exc

        atexit.register(self._atexit_hook)

    @property
    def id(self) -> str:
        return self._handle.id

    @property
    def name(self) -> Optional[str]:
        return self._handle.name

    @property
    def url(self) -> Optional[str]:
        return self._handle.url

    @property
    def config_source(self) -> Optional[ConfigSource]:
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
    def kind(self) -> RunKind:
        return self._spec.kind

    @property
    def attached(self) -> bool:
        return self._attached

    @property
    def dropped_records(self) -> int:
        """Records that reached no sink because a queue was full."""
        return self._worker.dropped + self._metrics_worker.dropped

    @property
    def failed_records(self) -> Dict[str, int]:
        """Records each sink could not store, by sink name."""
        return {**self._worker.failed_records, **self._metrics_worker.failed_records}

    def __repr__(self) -> str:
        return f"<Run id={self.id!r} mode={self._mode!r} status={self._status.value}>"

    def log_traces(self, traces: Iterable[Any]) -> None:
        """Queue bare traces: verifiers ``Trace`` objects or JSON mappings."""
        self._submit("log_traces", traces)

    def log_episodes(self, episodes: Iterable[Any]) -> None:
        """Queue episodes: verifiers ``Episode`` objects or JSON mappings with a
        ``traces`` list. The sample tables are projected from episode objects
        only; a JSON episode reaches Prime Traces alone."""
        self._submit("log_episodes", episodes)

    def log_metrics(self, values: Mapping[str, Any], *, step: Optional[int] = None) -> None:
        """Queue one step's metrics (training runs; a no-op for evals). ``step``
        becomes ``values["step"]`` unless already set, every row gets a
        ``_timestamp``, and non-finite numbers are dropped."""
        self._require_live("log_metrics")
        record = _clean_metrics(values)
        if step is not None:
            record.setdefault("step", step)
        record.setdefault("_timestamp", time.time())
        with self._state_lock:
            self._require_live("log_metrics")
            self._metrics_worker.submit([record])

    def _submit(self, operation: str, records: Iterable[Any]) -> None:
        self._require_live(operation)  # before consuming a lazy iterable
        batch = list(records)
        with self._state_lock:
            self._require_live(operation)
            if batch:
                self._worker.submit(batch)

    def update_summary(self, values: Mapping[str, Any]) -> None:
        """Merge run-level outputs into :attr:`summary` ahead of :meth:`finish`."""
        with self._state_lock:
            self._require_live("update_summary")
            self.summary.update(_clean_metrics(values))

    def flush(self, timeout: Optional[float] = 30.0) -> bool:
        """Block until queued records have been written. Under
        ``on_error="raise"`` this is the first place an upload failure surfaces."""
        deadline = deadline_after(timeout)
        flushed = self._worker.flush(timeout=time_left(deadline))
        flushed = self._metrics_worker.flush(timeout=time_left(deadline)) and flushed
        self._raise_deferred()
        return flushed

    def finish(
        self,
        summary: Optional[Mapping[str, Any]] = None,
        *,
        status: Union[RunStatus, str] = RunStatus.COMPLETED,
        error: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        """Drain the uploads and close the run out. Idempotent: the first caller
        reports the status, concurrent callers wait for that teardown.
        ``timeout`` overrides the run's ``finish_timeout`` for this call."""
        with self._finish_lock:
            if self._finished:
                return
            resolved = RunStatus(status) if not isinstance(status, RunStatus) else status
            if not resolved.is_terminal():
                raise ValueError(f"finish() requires a terminal status, got {resolved.value!r}")
            with self._state_lock:
                self._finishing = True
            try:
                self._finish_once(summary, resolved, error, timeout)
            finally:
                with self._state_lock:
                    self._finishing = False
                    self._finished = True

    def _finish_once(
        self,
        summary: Optional[Mapping[str, Any]],
        resolved: RunStatus,
        error: Optional[str],
        timeout: Optional[float],
    ) -> None:
        if summary:
            self.summary.update(_clean_metrics(summary))
        self._status = resolved

        budget = self._finish_timeout if timeout is None else max(0.0, float(timeout))
        deadline = deadline_after(budget)
        # Records first, so the dashboard never sees a finished run with samples landing.
        drained = self._worker.flush(timeout=time_left(deadline))
        drained = self._metrics_worker.flush(timeout=time_left(deadline)) and drained
        if not drained:
            logger.warning(
                "Run %s: uploads did not drain within %ss; finalizing anyway. "
                "Some records may be missing from this run.",
                self.id,
                budget,
            )
        self._worker.close(timeout=time_left(deadline))
        self._metrics_worker.close(timeout=time_left(deadline))

        if self._owns_lifecycle:
            self._teardown_step(
                "updating the run",
                lambda: self._backend.update(
                    self.id, config=self.config or None, summary=self.summary or None
                ),
            )
            if self._attached and resolved is not RunStatus.COMPLETED:
                # The launcher marks its own run failed; reporting it here would race that.
                logger.info(
                    "Run %s: %s, but the platform owns the status of an attached run; "
                    "leaving it to the launcher",
                    self.id,
                    resolved.value,
                )
            else:
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

        if self.dropped_records:
            logger.warning(
                "Run %s finished with %d record(s) that reached no sink; the producer "
                "outran the uploader.",
                self.id,
                self.dropped_records,
            )
        for sink_name, count in self.failed_records.items():
            logger.warning(
                "Run %s: the %s sink could not store %d record(s)", self.id, sink_name, count
            )
        self._raise_deferred()  # last, so the run is closed out even when this raises

    def fail(self, error: Union[str, BaseException], *, timeout: Optional[float] = None) -> None:
        self.finish(status=RunStatus.FAILED, error=_describe(error), timeout=timeout)

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if exc_type is None:
            self.finish()
            return False
        if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)):
            status, error = RunStatus.CANCELLED, "interrupted"  # a decision, not a fault
        else:
            status, error = RunStatus.FAILED, _describe(exc)
        try:
            self.finish(status=status, error=error)
        except Exception as finish_error:
            # The producer's exception is why this block is unwinding; a teardown
            # error must not replace it. Control-flow exceptions pass through.
            logger.warning(
                "Run %s: finishing after %s also failed: %s: %s",
                self.id,
                exc_type.__name__,
                type(finish_error).__name__,
                finish_error,
                exc_info=True,
            )
        return False

    def reset_after_fork(self) -> None:
        """Fresh locks, and the parent keeps the lifecycle: the child's atexit
        hook must not finalize a still-running run."""
        self._state_lock = threading.Lock()
        self._finish_lock = threading.Lock()
        self._finishing = False
        self._owns_lifecycle = False
        self._deferred_error = None

    def _on_process_exit(self) -> None:
        """atexit: nobody called ``finish()``, so report CRASHED rather than FAILED."""
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
        self._note(f"writing to the {sink_name} sink", exc)

    def _teardown_step(self, what: str, call: Any) -> None:
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
        exc = self._deferred_error
        if exc is None:
            return
        self._deferred_error = None
        raise exc


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
    finish_timeout: Optional[float] = None,
    kind: RunKind = "eval",
    id: Optional[str] = None,
    training: Optional[TrainingSpec] = None,
) -> Run:
    """Open a run and return its handle. Call it before the first rollout.

    ``mode`` defaults to ``$PRIME_RUNS_MODE``, else online when there is an API
    key and disabled (with a warning) when there is not. ``config`` is the path
    to the file the run was launched from (stored byte for byte under
    ``config_source``) or a mapping taken as given. ``finish_timeout`` bounds
    the drain in :meth:`Run.finish`.

    ``kind="train"`` opens an external training run: ``model`` is the base
    model, ``environments`` the hub ids, ``training`` the display fields, and a
    team is required. ``id`` attaches to an external run a launcher already
    created (``$RUN_ID``): nothing is registered, the platform keeps the run's
    failure marking, and a clean finish still completes it.
    """
    settings = Config()
    api_key = api_key if api_key is not None else settings.api_key
    base_url = base_url or settings.base_url
    team_id = team_id if team_id is not None else settings.team_id

    if kind not in ("eval", "train"):
        raise ConfigurationError(f"kind={kind!r} is not one of 'eval' or 'train'")
    if id is not None and kind != "train":
        raise ConfigurationError(
            "id= attaches to an existing training run; an eval run is always created here."
        )
    if training is not None and kind != "train":
        raise ConfigurationError("training= only applies to kind='train'")

    spec = RunSpec(
        name=name,
        environments=[EnvironmentRef.coerce(entry) for entry in (environments or [])],
        model=model,
        framework=framework,
        description=description,
        tags=list(tags or []),
        team_id=team_id,
        config=_normalize_config(config),
        kind=kind,
        training=training,
    )
    resolved_mode = _resolve_mode(mode, api_key=api_key)

    backend: Backend
    sinks: List[Sink]
    metrics_sinks: List[Sink] = []
    attached = False
    if resolved_mode == "disabled":
        backend = DisabledBackend()
        handle = RunHandle(id=id or disabled_run_id(), name=name)
        sinks = []
    else:
        if not api_key:
            raise ConfigurationError(
                'mode="online" needs an API key. Set PRIME_API_KEY, run `prime login`, '
                'or pass mode="disabled".'
            )
        client = PlatformClient(api_key=api_key, base_url=base_url, timeout=DEFAULT_TIMEOUT)
        if kind == "train":
            rft = RftBackend(client, frontend_url=settings.frontend_url, team_id=team_id)
            backend = rft
        else:
            backend = EvalsBackend(client, frontend_url=settings.frontend_url, team_id=team_id)
        try:
            if kind == "train" and id is not None:
                handle = rft.attach(id)
                attached = True
            else:
                handle = backend.create(spec)
        except BaseException:
            # Nothing else can release the connection pool yet.
            try:
                backend.close()
            except Exception as close_error:  # noqa: BLE001 - preserve the create failure
                logger.debug("Error closing the platform client: %s", close_error)
            raise
        # Both transports run during the transition: traces is the system of
        # record, the sample table is what today's viewer reads.
        traces = TracesSink(api_key=api_key, team_id=team_id)
        if kind == "train":
            sinks = [traces, RftSamplesSink(client)]
            metrics_sinks = [RftMetricsSink(client)]
        else:
            sinks = [traces, EvalSamplesSink(client)]

    run = Run(
        backend=backend,
        handle=handle,
        spec=spec,
        sinks=sinks,
        mode=resolved_mode,
        on_error=on_error,
        finish_timeout=finish_timeout,
        metrics_sinks=metrics_sinks,
        attached=attached,
        retire_cooldown=TRAIN_RETIRE_COOLDOWN if kind == "train" else None,
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
    """Upload-scoped provenance; the join key is ``run.id`` inside the document."""
    context = {"source": "prime-runs", "run_type": spec.kind}
    if spec.framework:
        context["framework"] = spec.framework
    if spec.model:
        context["model"] = spec.model
    return context


def _normalize_config(value: Any) -> Dict[str, Any]:
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


_DROP: Any = object()


def _finite(value: Any) -> Any:
    """``value`` without NaN/infinity anywhere inside it (strict JSON rejects
    them as an opaque 400), or ``_DROP``. Empty nested mappings are dropped."""
    if isinstance(value, float) and not math.isfinite(value):
        return _DROP
    if isinstance(value, Mapping):
        cleaned = {}
        for key, item in value.items():
            kept = _finite(item)
            # Identity checks only: an array-like value would make `==` ambiguous.
            if kept is _DROP or (isinstance(item, Mapping) and not kept):
                continue
            cleaned[key] = kept
        return cleaned
    if isinstance(value, (list, tuple)):
        return [item for item in (_finite(item) for item in value) if item is not _DROP]
    return value


def _clean_metrics(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    return _finite(dict(metrics))
