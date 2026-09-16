"""Legacy samples are used only when Prime Traces explicitly denies beta access."""

from typing import Any, Dict, Mapping, Optional, Sequence

from .base import Sink
from .traces import TracesSink


class LegacySamplesFallback(Sink):
    """Run after the traces sink, so its first denial routes the same batch here.

    Initialization is lazy: beta users do not need the legacy Parquet encoder.
    The wrapped sink keeps its own name, retirement policy, and loss accounting.
    """

    def __init__(self, traces: TracesSink, samples: Sink) -> None:
        self.name = samples.name
        self._traces = traces
        self._samples = samples
        self._run_id: Optional[str] = None
        self._context: Dict[str, str] = {}
        self._started = False

    @property
    def enabled(self) -> bool:
        return self._samples.enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._samples.enabled = value

    def start(self, run_id: str, context: Mapping[str, str]) -> None:
        self._run_id = run_id
        self._context = dict(context)

    def write(self, records: Sequence[Any]) -> None:
        if not self.enabled or not records or not self._traces.service_not_enabled:
            return
        if not self._started:
            if self._run_id is None:
                raise RuntimeError("LegacySamplesFallback.write called before start()")
            self._samples.start(self._run_id, self._context)
            self._started = True
        if self.enabled:
            self._samples.write(records)

    def flush(self) -> None:
        if self._started:
            self._samples.flush()

    def close(self) -> None:
        self._samples.close()
