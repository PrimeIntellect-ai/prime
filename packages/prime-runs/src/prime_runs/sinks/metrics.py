"""Training metrics sink: one ``POST /rft/metrics`` per logged step.

Requests are replayed after an ambiguous failure: the platform keeps one row per
step, so a duplicate collapses server-side, while a lost row is a hole in the
curves for good.
"""

from typing import Any, Callable, Mapping, Optional, Sequence

from .._http import PlatformClient
from .base import Sink, send_each


class RftMetricsSink(Sink):
    name = "rft_metrics"

    def __init__(self, client: PlatformClient) -> None:
        self.enabled = True
        self._client = client
        self._run_id: Optional[str] = None
        self.steps_written = 0

    def start(self, run_id: str, context: Mapping[str, str]) -> None:
        self._run_id = run_id

    def write(self, records: Sequence[Any]) -> None:
        if not self.enabled or not records:
            return
        if self._run_id is None:
            raise RuntimeError("RftMetricsSink.write called before start()")
        send_each([(1, self._sender(record)) for record in records])

    def _sender(self, record: Any) -> Callable[[], None]:
        def send() -> None:
            if not isinstance(record, Mapping):
                raise TypeError(f"metrics must be a mapping, got {type(record).__name__}")
            self._client.post(
                "/rft/metrics",
                json_body={"run_id": self._run_id, "metrics": dict(record)},
                idempotent=True,
            )
            self.steps_written += 1

        return send

    def flush(self) -> None:
        """Writes are synchronous."""

    def close(self) -> None:
        """The client belongs to the backend."""
