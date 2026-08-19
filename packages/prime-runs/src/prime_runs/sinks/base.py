"""The contract a sample sink implements.

A sink moves *records* — traces, episodes, rollouts — to wherever they are
stored. It knows nothing about run lifecycle; a backend closing a run and a
sink flushing its last batch are separate events on purpose.

Sinks are independent of backends and of each other. During the transition both
the traces sink and the legacy eval-samples sink run at once, so the dashboard
keeps working for accounts outside the traces beta while traces becomes the
system of record. When the Viewer API reads traces natively, the default sink
list drops one entry — and no producer changes.

Every sink must be *degradable*: a sink that cannot write sets ``enabled =
False`` and says why, once. A run whose traces are gated is still a valid run.
"""

from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable


@runtime_checkable
class Sink(Protocol):
    """A destination for run records."""

    name: str
    enabled: bool

    def start(self, run_id: str, context: Mapping[str, str]) -> None:
        """Bind the sink to a run before the first write."""
        ...

    def write(
        self,
        records: Sequence[Any],
        *,
        line_format: Optional[str] = None,
        step: Optional[int] = None,
    ) -> None:
        """Send one batch. Called from the uploader thread, never inline."""
        ...

    def flush(self) -> None:
        """Block until everything handed over so far has been written."""
        ...

    def close(self) -> None:
        """Release transport resources."""
        ...


def to_mapping(record: Any) -> Mapping[str, Any]:
    """The JSON mapping for a record, whatever shape the producer handed us.

    Mirrors ``prime_traces.SupportsToRecord``: verifiers ``Trace``/``Episode``
    and prime-rl ``Rollout`` all implement ``to_record()``, and plain dicts pass
    straight through.
    """
    if isinstance(record, Mapping):
        return record
    to_record = getattr(record, "to_record", None)
    if callable(to_record):
        value = to_record()
        if isinstance(value, Mapping):
            return value
        raise TypeError(f"{type(record).__name__}.to_record() must return a mapping")
    raise TypeError(f"{type(record).__name__} is not a mapping and has no to_record()")
