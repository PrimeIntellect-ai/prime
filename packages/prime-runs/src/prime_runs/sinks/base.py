"""The contract a record sink implements, plus the record helpers sinks share.

A sink moves records — traces, episodes — to wherever they are stored. It knows
nothing about run lifecycle. Sinks are independent of each other: during the
transition the traces sink and the legacy eval-samples sink both run, and
retiring one is a change to the default sink list, not to any producer.

Every sink must be degradable: one that cannot write sets ``enabled = False``
and says why, once.
"""

from typing import Any, Dict, Mapping, Protocol, Sequence

from ..models import RUN_KIND


class Sink(Protocol):
    """A destination for run records."""

    name: str
    enabled: bool

    def start(self, run_id: str, context: Mapping[str, str]) -> None:
        """Bind the sink to a run before the first write."""
        ...

    def write(self, records: Sequence[Any]) -> None:
        """Send one batch. Called from the uploader thread, never inline."""
        ...

    def flush(self) -> None:
        """Block until everything handed over so far has been written."""
        ...

    def close(self) -> None:
        """Release transport resources."""
        ...


def to_mapping(record: Any) -> Mapping[str, Any]:
    """The JSON mapping for a record: a dict passes through, anything else must
    implement ``to_record()`` (verifiers ``Trace``/``Episode`` do)."""
    if isinstance(record, Mapping):
        return record
    to_record = getattr(record, "to_record", None)
    if callable(to_record):
        value = to_record()
        if isinstance(value, Mapping):
            return value
        raise TypeError(f"{type(record).__name__}.to_record() must return a mapping")
    raise TypeError(f"{type(record).__name__} is not a mapping and has no to_record()")


def is_episode(record: Any) -> bool:
    """Whether a record (object or mapping) is an episode rather than a trace."""
    if isinstance(record, Mapping):
        return "traces" in record
    return hasattr(record, "traces")


def stamp_run(mapping: Mapping[str, Any], run_id: str) -> Dict[str, Any]:
    """A copy of ``mapping`` carrying ``run`` if it did not already. Producer
    objects are never stamped — they carry their own ``run`` — but a bare dict
    with no ``run.id`` is an orphaned, unqueryable upload."""
    if mapping.get("run"):
        return dict(mapping)
    return {**mapping, "run": {"id": run_id, "type": RUN_KIND}}
