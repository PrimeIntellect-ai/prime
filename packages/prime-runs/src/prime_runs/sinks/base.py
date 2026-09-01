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


class SinkWriteError(Exception):
    """A sink stored only part of a submitted batch.

    ``cause`` drives the worker's retry/retirement policy; ``failed_records``
    lets it account for only the input records that were not fully stored.
    """

    def __init__(self, cause: Exception, *, failed_records: int) -> None:
        self.cause = cause
        self.failed_records = failed_records
        super().__init__(
            f"{failed_records} record(s) were not fully stored: {type(cause).__name__}: {cause}"
        )


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


def stamp_run(mapping: Mapping[str, Any], run_id: str, run_type: str = RUN_KIND) -> Dict[str, Any]:
    """A copy of ``mapping`` carrying ``run`` at the top level, and on every
    member trace of an episode that lacks one.

    A record that already names a run keeps it; one without is stamped, since
    an upload with no ``run.id`` is orphaned and unqueryable. Members matter
    because the traces service derives ``run_id`` from ``trace.run.id`` only —
    the episode envelope's ``run`` is never read — while producers (verifiers)
    record the run on the episode and nowhere else.
    """
    stamped = dict(mapping)
    if not stamped.get("run"):
        stamped["run"] = {"id": run_id, "type": run_type}
    members = stamped.get("traces")
    if isinstance(members, list):
        run = stamped["run"]
        stamped["traces"] = [
            {**member, "run": run}
            if isinstance(member, Mapping) and not member.get("run")
            else member
            for member in members
        ]
    return stamped
