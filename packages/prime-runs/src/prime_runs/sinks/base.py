"""The sink contract, the shared write loop, and the record helpers."""

from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Sequence, Tuple

from ..exceptions import is_record_rejection, is_transient
from ..models import RUN_KIND


class SinkWriteError(Exception):
    """A sink stored only part of a batch. ``cause`` drives the worker's retry
    and retirement policy, ``failed_records`` its loss accounting."""

    def __init__(self, cause: Exception, *, failed_records: int) -> None:
        self.cause = cause
        self.failed_records = failed_records
        super().__init__(
            f"{failed_records} record(s) were not fully stored: {type(cause).__name__}: {cause}"
        )


class Sink(Protocol):
    """A destination for run records. ``write`` runs on the uploader thread. A
    sink that cannot write sets ``enabled = False`` and says why, once."""

    name: str
    enabled: bool

    def start(self, run_id: str, context: Mapping[str, str]) -> None: ...

    def write(self, records: Sequence[Any]) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


Unit = Tuple[int, Callable[[], Any]]
"""``(record count, send)``: one request's worth of a batch."""


def send_each(units: Sequence[Unit]) -> None:
    """Send a batch one unit at a time, then raise ``SinkWriteError`` if any
    failed. A record rejection or a transient failure costs its own unit; a
    sink-wide failure stops the batch and costs the rest too. A transient
    error is what gets reported, so the worker's strike accounting is not
    fooled by a record-local one in the same batch."""
    failed = 0
    reported: Optional[Exception] = None
    for position, (count, send) in enumerate(units):
        try:
            send()
        except Exception as exc:  # noqa: BLE001 - classified below
            failed += count
            if not (is_record_rejection(exc) or is_transient(exc)):
                failed += sum(rest for rest, _ in units[position + 1 :])
                reported = exc
                break
            if reported is None or is_transient(exc):
                reported = exc
    if reported is not None:
        raise SinkWriteError(reported, failed_records=failed) from reported


def to_mapping(record: Any) -> Mapping[str, Any]:
    """A dict passes through; anything else must implement ``to_record()``."""
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
    if isinstance(record, Mapping):
        return "traces" in record
    return hasattr(record, "traces")


def stamp_run(mapping: Mapping[str, Any], run_id: str, run_type: str = RUN_KIND) -> Dict[str, Any]:
    """A copy of ``mapping`` with ``run`` on the envelope and on every member
    trace that lacks one. A record that already names a run keeps it. Members
    matter because the traces service reads ``run_id`` from ``trace.run.id``
    only, while verifiers records the run on the episode."""
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
