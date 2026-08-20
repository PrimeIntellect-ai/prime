"""The primary sample transport, and how it degrades."""

import pytest
from _fakes import make_episode, make_trace
from prime_traces import LineFormat
from prime_traces.exceptions import ForbiddenError, RetryableAPIError

from prime_runs.sinks import TracesSink


class FakeTracesClient:
    def __init__(self, raises: Exception = None) -> None:
        self.raises = raises
        self.calls = []
        self.closed = False

    def upload_records(self, records, **kwargs):
        self.calls.append((list(records), kwargs))
        if self.raises is not None:
            raise self.raises
        return ["receipt"]

    def close(self) -> None:
        self.closed = True


def make_sink(client=None, **kwargs) -> TracesSink:
    sink = TracesSink(client=client or FakeTracesClient(), **kwargs)
    sink.start("run-1", {"source": "prime-runs", "run_kind": "eval", "framework": "verifiers"})
    return sink


def test_records_go_out_with_provenance_but_not_the_join_key():
    """``run.id`` inside the document is the indexed column; ``context`` is an
    upload-scoped map that answers a different question."""
    client = FakeTracesClient()
    sink = make_sink(client)

    sink.write([{"id": "t1", "run": {"id": "run-1"}}], step=4)

    _, kwargs = client.calls[0]
    assert kwargs["context"] == {
        "source": "prime-runs",
        "run_kind": "eval",
        "framework": "verifiers",
        "step": "4",
    }
    assert "run_id" not in kwargs["context"]


def test_the_line_format_is_inferred_from_the_records():
    client = FakeTracesClient()
    sink = make_sink(client)

    sink.write([make_trace()])
    sink.write([make_episode()])

    assert client.calls[0][1]["line_format"] is LineFormat.TRACE
    assert client.calls[1][1]["line_format"] is LineFormat.EPISODE


def test_an_explicit_line_format_wins():
    client = FakeTracesClient()
    sink = make_sink(client)

    sink.write([{"id": "t1"}], line_format="episode")

    assert client.calls[0][1]["line_format"] is LineFormat.EPISODE


def test_a_bare_mapping_gets_the_run_stamped_onto_a_copy():
    """A dict has no stamping convention, and an upload with no ``run.id`` is
    orphaned — unqueryable and undeletable by run."""
    client = FakeTracesClient()
    sink = make_sink(client)
    original = {"id": "t1"}

    sink.write([original])

    assert client.calls[0][0][0]["run"] == {"id": "run-1", "type": "eval"}
    assert original == {"id": "t1"}, "the caller's dict was not mutated"


def test_producer_objects_are_passed_through_untouched():
    """Verifiers and prime-rl stamp the run at rollout time; rewriting their
    objects here is how a second source of truth appears."""
    client = FakeTracesClient()
    sink = make_sink(client)
    trace = make_trace()

    sink.write([trace])

    assert client.calls[0][0][0] is trace


def test_a_gated_account_disables_the_sink_instead_of_failing_the_run(caplog):
    """Prime Traces is in closed beta; no runtime action fixes a 403, so
    retrying it for the rest of the run only produces noise."""
    client = FakeTracesClient(
        raises=ForbiddenError("not in beta", status_code=403, code="service_not_enabled")
    )
    sink = make_sink(client)

    with caplog.at_level("WARNING"):
        sink.write([{"id": "t1"}])

    assert sink.enabled is False
    assert "not enabled" in caplog.text


def test_a_transient_failure_is_raised_so_the_worker_can_report_it():
    """Unlike a 403, this one is about the moment, not the account."""
    client = FakeTracesClient(raises=RetryableAPIError("busy", status_code=503))
    sink = make_sink(client)

    with pytest.raises(RetryableAPIError):
        sink.write([{"id": "t1"}])
    assert sink.enabled is True


def test_a_disabled_sink_stops_calling_the_service():
    client = FakeTracesClient()
    sink = make_sink(client)
    sink.enabled = False

    sink.write([{"id": "t1"}])

    assert client.calls == []


def test_receipt_history_is_bounded_while_the_total_is_retained():
    client = FakeTracesClient()
    sink = make_sink(client, receipt_history_size=2)

    for index in range(5):
        sink.write([{"id": f"t{index}"}])

    assert sink.receipts_received == 5
    assert len(sink.receipts) == 2


def test_closing_the_sink_closes_the_client():
    client = FakeTracesClient()
    sink = make_sink(client)

    sink.close()

    assert client.closed is True


def test_a_missing_traces_client_disables_the_sink_rather_than_raising(monkeypatch, caplog):
    """Construction failures must not take down a run that has not started."""

    def explode(**kwargs):
        raise RuntimeError("no credentials")

    monkeypatch.setattr("prime_traces.TracesClient", explode)
    sink = TracesSink()

    with caplog.at_level("WARNING"):
        sink.start("run-1", {"run_kind": "eval"})

    assert sink.enabled is False
