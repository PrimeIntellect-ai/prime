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

    sink.write([{"id": "t1", "run": {"id": "run-1"}}])

    _, kwargs = client.calls[0]
    assert kwargs["context"] == {
        "source": "prime-runs",
        "run_kind": "eval",
        "framework": "verifiers",
    }
    assert "run_id" not in kwargs["context"]


def test_the_line_format_is_inferred_from_the_records():
    client = FakeTracesClient()
    sink = make_sink(client)

    sink.write([make_trace()])
    sink.write([make_episode()])
    sink.write([{"id": "t", "traces": []}])

    assert client.calls[0][1]["line_format"] is LineFormat.TRACE
    assert client.calls[1][1]["line_format"] is LineFormat.EPISODE
    assert client.calls[2][1]["line_format"] is LineFormat.EPISODE


def test_a_bare_mapping_gets_the_run_stamped_onto_a_copy():
    """A dict has no stamping convention, and an upload with no ``run.id`` is
    orphaned — unqueryable and undeletable by run."""
    client = FakeTracesClient()
    sink = make_sink(client)
    original = {"id": "t1"}

    sink.write([original])

    assert client.calls[0][0][0]["run"] == {"id": "run-1", "type": "eval"}
    assert original == {"id": "t1"}, "the caller's dict was not mutated"


def test_a_record_that_names_its_run_keeps_it():
    """Producers stamp the run at rollout time; the sink never overrides a
    ``run`` that is already there, which is how a second source of truth
    would appear."""
    client = FakeTracesClient()
    sink = make_sink(client)

    sink.write([{"id": "t1", "run": {"id": "theirs", "type": "eval"}}])

    assert client.calls[0][0][0]["run"] == {"id": "theirs", "type": "eval"}


def test_an_episode_s_run_reaches_every_member_trace():
    """The traces service derives ``run_id`` from ``trace.run.id`` only and never
    reads the envelope's ``run``, while verifiers records the run on the episode
    and its ``Trace`` has no ``run`` field at all. Without this, every row of an
    episode upload lands with an empty ``run_id`` (seen live, 2026-08-21)."""
    client = FakeTracesClient()
    sink = make_sink(client)
    episode = make_episode("ep-1", [make_trace(trace_id="a"), make_trace(trace_id="b")])

    sink.write([episode])

    sent = client.calls[0][0][0]
    assert sent["run"] == {"id": "run-1", "type": "eval"}
    assert [member["run"] for member in sent["traces"]] == [sent["run"], sent["run"]]
    assert [member["id"] for member in sent["traces"]] == ["a", "b"]


def test_a_member_that_names_its_own_run_is_left_alone():
    client = FakeTracesClient()
    sink = make_sink(client)
    theirs = {"id": "other-run", "type": "eval"}
    episode = {
        "id": "ep-1",
        "run": {"id": "env-run", "type": "eval"},
        "traces": [{"id": "a"}, {"id": "b", "run": theirs}],
    }

    sink.write([episode])

    members = client.calls[0][0][0]["traces"]
    assert members[0]["run"] == {"id": "env-run", "type": "eval"}
    assert members[1]["run"] == theirs
    assert episode["traces"][0] == {"id": "a"}, "the caller's members were not mutated"


def test_producer_objects_are_serialized_once_through_to_record():
    """The sink calls ``to_record()`` itself so members are reachable; the bytes
    the transport sees are the same ones it would have produced."""
    client = FakeTracesClient()
    sink = make_sink(client)
    trace = make_trace()

    sink.write([trace])

    sent = client.calls[0][0][0]
    assert sent == {**trace.to_record(), "run": {"id": "run-1", "type": "eval"}}


def test_an_account_outside_the_beta_retires_the_sink_without_a_failure(caplog):
    """Prime Traces is in closed beta. For everyone outside it there was never
    anywhere for these records to go, so the sink turns itself off quietly:
    no exception for the worker to count, nothing above INFO in the log."""
    client = FakeTracesClient(
        raises=ForbiddenError("not in beta", status_code=403, code="service_not_enabled")
    )
    sink = make_sink(client)

    with caplog.at_level("INFO"):
        sink.write([{"id": "t1"}])
        sink.write([{"id": "t2"}])

    assert sink.enabled is False
    assert len(client.calls) == 1
    assert "not enabled" in caplog.text
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_a_credential_without_the_traces_scope_is_still_a_failure(caplog):
    """The other 403 is fixable — mint a token with the scope — so it is raised
    for loss accounting and strict callers, and the sink still retires."""
    client = FakeTracesClient(
        raises=ForbiddenError("missing scope: traces", status_code=403, code="forbidden")
    )
    sink = make_sink(client)

    with caplog.at_level("WARNING"):
        with pytest.raises(ForbiddenError, match="missing scope"):
            sink.write([{"id": "t1"}])

    assert sink.enabled is False
    assert "cannot write traces" in caplog.text


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


def test_a_missing_traces_client_disables_the_sink_and_reports_the_failure(monkeypatch, caplog):
    """The run applies warn/raise policy, so the sink must surface this failure."""

    def explode(**kwargs):
        raise RuntimeError("no credentials")

    monkeypatch.setattr("prime_traces.TracesClient", explode)
    sink = TracesSink()

    with caplog.at_level("WARNING"):
        with pytest.raises(RuntimeError, match="no credentials"):
            sink.start("run-1", {"run_kind": "eval"})

    assert sink.enabled is False
