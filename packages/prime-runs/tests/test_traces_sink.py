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
    sink.start("run-1", {"source": "prime-runs", "run_type": "eval", "framework": "verifiers"})
    return sink


def test_records_go_out_with_provenance_but_not_the_join_key():
    client = FakeTracesClient()
    sink = make_sink(client)

    sink.write([{"id": "t1", "run": {"id": "run-1"}}])

    _, kwargs = client.calls[0]
    assert kwargs["context"] == {
        "source": "prime-runs",
        "run_type": "eval",
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
    client = FakeTracesClient()
    sink = make_sink(client)
    original = {"id": "t1"}

    sink.write([original])

    assert client.calls[0][0][0]["run"] == {"id": "run-1", "type": "eval"}
    assert original == {"id": "t1"}, "the caller's dict was not mutated"


def test_a_record_that_names_its_own_run_is_rekeyed_to_this_one():
    client = FakeTracesClient()
    sink = make_sink(client)
    original = {"id": "t1", "run": {"id": "theirs", "type": "train", "name": "local"}}

    sink.write([original])

    assert client.calls[0][0][0]["run"] == {"id": "run-1", "type": "eval", "name": "local"}
    assert original["run"] == {"id": "theirs", "type": "train", "name": "local"}, "not mutated"


def test_a_training_episode_keeps_its_dispatch_step_while_being_rekeyed():
    """prime-rl stamps ``TrainRunInfo(id=$PRL_RUN_ID, work=...)`` on every
    dispatched episode; the platform's id replaces the launcher's, the work
    (the dispatch step) stays."""
    client = FakeTracesClient()
    sink = TracesSink(client=client)
    sink.start("run-1", {"run_type": "train"})
    work = {"type": "train", "step": 10}
    episode = {"id": "ep-1", "run": {"id": "prl-run", "type": "train", "work": work}, "traces": []}

    sink.write([episode])

    assert client.calls[0][0][0]["run"] == {"id": "run-1", "type": "train", "work": work}


def test_an_episode_s_run_reaches_every_member_trace():
    client = FakeTracesClient()
    sink = make_sink(client)
    episode = make_episode("ep-1", [make_trace(trace_id="a"), make_trace(trace_id="b")])

    sink.write([episode])

    sent = client.calls[0][0][0]
    assert sent["run"] == {"id": "run-1", "type": "eval"}
    assert [member["run"] for member in sent["traces"]] == [sent["run"], sent["run"]]
    assert [member["id"] for member in sent["traces"]] == ["a", "b"]


def test_every_member_trace_is_rekeyed_too():
    client = FakeTracesClient()
    sink = make_sink(client)
    theirs = {"id": "other-run", "type": "eval", "name": "other"}
    episode = {
        "id": "ep-1",
        "run": {"id": "env-run", "type": "eval", "name": "local"},
        "traces": [{"id": "a"}, {"id": "b", "run": theirs}],
    }

    sink.write([episode])

    sent = client.calls[0][0][0]
    assert sent["run"] == {"id": "run-1", "type": "eval", "name": "local"}
    assert sent["traces"][0]["run"] == sent["run"]
    assert sent["traces"][1]["run"] == {"id": "run-1", "type": "eval", "name": "other"}
    assert episode["traces"][0] == {"id": "a"}, "the caller's members were not mutated"
    assert theirs == {"id": "other-run", "type": "eval", "name": "other"}


def test_an_account_outside_the_beta_retires_the_sink_without_a_failure(caplog):
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
    client = FakeTracesClient(raises=RetryableAPIError("busy", status_code=503))
    sink = make_sink(client)

    with pytest.raises(RetryableAPIError):
        sink.write([{"id": "t1"}])
    assert sink.enabled is True


def test_receipt_history_is_bounded_while_the_total_is_retained():
    client = FakeTracesClient()
    sink = make_sink(client, receipt_history_size=2)

    for index in range(5):
        sink.write([{"id": f"t{index}"}])

    assert sink.receipts_received == 5
    assert len(sink.receipts) == 2


def test_a_missing_traces_client_disables_the_sink_and_reports_the_failure(monkeypatch, caplog):
    def explode(**kwargs):
        raise RuntimeError("no credentials")

    monkeypatch.setattr("prime_runs.sinks.traces.TracesClient", explode)
    sink = TracesSink()

    with caplog.at_level("WARNING"):
        with pytest.raises(RuntimeError, match="no credentials"):
            sink.start("run-1", {"run_type": "eval"})

    assert sink.enabled is False
