"""The run handle: lifecycle, containment, ranks, terminal status."""

import signal
from typing import Any, Dict, List, Optional

import pytest
from conftest import FakeSink

from prime_runs.exceptions import RunFinishedError
from prime_runs.models import RunHandle, RunSpec, RunStatus
from prime_runs.run import Run


class FakeBackend:
    def __init__(self, supports_step_metrics: bool = False, fail_on: Optional[str] = None) -> None:
        self.kind = "eval"
        self.supports_step_metrics = supports_step_metrics
        self.fail_on = fail_on
        self.updates: List[Dict[str, Any]] = []
        self.points: List[Any] = []
        self.finalized: List[Dict[str, Any]] = []
        self.closed = False

    def create(self, spec: RunSpec) -> RunHandle:
        return RunHandle(id="run-1", name=spec.name, url="https://app.example/run-1")

    def attach(self, run_id: str) -> RunHandle:
        return RunHandle(id=run_id)

    def update(self, run_id, *, config=None, summary=None) -> None:
        if self.fail_on == "update":
            raise RuntimeError("update exploded")
        self.updates.append({"config": config, "summary": summary})

    def log_metrics(self, run_id, metrics, step=None) -> None:
        self.points.append((metrics, step))

    def finalize(self, run_id, *, status, summary=None, error=None, config=None) -> None:
        if self.fail_on == "finalize":
            raise RuntimeError("finalize exploded")
        self.finalized.append(
            {"status": status, "summary": summary, "error": error, "config": config}
        )

    def close(self) -> None:
        self.closed = True


def make_run(backend=None, sinks=None, **kwargs) -> Run:
    backend = backend or FakeBackend()
    spec = RunSpec(name="test-run", kind="eval", framework="verifiers", model="Qwen3-8B")
    return Run(
        backend=backend,
        handle=backend.create(spec),
        spec=spec,
        sinks=sinks if sinks is not None else [],
        **kwargs,
    )


def test_the_handle_exposes_what_a_producer_prints():
    run = make_run()

    assert run.id == "run-1"
    assert run.url == "https://app.example/run-1"
    assert run.status is RunStatus.RUNNING
    assert run.is_primary is True
    run.finish()


def test_sinks_are_started_with_the_run_id_and_provenance():
    sink = FakeSink()
    run = make_run(sinks=[sink])

    run_id, context = sink.started[0]

    assert run_id == "run-1"
    # Provenance only — the join key is run.id inside the trace document.
    assert context["source"] == "prime-runs"
    assert context["framework"] == "verifiers"
    assert "evaluation_id" not in context
    run.finish()


def test_traces_reach_the_sinks_while_the_run_is_still_going():
    sink = FakeSink()
    run = make_run(sinks=[sink])

    run.log_traces([{"id": "t1"}], step=2)
    run.flush()

    assert sink.batches[0][0] == [{"id": "t1"}]
    assert sink.batches[0][2] == 2
    assert not run.finished
    run.finish()


def test_an_empty_batch_is_not_sent():
    sink = FakeSink()
    run = make_run(sinks=[sink])

    run.log_traces([])
    run.flush()

    assert sink.batches == []
    run.finish()


def test_metrics_land_in_the_summary_when_the_backend_has_no_time_series():
    backend = FakeBackend(supports_step_metrics=False)
    run = make_run(backend, summary_flush_seconds=0.0)

    run.log({"reward": 0.5}, step=1)
    run.log({"reward": 0.75}, step=2)

    assert run.summary["reward"] == 0.75
    assert backend.points == []
    assert backend.updates, "the summary was flushed"
    run.finish()


def test_metrics_become_a_time_series_when_the_backend_has_one():
    backend = FakeBackend(supports_step_metrics=True)
    run = make_run(backend)

    run.log({"loss": 2.0}, step=1)
    run.flush()

    assert backend.points == [({"loss": 2.0}, 1)]
    run.finish()


def test_commit_false_stages_without_writing():
    backend = FakeBackend(supports_step_metrics=True)
    run = make_run(backend)

    run.log({"loss": 2.0}, step=1, commit=False)
    run.flush()

    assert backend.points == []
    assert run.summary["loss"] == 2.0
    run.finish()


def test_non_finite_metrics_are_dropped_rather_than_failing_the_request():
    """A diverged loss serializes as bare ``NaN``, which strict JSON rejects —
    the whole request fails on a payload nobody can inspect."""
    run = make_run(summary_flush_seconds=0.0)

    run.log({"loss": float("nan"), "grad": float("inf"), "reward": 0.5})

    assert run.summary == {"reward": 0.5}
    run.finish()


def test_finish_flushes_records_before_reporting_the_terminal_status():
    """A dashboard reacting to the status must never see a finished run with
    samples still landing."""
    order: List[str] = []

    class OrderedSink(FakeSink):
        def write(self, records, *, line_format=None, step=None) -> None:
            order.append("write")
            super().write(records, line_format=line_format, step=step)

    class OrderedBackend(FakeBackend):
        def finalize(self, run_id, *, status, summary=None, error=None, config=None) -> None:
            order.append("finalize")
            super().finalize(run_id, status=status, summary=summary, error=error, config=config)

    run = make_run(OrderedBackend(), sinks=[OrderedSink()])
    run.log_traces([{"id": "t1"}])
    run.finish()

    assert order == ["write", "finalize"]


def test_finish_is_idempotent():
    backend = FakeBackend()
    run = make_run(backend)

    run.finish(summary={"avg_reward": 1.0})
    run.finish(status=RunStatus.FAILED)

    assert len(backend.finalized) == 1
    assert backend.finalized[0]["status"] is RunStatus.COMPLETED
    assert backend.finalized[0]["summary"] == {"avg_reward": 1.0}


def test_logging_after_finish_is_a_producer_bug():
    run = make_run()
    run.finish()

    with pytest.raises(RunFinishedError):
        run.log({"reward": 1.0})
    with pytest.raises(RunFinishedError):
        run.log_traces([{"id": "t1"}])


def test_the_context_manager_completes_a_clean_run():
    backend = FakeBackend()
    with make_run(backend) as run:
        run.log({"reward": 1.0})

    assert backend.finalized[0]["status"] is RunStatus.COMPLETED


def test_an_exception_inside_the_block_fails_the_run_and_still_propagates():
    backend = FakeBackend()

    with pytest.raises(ValueError):
        with make_run(backend):
            raise ValueError("rollout blew up")

    assert backend.finalized[0]["status"] is RunStatus.FAILED
    assert "rollout blew up" in backend.finalized[0]["error"]


def test_an_interrupt_is_recorded_as_a_decision_not_a_fault():
    """Ctrl-C must not land in the same bucket as a broken eval — and it must
    agree with the SIGINT handler, which normally gets there first."""
    backend = FakeBackend()

    with pytest.raises(KeyboardInterrupt):
        with make_run(backend):
            raise KeyboardInterrupt

    assert backend.finalized[0]["status"] is RunStatus.CRASHED
    assert backend.finalized[0]["error"] == "interrupted"


def test_a_termination_signal_reports_crashed_like_atexit_does():
    """The producer never said the run failed; it was stopped from outside."""
    backend = FakeBackend()
    run = make_run(backend)
    chained = []
    # Stand in for the handler the SDK displaced. Anything but SIG_DFL, which
    # would re-raise the signal and take the test runner down with it.
    run._previous_signal_handlers[signal.SIGTERM] = lambda *a: chained.append(a)

    run._handle_signal(signal.SIGTERM, None)

    assert backend.finalized[0]["status"] is RunStatus.CRASHED
    assert "SIGTERM" in backend.finalized[0]["error"]
    assert chained, "the displaced handler still runs"
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


def test_finish_hands_the_full_config_to_finalize():
    """The evaluations API replaces metadata wholesale, so a backend recording
    terminal state inside it needs the whole picture to merge into."""
    backend = FakeBackend()
    run = make_run(backend)
    run.update_config({"num_rollouts": 4})

    run.finish(status=RunStatus.FAILED, error="boom")

    assert backend.finalized[0]["config"]["num_rollouts"] == 4


def test_finish_warns_when_uploads_do_not_drain(caplog):
    """Finalizing over an unfinished upload silently drops records."""
    run = make_run(sinks=[FakeSink()])
    run._finish_timeout = 0.01
    run._worker.flush = lambda timeout=None: False

    with caplog.at_level("WARNING"):
        run.finish()

    assert "did not drain" in caplog.text


def test_a_process_that_exits_without_finishing_reports_crashed():
    """The producer never said the run failed — it stopped existing. The
    distinction tells an operator where to look."""
    backend = FakeBackend()
    run = make_run(backend)

    run._on_process_exit()

    assert backend.finalized[0]["status"] is RunStatus.CRASHED


def test_a_backend_failure_does_not_escape_into_the_producer_by_default():
    """Six hours of rollouts must not be lost to a 502 on a telemetry call."""
    backend = FakeBackend(fail_on="finalize")
    run = make_run(backend)

    run.finish()

    assert run.errors and "finalize exploded" in run.errors[0]


def test_on_error_raise_surfaces_the_failure_for_tests_and_ci():
    backend = FakeBackend(fail_on="finalize")
    run = make_run(backend, on_error="raise")

    with pytest.raises(RuntimeError, match="finalize exploded"):
        run.finish()


def test_a_sink_error_is_recorded_on_the_run():
    run = make_run(sinks=[FakeSink("broken", fail_on_write=True)])

    run.log_traces([{"id": "t1"}])
    run.flush()

    assert any("broken" in error for error in run.errors)
    run.finish()


def test_a_non_primary_rank_does_not_close_the_shared_run():
    """Eight ranks racing to finalize produce seven confusing failures."""
    backend = FakeBackend()
    run = make_run(backend, is_primary=False)

    run.log({"reward": 1.0})
    run.finish()

    assert backend.finalized == []
    assert backend.updates == []
    assert run.is_primary is False


def test_a_non_primary_rank_still_uploads_its_own_records():
    """The point of eight ranks is that they contribute to one run."""
    sink = FakeSink()
    run = make_run(sinks=[sink], is_primary=False)

    run.log_traces([{"id": "t1"}])
    run.flush()

    assert sink.batches
    run.finish()


def test_dropped_records_are_reported_on_the_handle():
    run = make_run()
    run._worker.dropped = 3

    assert run.dropped_records == 3
    run.finish()


def test_an_upload_failure_reaches_the_caller_in_raise_mode():
    """A sink fails on the uploader thread, where raising reaches nobody. Under
    on_error="raise" the failure has to surface where a test is looking."""
    run = make_run(sinks=[FakeSink("broken", fail_on_write=True)], on_error="raise")

    run.log_traces([{"id": "t1"}])
    with pytest.raises(RuntimeError, match="sink is broken"):
        run.flush()

    run.finish()


def test_an_upload_failure_surfaces_from_finish_too():
    run = make_run(sinks=[FakeSink("broken", fail_on_write=True)], on_error="raise")
    run.log_traces([{"id": "t1"}])

    with pytest.raises(RuntimeError, match="sink is broken"):
        run.finish()

    # Still closed out: the failure is reported after teardown, not instead of it.
    assert run.finished


def test_an_upload_failure_is_reported_once():
    run = make_run(sinks=[FakeSink("broken", fail_on_write=True)], on_error="raise")
    run.log_traces([{"id": "t1"}])

    with pytest.raises(RuntimeError):
        run.flush()
    run.flush()  # nothing left to raise

    run.finish()


def test_signal_handlers_are_restored_when_the_run_finishes():
    """`self._handle_signal` builds a new bound method on every access, so an
    identity check against a fresh one never matches — leaving the handler
    installed, pinning the finished run, and blocking the next run in the
    process from installing its own."""
    original = signal.getsignal(signal.SIGTERM)
    run = make_run()
    run.install_signal_handlers()
    assert signal.getsignal(signal.SIGTERM) is run._signal_handler

    run.finish()

    assert signal.getsignal(signal.SIGTERM) is original


def test_a_later_run_can_install_its_own_handlers():
    first = make_run()
    first.install_signal_handlers()
    first.finish()

    second = make_run()
    second.install_signal_handlers()

    assert signal.getsignal(signal.SIGTERM) is second._signal_handler
    second.finish()
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
