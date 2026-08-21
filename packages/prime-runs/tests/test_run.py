"""The run handle: lifecycle, containment, terminal status."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import pytest
from conftest import FakeSink
from prime_traces.exceptions import ForbiddenError

from prime_runs.exceptions import RunFinishedError
from prime_runs.models import RunHandle, RunSpec, RunStatus
from prime_runs.run import Run
from prime_runs.sinks import TracesSink


class FakeBackend:
    def __init__(self, fail_on: Optional[str] = None) -> None:
        self.fail_on = fail_on
        self.updates: List[Dict[str, Any]] = []
        self.finalized: List[Dict[str, Any]] = []
        self.closed = False

    def create(self, spec: RunSpec) -> RunHandle:
        return RunHandle(id="run-1", name=spec.name, url="https://app.example/run-1")

    def update(self, run_id, *, config=None, summary=None) -> None:
        if self.fail_on == "update":
            raise RuntimeError("update exploded")
        self.updates.append({"config": config, "summary": summary})

    def finalize(self, run_id, *, status, summary=None, error=None, config=None) -> None:
        if self.fail_on == "finalize":
            raise RuntimeError("finalize exploded")
        self.finalized.append(
            {"status": status, "summary": summary, "error": error, "config": config}
        )

    def close(self) -> None:
        self.closed = True


def make_run(backend=None, sinks=None, config=None, **kwargs) -> Run:
    backend = backend or FakeBackend()
    spec = RunSpec(name="test-run", framework="verifiers", model="Qwen3-8B", config=config or {})
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

    run.log_traces([{"id": "t1"}])
    run.flush()

    assert sink.batches == [[{"id": "t1"}]]
    assert not run.finished
    run.finish()


def test_an_empty_batch_is_not_sent():
    sink = FakeSink()
    run = make_run(sinks=[sink])

    run.log_traces([])
    run.flush()

    assert sink.batches == []
    run.finish()


def test_non_finite_summary_values_are_dropped_rather_than_failing_the_request():
    """A diverged loss serializes as bare ``NaN``, which strict JSON rejects —
    the whole request fails on a payload nobody can inspect."""
    backend = FakeBackend()
    run = make_run(backend)

    run.finish(summary={"loss": float("nan"), "grad": float("inf"), "reward": 0.5})

    assert run.summary == {"reward": 0.5}
    assert backend.finalized[0]["summary"] == {"reward": 0.5}


def test_finish_flushes_records_before_reporting_the_terminal_status():
    """A dashboard reacting to the status must never see a finished run with
    samples still landing."""
    order: List[str] = []

    class OrderedSink(FakeSink):
        def write(self, records) -> None:
            order.append("write")
            super().write(records)

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


def test_concurrent_finish_waits_for_the_first_teardown_to_complete():
    class BlockingBackend(FakeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.finalize_started = threading.Event()
            self.release_finalize = threading.Event()

        def finalize(self, run_id, *, status, summary=None, error=None, config=None) -> None:
            self.finalize_started.set()
            assert self.release_finalize.wait(2.0)
            super().finalize(run_id, status=status, summary=summary, error=error, config=config)

    backend = BlockingBackend()
    run = make_run(backend)
    second_started = threading.Event()

    def finish_second() -> None:
        second_started.set()
        run.finish(status=RunStatus.FAILED)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(run.finish)
        assert backend.finalize_started.wait(2.0)
        assert run.finished is False

        second = executor.submit(finish_second)
        assert second_started.wait(2.0)
        assert second.done() is False

        backend.release_finalize.set()
        first.result(timeout=2.0)
        second.result(timeout=2.0)

    assert run.finished is True
    assert [entry["status"] for entry in backend.finalized] == [RunStatus.COMPLETED]


@pytest.mark.parametrize("status", [RunStatus.RUNNING, "running"])
def test_finish_rejects_a_nonterminal_status_without_closing_the_run(status):
    backend = FakeBackend()
    run = make_run(backend)

    with pytest.raises(ValueError, match="requires a terminal status"):
        run.finish(status=status)

    assert not run.finished
    assert backend.finalized == []

    run.finish()
    assert backend.finalized[0]["status"] is RunStatus.COMPLETED


def test_logging_after_finish_is_a_producer_bug():
    run = make_run()
    run.finish()

    with pytest.raises(RunFinishedError):
        run.log_traces([{"id": "t1"}])


def test_the_context_manager_completes_a_clean_run():
    backend = FakeBackend()
    with make_run(backend) as run:
        run.log_traces([{"id": "t1"}])

    assert backend.finalized[0]["status"] is RunStatus.COMPLETED


def test_an_exception_inside_the_block_fails_the_run_and_still_propagates():
    backend = FakeBackend()

    with pytest.raises(ValueError):
        with make_run(backend):
            raise ValueError("rollout blew up")

    assert backend.finalized[0]["status"] is RunStatus.FAILED
    assert "rollout blew up" in backend.finalized[0]["error"]


def test_a_finish_failure_does_not_mask_the_context_exception():
    backend = FakeBackend(fail_on="finalize")

    with pytest.raises(ValueError, match="rollout blew up"):
        with make_run(backend, on_error="raise"):
            raise ValueError("rollout blew up")

    assert backend.closed is True


@pytest.mark.parametrize("teardown_error", [KeyboardInterrupt, SystemExit])
def test_teardown_does_not_swallow_control_flow_exceptions(teardown_error):
    run = make_run()
    original_finish = run.finish

    def interrupt_finish(*args, **kwargs):
        raise teardown_error()

    run.finish = interrupt_finish
    try:
        with pytest.raises(teardown_error):
            with run:
                raise ValueError("rollout blew up")
    finally:
        run.finish = original_finish
        run.finish()


def test_an_interrupt_is_recorded_as_a_decision_not_a_fault():
    """Ctrl-C must not land in the same bucket as a broken eval."""
    backend = FakeBackend()

    with pytest.raises(KeyboardInterrupt):
        with make_run(backend):
            raise KeyboardInterrupt

    assert backend.finalized[0]["status"] is RunStatus.CRASHED
    assert backend.finalized[0]["error"] == "interrupted"


def test_finish_hands_the_full_config_to_finalize():
    """The evaluations API replaces metadata wholesale, so a backend recording
    terminal state inside it needs the whole picture to merge into."""
    backend = FakeBackend()
    run = make_run(backend, config={"num_rollouts": 4})

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


def test_finish_uses_one_timeout_budget_for_flush_and_close():
    run = make_run()
    run._finish_timeout = 0.1
    observed = {}

    def slow_flush(timeout=None):
        observed["flush"] = timeout
        time.sleep(0.02)
        return False

    def record_close(timeout=None):
        observed["close"] = timeout

    run._worker.flush = slow_flush
    run._worker.close = record_close

    run.finish()

    assert 0.0 <= observed["close"] < observed["flush"] <= run._finish_timeout


def test_a_process_that_exits_without_finishing_reports_crashed():
    """The producer never said the run failed — it stopped existing. The
    distinction tells an operator where to look."""
    backend = FakeBackend()
    run = make_run(backend)

    run._on_process_exit()

    assert backend.finalized[0]["status"] is RunStatus.CRASHED


def test_a_forked_handle_gets_a_fresh_lock_and_loses_lifecycle_ownership():
    """The child's inherited atexit hook must never finalize the parent's run."""
    backend = FakeBackend()
    run = make_run(backend)
    inherited_lock = run._finish_lock

    run.reset_after_fork()

    assert run._finish_lock is not inherited_lock
    run.finish()
    assert backend.finalized == []
    assert backend.closed is True


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

    assert backend.closed is True


def test_strict_sink_start_failure_closes_the_created_run():
    class StartFailingSink(FakeSink):
        def start(self, run_id, context) -> None:
            raise RuntimeError("could not start sink")

    backend = FakeBackend()
    sink = StartFailingSink()

    with pytest.raises(RuntimeError, match="could not start sink"):
        make_run(backend, sinks=[sink], on_error="raise")

    assert backend.finalized == [
        {
            "status": RunStatus.FAILED,
            "summary": None,
            "error": "RuntimeError: could not start sink",
            "config": None,
        }
    ]
    assert backend.closed is True
    assert sink.closed is True


def test_update_failure_in_raise_mode_still_finalizes_and_closes():
    backend = FakeBackend(fail_on="update")
    run = make_run(backend, on_error="raise")

    with pytest.raises(RuntimeError, match="update exploded"):
        run.finish()

    assert len(backend.finalized) == 1
    assert backend.closed is True


def test_a_sink_error_is_recorded_on_the_run():
    run = make_run(sinks=[FakeSink("broken", fail_on_write=True)])

    run.log_traces([{"id": "t1"}])
    run.flush()

    assert any("broken" in error for error in run.errors)
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


class _ForbiddenClient:
    def __init__(self, code: str) -> None:
        self.code = code

    def upload_records(self, records, **kwargs):
        raise ForbiddenError("403", status_code=403, code=self.code)

    def close(self) -> None:
        pass


def test_an_account_outside_the_beta_is_not_a_failed_run():
    """Nothing was lost — the records went to every sink that applies to this
    account — so the run finishes clean even under ``on_error="raise"``."""
    sink = TracesSink(client=_ForbiddenClient("service_not_enabled"))
    run = make_run(sinks=[sink], on_error="raise")
    run.log_traces([{"id": "t1"}])

    run.finish()

    assert run.failed_records == {}
    assert run.errors == []
    assert sink.enabled is False


def test_a_credential_without_the_traces_scope_reaches_strict_callers_and_loss_accounting():
    run = make_run(sinks=[TracesSink(client=_ForbiddenClient("forbidden"))], on_error="raise")
    run.log_traces([{"id": "t1"}])

    with pytest.raises(ForbiddenError):
        run.finish()

    assert run.failed_records == {"traces": 1}
