"""The run handle: lifecycle, containment, terminal status."""

import asyncio
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


def test_sinks_are_started_with_the_run_id_and_provenance():
    sink = FakeSink()
    run = make_run(sinks=[sink])

    run_id, context = sink.started[0]

    assert run_id == "run-1"
    # Provenance only — the join key is run.id inside the trace document.
    assert context["source"] == "prime-runs"
    assert context["run_type"] == "eval"
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


def test_summary_can_be_built_up_before_finish():
    backend = FakeBackend()
    run = make_run(backend)

    run.update_summary({"avg_reward": 0.5, "loss": float("nan")})
    run.update_summary({"avg_error": 0.0})
    run.finish(summary={"avg_reward": 0.75})

    assert run.summary == {"avg_reward": 0.75, "avg_error": 0.0}
    assert backend.finalized[0]["summary"] == {"avg_reward": 0.75, "avg_error": 0.0}


def test_non_finite_summary_values_are_dropped_rather_than_failing_the_request():
    backend = FakeBackend()
    run = make_run(backend)

    run.finish(summary={"loss": float("nan"), "grad": float("inf"), "reward": 0.5})

    assert run.summary == {"reward": 0.5}
    assert backend.finalized[0]["summary"] == {"reward": 0.5}


def test_finish_flushes_records_before_reporting_the_terminal_status():
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


@pytest.mark.parametrize(
    "operation",
    [
        lambda run: run.log_traces([{"id": "t1"}]),
        lambda run: run.log_episodes([{"id": "ep1", "traces": []}]),
        lambda run: run.log_metrics({"loss": 0.5}),
        lambda run: run.update_summary({"late": 1.0}),
    ],
    ids=["traces", "episodes", "metrics", "summary"],
)
def test_writes_after_finish_are_a_producer_bug(operation):
    run = make_run(metrics_sinks=[FakeSink("metrics")])
    run.finish()

    with pytest.raises(RunFinishedError):
        operation(run)


def test_logging_that_started_first_is_queued_before_concurrent_finish():
    class SignalingBackend(FakeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.finalize_started = threading.Event()

        def finalize(self, run_id, *, status, summary=None, error=None, config=None) -> None:
            self.finalize_started.set()
            super().finalize(
                run_id,
                status=status,
                summary=summary,
                error=error,
                config=config,
            )

    backend = SignalingBackend()
    sink = FakeSink()
    run = make_run(backend, sinks=[sink])
    submit_started = threading.Event()
    release_submit = threading.Event()
    finish_called = threading.Event()
    original_submit = run._worker.submit

    def blocking_submit(records):
        submit_started.set()
        assert release_submit.wait(2.0)
        return original_submit(records)

    run._worker.submit = blocking_submit

    def finish_run() -> None:
        finish_called.set()
        run.finish()

    with ThreadPoolExecutor(max_workers=2) as executor:
        logging = executor.submit(run.log_traces, [{"id": "t1"}])
        assert submit_started.wait(2.0)
        finishing = executor.submit(finish_run)
        assert finish_called.wait(2.0)
        try:
            assert not backend.finalize_started.wait(0.1)
        finally:
            release_submit.set()
        logging.result(timeout=2.0)
        finishing.result(timeout=2.0)

    assert sink.batches == [[{"id": "t1"}]]
    assert sink.closed is True


@pytest.mark.parametrize(
    "operation",
    [
        lambda run: run.log_traces([{"id": "t1"}]),
        lambda run: run.log_episodes([{"id": "ep1", "traces": []}]),
        lambda run: run.update_summary({"reward": 1.0}),
    ],
    ids=["traces", "episodes", "summary"],
)
def test_writes_during_finish_fail_without_waiting_for_teardown(operation):
    class BlockingBackend(FakeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.finalize_started = threading.Event()
            self.release_finalize = threading.Event()

        def finalize(self, run_id, *, status, summary=None, error=None, config=None) -> None:
            self.finalize_started.set()
            assert self.release_finalize.wait(2.0)
            super().finalize(
                run_id,
                status=status,
                summary=summary,
                error=error,
                config=config,
            )

    backend = BlockingBackend()
    run = make_run(backend)

    with ThreadPoolExecutor(max_workers=2) as executor:
        finishing = executor.submit(run.finish)
        assert backend.finalize_started.wait(2.0)
        writing = executor.submit(operation, run)
        try:
            with pytest.raises(RunFinishedError):
                writing.result(timeout=0.2)
        finally:
            backend.release_finalize.set()
        finishing.result(timeout=2.0)


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


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, asyncio.CancelledError])
def test_an_interrupt_is_recorded_as_a_decision_not_a_fault(interrupt):
    backend = FakeBackend()

    with pytest.raises(interrupt):
        with make_run(backend):
            raise interrupt

    assert backend.finalized[0]["status"] is RunStatus.CANCELLED
    assert backend.finalized[0]["error"] == "interrupted"


def test_cancelled_is_terminal_and_finalizes_like_failed():
    backend = FakeBackend()
    run = make_run(backend)

    assert RunStatus.CANCELLED.is_terminal()
    run.finish(status="cancelled", error="operator stopped it")

    assert run.status is RunStatus.CANCELLED
    assert backend.finalized[0]["status"] is RunStatus.CANCELLED
    assert backend.finalized[0]["error"] == "operator stopped it"


def test_finish_timeout_is_set_per_run_and_overridden_per_call():
    run = make_run(finish_timeout=1.5)
    observed = {}

    def record_flush(timeout=None):
        observed["flush"] = timeout
        return True

    run._worker.flush = record_flush
    run.finish(timeout=0.25)

    assert run._finish_timeout == 1.5
    assert 0.0 < observed["flush"] <= 0.25


def test_finish_hands_the_full_config_to_finalize():
    backend = FakeBackend()
    run = make_run(backend, config={"num_rollouts": 4})

    run.finish(status=RunStatus.FAILED, error="boom")

    assert backend.finalized[0]["config"]["num_rollouts"] == 4


def test_finish_warns_when_uploads_do_not_drain(caplog):
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
    backend = FakeBackend()
    run = make_run(backend)

    run._on_process_exit()

    assert backend.finalized[0]["status"] is RunStatus.CRASHED


def test_a_forked_handle_gets_a_fresh_lock_and_loses_lifecycle_ownership():
    backend = FakeBackend()
    run = make_run(backend)
    inherited_lock = run._finish_lock

    run.reset_after_fork()

    assert run._finish_lock is not inherited_lock
    run.finish()
    assert backend.finalized == []
    assert backend.closed is True


def test_a_backend_failure_does_not_escape_into_the_producer_by_default():
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


def test_an_upload_failure_reaches_the_caller_in_raise_mode():
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


# ------------------------------------------------------------------ metrics


def test_log_metrics_goes_to_the_metrics_sinks_only():
    records, metrics = FakeSink("records"), FakeSink("metrics")
    run = make_run(sinks=[records], metrics_sinks=[metrics])

    run.log_metrics({"loss": 0.5, "nan": float("nan")}, step=3)
    run.log_traces([{"id": "t1"}])
    run.flush()

    assert records.batches == [[{"id": "t1"}]]
    (batch,) = metrics.batches
    (row,) = batch
    assert row["loss"] == 0.5
    assert row["step"] == 3
    assert "nan" not in row
    assert isinstance(row["_timestamp"], float)
    run.finish()


def test_log_metrics_drops_non_finite_values_inside_sequences():
    metrics = FakeSink("metrics")
    run = make_run(metrics_sinks=[metrics])

    run.log_metrics(
        {
            "quantiles": [0.1, float("nan"), 0.9],
            "nested": [{"loss": float("inf"), "reward": 1.0}, [float("-inf"), 2.0]],
        }
    )
    run.flush()

    row = metrics.batches[0][0]
    assert row["quantiles"] == [0.1, 0.9]
    assert row["nested"] == [{"reward": 1.0}, [2.0]]
    run.finish()


def test_log_metrics_keeps_a_step_the_producer_already_set():
    metrics = FakeSink("metrics")
    run = make_run(metrics_sinks=[metrics])

    run.log_metrics({"step": 7, "loss": 0.5}, step=8)
    run.flush()

    assert metrics.batches[0][0]["step"] == 7
    run.finish()


def test_log_metrics_without_a_metrics_sink_is_a_no_op():
    run = make_run()

    run.log_metrics({"loss": 0.5}, step=1)

    assert run._metrics_worker._thread is None
    assert run.dropped_records == 0
    run.finish()


def test_finish_drains_and_closes_the_metrics_uploader_too():
    metrics = FakeSink("metrics")
    run = make_run(metrics_sinks=[metrics])

    run.log_metrics({"loss": 0.5}, step=1)
    run.finish()

    assert len(metrics.batches) == 1
    assert metrics.closed is True


def test_losses_are_reported_across_both_uploaders():
    broken = FakeSink("metrics", fail_on_write=True)
    run = make_run(metrics_sinks=[broken])

    run.log_metrics({"loss": 0.5}, step=1)
    run.flush()

    assert run.failed_records == {"metrics": 1}
    run.finish()
