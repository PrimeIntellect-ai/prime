"""The background uploader: backpressure, containment, fork safety."""

import queue
import threading

from conftest import FakeSink

from prime_runs.worker import MetricItem, RunUpdateItem, UploadWorker, WriteItem


class BlockingSink(FakeSink):
    def __init__(self) -> None:
        super().__init__("blocking")
        self.entered = threading.Event()
        self.released = threading.Event()

    def write(self, records, *, line_format=None, step=None) -> None:
        self.entered.set()
        self.released.wait(5.0)
        super().write(records, line_format=line_format, step=step)


def drain(worker: UploadWorker) -> None:
    assert worker.flush(timeout=5.0)


def test_records_reach_every_enabled_sink():
    sinks = [FakeSink("a"), FakeSink("b")]
    worker = UploadWorker(sinks)

    worker.submit(WriteItem(records=[{"id": 1}], line_format="trace", step=3))
    drain(worker)

    for sink in sinks:
        assert sink.batches == [([{"id": 1}], "trace", 3)]
    worker.close()


def test_a_disabled_sink_is_skipped():
    live, dead = FakeSink("live"), FakeSink("dead")
    dead.enabled = False
    worker = UploadWorker([live, dead])

    worker.submit(WriteItem(records=[{"id": 1}]))
    drain(worker)

    assert live.batches and not dead.batches
    worker.close()


def test_one_sink_failing_does_not_stop_the_others():
    broken, healthy = FakeSink("broken", fail_on_write=True), FakeSink("healthy")
    reported = []
    worker = UploadWorker([broken, healthy], on_error=lambda name, exc: reported.append(name))

    worker.submit(WriteItem(records=[{"id": 1}]))
    drain(worker)

    assert healthy.batches
    assert broken.enabled is False
    assert reported == ["broken"]
    worker.close()


def test_a_failed_sink_is_not_called_again():
    """One log line per batch for the rest of a run hides whatever failed first."""
    broken = FakeSink("broken", fail_on_write=True)
    reported = []
    worker = UploadWorker([broken], on_error=lambda name, exc: reported.append(name))

    for _ in range(3):
        worker.submit(WriteItem(records=[{"id": 1}]))
    drain(worker)

    assert reported == ["broken"]
    worker.close()


def test_a_full_queue_drops_rather_than_blocking_the_producer():
    """Stalling a training run to protect telemetry is the wrong trade."""
    sink = BlockingSink()
    worker = UploadWorker([sink], max_queue_size=1, put_timeout=0.05)

    # First item is picked up and wedges the uploader inside sink.write().
    assert worker.submit(WriteItem(records=[{"id": 0}]))
    assert sink.entered.wait(5.0), "the uploader never reached the sink"
    # Second fills the one-slot queue; third has nowhere to go.
    worker.submit(WriteItem(records=[{"id": 1}]))
    accepted = worker.submit(WriteItem(records=[{"id": 2}, {"id": 3}]))

    assert accepted is False
    assert worker.dropped == 2

    sink.released.set()
    worker.close()


def test_flush_uses_the_drain_budget_to_get_behind_a_full_queue():
    sink = BlockingSink()
    worker = UploadWorker([sink], max_queue_size=1, put_timeout=0.01)
    assert worker.submit(WriteItem(records=[{"id": 0}]))
    assert sink.entered.wait(1.0)
    assert worker.submit(WriteItem(records=[{"id": 1}]))

    release = threading.Timer(0.05, sink.released.set)
    release.start()
    try:
        assert worker.flush(timeout=0.5) is True
    finally:
        sink.released.set()
        release.join()
        worker.close(timeout=1.0)

    assert [batch[0][0]["id"] for batch in sink.batches] == [0, 1]


def test_close_uses_its_budget_to_queue_the_stop_behind_pending_records():
    sink = BlockingSink()
    worker = UploadWorker([sink], max_queue_size=1, put_timeout=0.01)
    assert worker.submit(WriteItem(records=[{"id": 0}]))
    assert sink.entered.wait(1.0)
    assert worker.submit(WriteItem(records=[{"id": 1}]))

    release = threading.Timer(0.05, sink.released.set)
    release.start()
    try:
        worker.close(timeout=0.5)
    finally:
        sink.released.set()
        release.join()

    assert [batch[0][0]["id"] for batch in sink.batches] == [0, 1]
    assert sink.closed is True


def test_metrics_ride_the_same_queue_when_the_backend_stores_a_time_series():
    points = []
    worker = UploadWorker([], metric_writer=lambda metrics, step: points.append((metrics, step)))

    worker.submit(MetricItem(metrics={"loss": 0.5}, step=7))
    drain(worker)

    assert points == [({"loss": 0.5}, 7)]
    worker.close()


def test_run_updates_ride_the_uploader_queue():
    updates = []
    worker = UploadWorker(
        [], update_writer=lambda config, summary: updates.append((config, summary))
    )

    worker.submit(RunUpdateItem(config={"seed": 7}, summary={"reward": 0.5}))
    drain(worker)

    assert updates == [({"seed": 7}, {"reward": 0.5})]
    worker.close()


def test_a_metric_write_that_raises_does_not_kill_the_uploader():
    sink = FakeSink()

    def explode(metrics, step):
        raise RuntimeError("nope")

    worker = UploadWorker([sink], metric_writer=explode)
    worker.submit(MetricItem(metrics={"loss": 0.5}, step=1))
    worker.submit(WriteItem(records=[{"id": 1}]))
    drain(worker)

    assert sink.batches, "the uploader survived the metric failure"
    worker.close()


def test_close_drains_then_closes_every_sink():
    sink = FakeSink()
    worker = UploadWorker([sink])

    worker.submit(WriteItem(records=[{"id": 1}]))
    worker.close()

    assert sink.batches
    assert sink.closed is True


def test_submitting_after_close_is_refused():
    worker = UploadWorker([FakeSink()])
    worker.close()

    assert worker.submit(WriteItem(records=[{"id": 1}])) is False


def test_flush_without_a_running_thread_still_flushes_the_sinks():
    sink = FakeSink()
    worker = UploadWorker([sink])

    assert worker.flush(timeout=1.0) is True
    assert sink.flushes == 1


def test_a_forked_child_starts_over_instead_of_re_uploading_the_parents_queue():
    """The queued records belong to the parent, which still has a live thread.

    Inheriting them would upload each record twice; inheriting the lock could
    deadlock the child on its first write.
    """
    sink = FakeSink()
    worker = UploadWorker([sink], max_queue_size=4)
    worker._queue.put(WriteItem(records=[{"id": "parents"}]))
    old_queue = worker._queue

    worker.reset_after_fork()

    assert worker._queue is not old_queue
    assert worker._queue.empty()
    assert worker._thread is None
    assert isinstance(worker._queue, queue.Queue)
    assert isinstance(worker._lock, type(threading.Lock())) or worker._lock is not None


def test_close_leaves_sinks_open_when_the_uploader_will_not_stop(caplog):
    """Closing them would pull an httpx client or a file handle out from under
    a request still running on that thread."""

    class WedgedSink(FakeSink):
        def __init__(self) -> None:
            super().__init__("wedged")
            self.released = threading.Event()

        def write(self, records, *, line_format=None, step=None) -> None:
            self.released.wait(10.0)

    sink = WedgedSink()
    worker = UploadWorker([sink])
    worker.submit(WriteItem(records=[{"id": 1}]))

    with caplog.at_level("WARNING"):
        worker.close(timeout=0.2)

    assert sink.closed is False
    assert "leaving it and its sinks open" in caplog.text
    sink.released.set()


def test_a_forked_child_resets_every_registered_holder_of_a_connection():
    """One process-wide hook, not one per object: a per-instance
    register_at_fork can never be undone, so it would pin every run the process
    ever opened and re-run hooks for runs that finished hours ago."""
    from prime_runs import _fork

    class Holder:
        def __init__(self) -> None:
            self.reset = 0

        def reset_after_fork(self) -> None:
            self.reset += 1

    holder = Holder()
    _fork.register(holder)

    _fork._reset_all()

    assert holder.reset == 1


def test_one_registered_object_raising_does_not_block_the_others():
    from prime_runs import _fork

    class Boom:
        def reset_after_fork(self) -> None:
            raise RuntimeError("nope")

    class Fine:
        def __init__(self) -> None:
            self.reset = 0

        def reset_after_fork(self) -> None:
            self.reset += 1

    fine = Fine()
    _fork.register(Boom())
    _fork.register(fine)

    _fork._reset_all()

    assert fine.reset == 1


def test_a_transient_failure_drops_the_batch_but_keeps_the_sink():
    """One gateway blip must not empty the rest of the run's dashboard. The
    batch is already lost; retiring the sink loses every batch after it too."""
    from prime_runs.exceptions import RetryableAPIError

    class BlipSink(FakeSink):
        def write(self, records, *, line_format=None, step=None) -> None:
            raise RetryableAPIError("bad gateway", status_code=502)

    sink = BlipSink("blippy")
    worker = UploadWorker([sink])

    worker.submit(WriteItem(records=[{"id": 1}, {"id": 2}]))
    drain(worker)

    assert sink.enabled is True
    # Per sink, and not folded into `dropped`: the queue accepted these records
    # fine, and with both sinks enabled the other one may well have stored them.
    assert worker.failed_records == {"blippy": 2}
    assert worker.dropped == 0
    worker.close()


def test_a_sustained_outage_eventually_retires_the_sink():
    """A blip is forgiven; hours of re-attempting every batch is not useful."""
    from prime_runs.exceptions import TransportError
    from prime_runs.worker import TRANSIENT_FAILURE_LIMIT

    class DeadSink(FakeSink):
        def write(self, records, *, line_format=None, step=None) -> None:
            raise TransportError("connection refused")

    sink = DeadSink("dead")
    worker = UploadWorker([sink])

    for _ in range(TRANSIENT_FAILURE_LIMIT):
        worker.submit(WriteItem(records=[{"id": 1}]))
    drain(worker)

    assert sink.enabled is False
    worker.close()


def test_a_success_forgives_earlier_blips():
    """Strikes are consecutive: an intermittent gateway must never accumulate
    its way to a retirement across an otherwise healthy run."""
    from prime_runs.exceptions import RetryableAPIError
    from prime_runs.worker import TRANSIENT_FAILURE_LIMIT

    class FlakySink(FakeSink):
        """Fails every other batch, forever."""

        def __init__(self) -> None:
            super().__init__("flaky")
            self.calls = 0

        def write(self, records, *, line_format=None, step=None) -> None:
            self.calls += 1
            if self.calls % 2 == 1:
                raise RetryableAPIError("bad gateway", status_code=502)
            super().write(records, line_format=line_format, step=step)

    sink = FlakySink()
    worker = UploadWorker([sink])

    # Far more failures than the limit, but never two in a row.
    for _ in range(TRANSIENT_FAILURE_LIMIT * 4):
        worker.submit(WriteItem(records=[{"id": 1}]))
    drain(worker)

    assert sink.calls == TRANSIENT_FAILURE_LIMIT * 4
    assert sink.enabled is True
    assert len(sink.batches) == TRANSIENT_FAILURE_LIMIT * 2
    worker.close()


def test_a_permanent_failure_retires_the_sink_immediately():
    """A gated account or a rejected credential fails identically forever."""
    from prime_runs.exceptions import UnauthorizedError

    class DeniedSink(FakeSink):
        def write(self, records, *, line_format=None, step=None) -> None:
            raise UnauthorizedError("nope", status_code=401)

    sink = DeniedSink("denied")
    worker = UploadWorker([sink])

    worker.submit(WriteItem(records=[{"id": 1}]))
    drain(worker)

    assert sink.enabled is False
    worker.close()


def test_a_failed_batch_is_counted_once_per_sink_not_once_per_run():
    """Default online runs write to two sinks. Adding both to one total would
    report twice the loss, and would report loss at all when the other sink
    stored the records."""
    from prime_runs.exceptions import RetryableAPIError

    class BlipSink(FakeSink):
        def write(self, records, *, line_format=None, step=None) -> None:
            raise RetryableAPIError("bad gateway", status_code=502)

    broken, healthy = BlipSink("broken"), FakeSink("healthy")
    worker = UploadWorker([broken, healthy])

    worker.submit(WriteItem(records=[{"id": 1}, {"id": 2}]))
    drain(worker)

    assert worker.failed_records == {"broken": 2}
    assert worker.dropped == 0
    assert healthy.batches, "the healthy sink stored them"
    worker.close()
