"""The background uploader: backpressure, containment, fork safety."""

import queue
import threading

import pytest
from conftest import FakeSink

from prime_runs.exceptions import APIError, ValidationRejectedError
from prime_runs.worker import UploadWorker


class BlockingSink(FakeSink):
    def __init__(self) -> None:
        super().__init__("blocking")
        self.entered = threading.Event()
        self.released = threading.Event()

    def write(self, records) -> None:
        self.entered.set()
        self.released.wait(5.0)
        super().write(records)


def drain(worker: UploadWorker) -> None:
    assert worker.flush(timeout=5.0)


def test_records_reach_every_enabled_sink():
    sinks = [FakeSink("a"), FakeSink("b")]
    worker = UploadWorker(sinks)

    worker.submit([{"id": 1}])
    drain(worker)

    for sink in sinks:
        assert sink.batches == [[{"id": 1}]]
    worker.close()


def test_a_disabled_sink_is_skipped():
    live, dead = FakeSink("live"), FakeSink("dead")
    dead.enabled = False
    worker = UploadWorker([live, dead])

    worker.submit([{"id": 1}])
    drain(worker)

    assert live.batches and not dead.batches
    worker.close()


def test_one_sink_failing_does_not_stop_the_others():
    broken, healthy = FakeSink("broken", fail_on_write=True), FakeSink("healthy")
    reported = []
    worker = UploadWorker([broken, healthy], on_error=lambda name, exc: reported.append(name))

    worker.submit([{"id": 1}])
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
        worker.submit([{"id": 1}])
    drain(worker)

    assert reported == ["broken"]
    worker.close()


def test_records_that_skip_a_sink_retired_by_error_are_counted_as_lost_to_it():
    """After the first failed batch the sink is off, but the producer keeps
    logging; every later batch is just as lost to that sink as the first. Live
    run 2026-08-21: five episodes, footer said "1 failed via traces"."""
    broken = FakeSink("broken", fail_on_write=True)
    worker = UploadWorker([broken])

    for _ in range(5):
        worker.submit([{"id": 1}])
    drain(worker)

    assert worker.failed_records == {"broken": 5}
    worker.close()


def test_records_that_skip_a_sink_which_switched_itself_off_are_not_counted():
    """A sink that retires quietly (nowhere for the records to go, e.g. outside
    the traces beta) lost nothing, and must not start a failure count."""

    class QuietSink(FakeSink):
        def write(self, records) -> None:
            self.enabled = False  # retires without raising, like service_not_enabled

    quiet = QuietSink("quiet")
    worker = UploadWorker([quiet])

    for _ in range(3):
        worker.submit([{"id": 1}])
    drain(worker)

    assert worker.failed_records == {}
    worker.close()


def test_a_full_queue_drops_rather_than_blocking_the_producer():
    """Stalling a training run to protect telemetry is the wrong trade."""
    sink = BlockingSink()
    worker = UploadWorker([sink], max_queue_size=1, put_timeout=0.05)

    # First item is picked up and wedges the uploader inside sink.write().
    assert worker.submit([{"id": 0}])
    assert sink.entered.wait(5.0), "the uploader never reached the sink"
    # Second fills the one-slot queue; third has nowhere to go.
    worker.submit([{"id": 1}])
    accepted = worker.submit([{"id": 2}, {"id": 3}])

    assert accepted is False
    assert worker.dropped == 2

    sink.released.set()
    worker.close()


def test_flush_uses_the_drain_budget_to_get_behind_a_full_queue():
    sink = BlockingSink()
    worker = UploadWorker([sink], max_queue_size=1, put_timeout=0.01)
    assert worker.submit([{"id": 0}])
    assert sink.entered.wait(1.0)
    assert worker.submit([{"id": 1}])

    release = threading.Timer(0.05, sink.released.set)
    release.start()
    try:
        assert worker.flush(timeout=0.5) is True
    finally:
        sink.released.set()
        release.join()
        worker.close(timeout=1.0)

    assert [batch[0]["id"] for batch in sink.batches] == [0, 1]


def test_close_uses_its_budget_to_queue_the_stop_behind_pending_records():
    sink = BlockingSink()
    worker = UploadWorker([sink], max_queue_size=1, put_timeout=0.01)
    assert worker.submit([{"id": 0}])
    assert sink.entered.wait(1.0)
    assert worker.submit([{"id": 1}])

    release = threading.Timer(0.05, sink.released.set)
    release.start()
    try:
        worker.close(timeout=0.5)
    finally:
        sink.released.set()
        release.join()

    assert [batch[0]["id"] for batch in sink.batches] == [0, 1]
    assert sink.closed is True


def test_close_drains_then_closes_every_sink():
    sink = FakeSink()
    worker = UploadWorker([sink])

    worker.submit([{"id": 1}])
    worker.close()

    assert sink.batches
    assert sink.closed is True


def test_submitting_after_close_is_refused():
    sink = FakeSink()
    worker = UploadWorker([sink])
    worker.close()
    worker.start()

    assert worker.submit([{"id": 1}]) is False
    assert worker._thread is None
    assert sink.batches == []


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
    worker._queue.put([{"id": "parents"}])
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

        def write(self, records) -> None:
            self.released.wait(10.0)

    sink = WedgedSink()
    worker = UploadWorker([sink])
    worker.submit([{"id": 1}])

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
        def write(self, records) -> None:
            raise RetryableAPIError("bad gateway", status_code=502)

    sink = BlipSink("blippy")
    worker = UploadWorker([sink])

    worker.submit([{"id": 1}, {"id": 2}])
    drain(worker)

    assert sink.enabled is True
    # Per sink, and not folded into `dropped`: the queue accepted these records
    # fine, and with both sinks enabled the other one may well have stored them.
    assert worker.failed_records == {"blippy": 2}
    assert worker.dropped == 0
    worker.close()


@pytest.mark.parametrize(
    "error",
    [
        ValueError("Out of range float values are not JSON compliant"),
        ValidationRejectedError("invalid record", status_code=400),
        APIError("unprocessable record", status_code=422),
    ],
    ids=["local-encoding", "traces-validation", "samples-validation"],
)
def test_a_record_rejection_drops_only_its_batch(error):
    """Malformed content must not prevent later valid records from uploading."""

    class RejectOnceSink(FakeSink):
        def __init__(self) -> None:
            super().__init__("reject-once")
            self.calls = 0

        def write(self, records) -> None:
            self.calls += 1
            if self.calls == 1:
                raise error
            super().write(records)

    sink = RejectOnceSink()
    reported = []
    worker = UploadWorker([sink], on_error=lambda name, exc: reported.append((name, exc)))

    # Drained between the two: queued batches coalesce into one write, and a
    # rejection is per write.
    worker.submit([{"id": "bad"}])
    drain(worker)
    worker.submit([{"id": "good"}])
    drain(worker)

    assert sink.enabled is True
    assert sink.calls == 2
    assert sink.batches == [[{"id": "good"}]]
    assert worker.failed_records == {"reject-once": 1}
    assert reported == [("reject-once", error)]
    worker.close()


def test_a_sustained_outage_eventually_retires_the_sink():
    """A blip is forgiven; hours of re-attempting every batch is not useful."""
    from prime_runs.exceptions import TransportError
    from prime_runs.worker import TRANSIENT_FAILURE_LIMIT

    class DeadSink(FakeSink):
        def write(self, records) -> None:
            raise TransportError("connection refused")

    sink = DeadSink("dead")
    worker = UploadWorker([sink])

    # One write per strike: drained between submits so they do not coalesce.
    for _ in range(TRANSIENT_FAILURE_LIMIT):
        worker.submit([{"id": 1}])
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

        def write(self, records) -> None:
            self.calls += 1
            if self.calls % 2 == 1:
                raise RetryableAPIError("bad gateway", status_code=502)
            super().write(records)

    sink = FlakySink()
    worker = UploadWorker([sink])

    # Far more failures than the limit, but never two in a row. Drained one at
    # a time: queued batches coalesce, and this is about consecutive *writes*.
    for _ in range(TRANSIENT_FAILURE_LIMIT * 4):
        worker.submit([{"id": 1}])
        drain(worker)

    assert sink.calls == TRANSIENT_FAILURE_LIMIT * 4
    assert sink.enabled is True
    assert len(sink.batches) == TRANSIENT_FAILURE_LIMIT * 2
    worker.close()


def test_a_permanent_failure_retires_the_sink_immediately():
    """A gated account or a rejected credential fails identically forever."""
    from prime_runs.exceptions import UnauthorizedError

    class DeniedSink(FakeSink):
        def write(self, records) -> None:
            raise UnauthorizedError("nope", status_code=401)

    sink = DeniedSink("denied")
    worker = UploadWorker([sink])

    worker.submit([{"id": 1}])
    drain(worker)

    assert sink.enabled is False
    worker.close()


def test_a_failed_batch_is_counted_once_per_sink_not_once_per_run():
    """Default online runs write to two sinks. Adding both to one total would
    report twice the loss, and would report loss at all when the other sink
    stored the records."""
    from prime_runs.exceptions import RetryableAPIError

    class BlipSink(FakeSink):
        def write(self, records) -> None:
            raise RetryableAPIError("bad gateway", status_code=502)

    broken, healthy = BlipSink("broken"), FakeSink("healthy")
    worker = UploadWorker([broken, healthy])

    worker.submit([{"id": 1}, {"id": 2}])
    drain(worker)

    assert worker.failed_records == {"broken": 2}
    assert worker.dropped == 0
    assert healthy.batches, "the healthy sink stored them"
    worker.close()


def test_records_queued_during_an_upload_go_out_as_one_batch():
    """One request per episode runs into the platform's per-minute limit on a
    fast eval; whatever accumulates while a request is in flight must go out
    together, so the request rate tracks upload latency instead."""
    sink = BlockingSink()
    worker = UploadWorker([sink])

    assert worker.submit([{"id": 0}])
    assert sink.entered.wait(1.0), "the uploader never reached the sink"
    for n in (1, 2, 3):
        assert worker.submit([{"id": n}])
    sink.released.set()
    drain(worker)

    assert [[r["id"] for r in batch] for batch in sink.batches] == [[0], [1, 2, 3]]
    worker.close()


def test_coalescing_keeps_traces_and_episodes_apart():
    """The traces sink infers the line format from a batch's first record, so
    mixing episodes into a batch of bare traces would misfile every record."""
    sink = BlockingSink()
    worker = UploadWorker([sink])

    assert worker.submit([{"id": "t0"}])
    assert sink.entered.wait(1.0)
    worker.submit([{"id": "t1"}])
    worker.submit([{"id": "e1", "traces": []}])
    worker.submit([{"id": "e2", "traces": []}])
    worker.submit([{"id": "t2"}])
    sink.released.set()
    drain(worker)

    assert [[r["id"] for r in batch] for batch in sink.batches] == [
        ["t0"],
        ["t1"],
        ["e1", "e2"],
        ["t2"],
    ]
    worker.close()


def test_a_flush_barrier_is_not_reordered_past_the_records_before_it():
    """Coalescing must stop at the barrier: flush() promises that everything
    submitted before it has been written when it returns."""
    sink = BlockingSink()
    worker = UploadWorker([sink])

    assert worker.submit([{"id": 0}])
    assert sink.entered.wait(1.0)
    worker.submit([{"id": 1}])
    release = threading.Timer(0.05, sink.released.set)
    release.start()
    try:
        assert worker.flush(timeout=2.0) is True
    finally:
        release.join()
    worker.submit([{"id": 2}])

    assert [[r["id"] for r in batch] for batch in sink.batches] == [[0], [1]]
    drain(worker)
    assert [[r["id"] for r in batch] for batch in sink.batches] == [[0], [1], [2]]
    worker.close()


def test_a_worker_with_no_sinks_never_starts_a_thread():
    """A disabled run submits every batch; copying them into a queue and
    starting a thread to find there is nowhere to put them is pure cost."""
    worker = UploadWorker([])

    assert worker.submit([{"id": 1}]) is True
    assert worker.submit([{"id": 2}]) is True

    assert worker._thread is None
    assert worker.dropped == 0
    assert worker.flush(timeout=1.0) is True
    worker.close()
