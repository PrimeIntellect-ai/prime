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


def test_records_that_skip_a_sink_retired_by_error_are_counted_as_lost_to_it():
    broken = FakeSink("broken", fail_on_write=True)
    worker = UploadWorker([broken])

    for _ in range(5):
        worker.submit([{"id": 1}])
    drain(worker)

    assert worker.failed_records == {"broken": 5}
    worker.close()


def test_records_that_skip_a_sink_which_switched_itself_off_are_not_counted():
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


def test_a_forked_child_starts_over_instead_of_re_uploading_the_parents_queue():
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


def test_a_transient_failure_drops_the_batch_but_keeps_the_sink():
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


def test_a_cooldown_gives_a_retired_sink_another_chance():
    from prime_runs.exceptions import TransportError
    from prime_runs.worker import TRANSIENT_FAILURE_LIMIT

    class OutageSink(FakeSink):
        def __init__(self) -> None:
            super().__init__("outage")
            self.down = True

        def write(self, records) -> None:
            if self.down:
                raise TransportError("connection refused")
            super().write(records)

    sink = OutageSink()
    worker = UploadWorker([sink], retire_cooldown=0.05)

    for _ in range(TRANSIENT_FAILURE_LIMIT):
        worker.submit([{"id": "lost"}])
        drain(worker)
    assert sink.enabled is False
    worker.submit([{"id": "during"}])
    drain(worker)
    assert sink.batches == []

    # A real wait: the autouse no_sleep fixture stubs out time.sleep.
    threading.Event().wait(0.06)
    sink.down = False
    worker.submit([{"id": "after"}])
    drain(worker)

    assert sink.enabled is True
    assert sink.batches == [[{"id": "after"}]]
    assert worker.failed_records == {"outage": TRANSIENT_FAILURE_LIMIT + 1}
    worker.close()


def test_a_permanent_failure_is_not_revived_by_the_cooldown():
    from prime_runs.exceptions import UnauthorizedError

    class DeniedSink(FakeSink):
        def write(self, records) -> None:
            raise UnauthorizedError("nope", status_code=401)

    sink = DeniedSink("denied")
    worker = UploadWorker([sink], retire_cooldown=0.01)

    worker.submit([{"id": 1}])
    drain(worker)
    threading.Event().wait(0.02)
    worker.submit([{"id": 2}])
    drain(worker)

    assert sink.enabled is False
    assert worker.failed_records == {"denied": 2}
    worker.close()


def test_a_success_forgives_earlier_blips():
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
    worker = UploadWorker([])

    assert worker.submit([{"id": 1}]) is True
    assert worker.submit([{"id": 2}]) is True

    assert worker._thread is None
    assert worker.dropped == 0
    assert worker.flush(timeout=1.0) is True
    worker.close()
