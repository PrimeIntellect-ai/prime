"""The legacy sample sink that keeps today's viewer working."""

import httpx
from _fakes import make_episode, make_trace
from conftest import RecordingHandler

from prime_runs.sinks import EvalSamplesSink
from prime_runs.worker import UploadWorker


def make_sink(make_platform_client, eval_routes):
    handler = RecordingHandler(eval_routes)
    sink = EvalSamplesSink(make_platform_client(handler))
    sink.start("eval-abc", {})
    return sink, handler


def test_episodes_are_projected_and_posted(make_platform_client, eval_routes):
    sink, handler = make_sink(make_platform_client, eval_routes)

    sink.write([make_episode("ep-1", [make_trace()])])

    body = handler.bodies_for("/api/v1/evaluations/eval-abc/samples")[0]
    assert body["samples"][0]["sample_id"] == "ep-1"
    assert sink.samples_written == 1


def test_rollout_numbering_is_continuous_across_streamed_batches(make_platform_client, eval_routes):
    sink, handler = make_sink(make_platform_client, eval_routes)

    sink.write([make_episode("ep-1", [make_trace(idx=0)])])
    sink.write([make_episode("ep-2", [make_trace(idx=0)])])

    posted = handler.bodies_for("/api/v1/evaluations/eval-abc/samples")
    assert [body["samples"][0]["rollout_number"] for body in posted] == [1, 2]


def test_a_producer_that_already_speaks_v0_is_passed_through(make_platform_client, eval_routes):
    sink, handler = make_sink(make_platform_client, eval_routes)

    sink.write([{"sample_id": "s1", "reward": 1.0}])

    body = handler.bodies_for("/api/v1/evaluations/eval-abc/samples")[0]
    assert body["samples"] == [{"sample_id": "s1", "reward": 1.0}]


def test_records_this_sink_cannot_project_are_skipped_not_fatal(
    make_platform_client, eval_routes, caplog
):
    sink, handler = make_sink(make_platform_client, eval_routes)

    with caplog.at_level("WARNING"):
        sink.write(
            [
                make_episode("ep-1", [make_trace()]),
                {"id": "ep-json", "traces": [{"id": "t"}]},
            ]
        )
        sink.write([make_trace()])

    body = handler.bodies_for("/api/v1/evaluations/eval-abc/samples")[0]
    assert [row["sample_id"] for row in body["samples"]] == ["ep-1"]
    assert len(handler.bodies_for("/api/v1/evaluations/eval-abc/samples")) == 1
    assert sink.skipped == 2
    assert sink.enabled is True
    warnings = [
        record.getMessage() for record in caplog.records if "no projection" in record.getMessage()
    ]
    assert len(warnings) == 1
    assert "(dict)" in warnings[0]


def test_an_empty_batch_makes_no_request(make_platform_client, eval_routes):
    sink, handler = make_sink(make_platform_client, eval_routes)

    sink.write([])

    assert handler.requests == []


def test_a_rejected_sample_batch_does_not_abort_later_batches(
    make_platform_client, eval_routes, monkeypatch
):
    statuses = iter([200, 422, 200, 200])

    def respond(request):
        status = next(statuses)
        return httpx.Response(status, json={"detail": "invalid sample"} if status == 422 else {})

    eval_routes["POST /api/v1/evaluations/eval-abc/samples"] = respond
    monkeypatch.setattr(
        "prime_runs.sinks.samples.batch_samples",
        lambda samples: [[sample] for sample in samples],
    )
    sink, handler = make_sink(make_platform_client, eval_routes)
    worker = UploadWorker([sink])

    worker.submit([{"sample_id": sample_id} for sample_id in ("a", "b", "c")])
    assert worker.flush(timeout=5.0)
    worker.submit([{"sample_id": "d"}])
    assert worker.flush(timeout=5.0)

    posted = handler.bodies_for("/api/v1/evaluations/eval-abc/samples")
    assert [body["samples"][0]["sample_id"] for body in posted] == ["a", "b", "c", "d"]
    assert sink.samples_written == 3
    assert sink.enabled is True
    assert worker.failed_records == {"eval_samples": 1}
    worker.close()


def test_a_sink_wide_failure_counts_only_current_and_unattempted_records(
    make_platform_client, eval_routes, monkeypatch
):
    statuses = iter([200, 401])

    def respond(request):
        status = next(statuses)
        return httpx.Response(status, json={"detail": "unauthorized"})

    eval_routes["POST /api/v1/evaluations/eval-abc/samples"] = respond
    monkeypatch.setattr(
        "prime_runs.sinks.samples.batch_samples",
        lambda samples: [[sample] for sample in samples],
    )
    sink, handler = make_sink(make_platform_client, eval_routes)
    worker = UploadWorker([sink])

    worker.submit([{"sample_id": sample_id} for sample_id in ("a", "b", "c")])
    assert worker.flush(timeout=5.0)

    posted = handler.bodies_for("/api/v1/evaluations/eval-abc/samples")
    assert [body["samples"][0]["sample_id"] for body in posted] == ["a", "b"]
    assert sink.samples_written == 1
    assert sink.enabled is False
    assert worker.failed_records == {"eval_samples": 2}
    worker.close()
