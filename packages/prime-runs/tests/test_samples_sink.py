"""The legacy sample sink that keeps today's viewer working."""

from _fakes import make_episode, make_trace
from conftest import RecordingHandler

from prime_runs.sinks import EvalSamplesSink


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
    """Streaming must produce the numbering the old one-shot upload produced."""
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
    """A JSON episode or a bare trace has no v0 projection. Raising would have
    the worker retire the sink for the run — outside the traces beta that is
    an empty viewer — so the record is skipped, warned once, and counted,
    while the episode object in the same batch is still stored."""
    sink, handler = make_sink(make_platform_client, eval_routes)

    with caplog.at_level("WARNING"):
        sink.write(
            [{"id": "ep-json", "traces": [{"id": "t"}]}, make_episode("ep-1", [make_trace()])]
        )
        sink.write([make_trace()])

    body = handler.bodies_for("/api/v1/evaluations/eval-abc/samples")[0]
    assert [row["sample_id"] for row in body["samples"]] == ["ep-1"]
    assert len(handler.bodies_for("/api/v1/evaluations/eval-abc/samples")) == 1
    assert sink.skipped == 2
    assert sink.enabled is True
    assert sum("no projection" in r.getMessage() for r in caplog.records) == 1


def test_an_empty_batch_makes_no_request(make_platform_client, eval_routes):
    sink, handler = make_sink(make_platform_client, eval_routes)

    sink.write([])

    assert handler.requests == []
