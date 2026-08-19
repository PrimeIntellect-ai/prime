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


def test_records_this_sink_cannot_project_are_skipped_not_posted(make_platform_client, eval_routes):
    """A malformed row would be rejected for the whole batch, taking the valid
    rows with it."""
    sink, handler = make_sink(make_platform_client, eval_routes)

    sink.write([{"unrelated": True}])

    assert handler.requests == []


def test_an_empty_batch_makes_no_request(make_platform_client, eval_routes):
    sink, handler = make_sink(make_platform_client, eval_routes)

    sink.write([])

    assert handler.requests == []
