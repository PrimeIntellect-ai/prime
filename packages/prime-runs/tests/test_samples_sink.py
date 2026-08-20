"""The legacy sample sink that keeps today's viewer working."""

import pytest
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


def test_individual_traces_are_projected_with_continuous_rollout_numbers(
    make_platform_client, eval_routes
):
    sink, handler = make_sink(make_platform_client, eval_routes)

    sink.write([make_trace(trace_id="trace-1", idx=0)])
    sink.write([make_trace(trace_id="trace-2", idx=0)])

    posted = handler.bodies_for("/api/v1/evaluations/eval-abc/samples")
    assert [body["samples"][0]["sample_id"] for body in posted] == ["trace-1", "trace-2"]
    assert [body["samples"][0]["rollout_number"] for body in posted] == [1, 2]


def test_a_producer_that_already_speaks_v0_is_passed_through(make_platform_client, eval_routes):
    sink, handler = make_sink(make_platform_client, eval_routes)

    sink.write([{"sample_id": "s1", "reward": 1.0}])

    body = handler.bodies_for("/api/v1/evaluations/eval-abc/samples")[0]
    assert body["samples"] == [{"sample_id": "s1", "reward": 1.0}]


def test_serialized_trace_records_are_projected(make_platform_client, eval_routes):
    sink, handler = make_sink(make_platform_client, eval_routes)

    sink.write([make_trace(trace_id="serialized-trace", reward=0.75).to_record()])

    body = handler.bodies_for("/api/v1/evaluations/eval-abc/samples")[0]
    assert body["samples"][0]["sample_id"] == "serialized-trace"
    assert body["samples"][0]["reward"] == 0.75


def test_serialized_episode_records_keep_the_native_wrapper(make_platform_client, eval_routes):
    sink, handler = make_sink(make_platform_client, eval_routes)
    record = make_episode("serialized-episode", [make_trace()]).to_record()

    sink.write([record])

    sample = handler.bodies_for("/api/v1/evaluations/eval-abc/samples")[0]["samples"][0]
    assert sample["sample_id"] == "serialized-episode"
    assert sample["info"]["native_wrapper"] == record


def test_serialized_message_graphs_recover_the_viewer_completion(make_platform_client, eval_routes):
    sink, handler = make_sink(make_platform_client, eval_routes)
    record = {
        "id": "graph-trace",
        "task": {"data": {"idx": 7, "answer": "42"}},
        "agent": {"name": "solver", "trainable": True},
        "nodes": [
            {"parent": None, "message": {"role": "user", "content": "6 * 7?"}},
            {"parent": 0, "message": {"role": "assistant", "content": "42"}},
        ],
        "calls": [
            {
                "node": 1,
                "usage": {"prompt_tokens": 4, "completion_tokens": 1},
            }
        ],
        "rewards": {"correct": {"score": 1.0, "weight": 1.0}},
    }

    sink.write([record])

    sample = handler.bodies_for("/api/v1/evaluations/eval-abc/samples")[0]["samples"][0]
    assert sample["example_id"] == 7
    assert sample["completion"][-1] == {"role": "assistant", "content": "42"}
    assert sample["reward"] == 1.0


def test_records_this_sink_cannot_project_fail_explicitly(make_platform_client, eval_routes):
    sink, handler = make_sink(make_platform_client, eval_routes)

    with pytest.raises(TypeError, match="non-empty 'id'"):
        sink.write([{"unrelated": True}])

    assert handler.requests == []


def test_an_empty_batch_makes_no_request(make_platform_client, eval_routes):
    sink, handler = make_sink(make_platform_client, eval_routes)

    sink.write([])

    assert handler.requests == []
