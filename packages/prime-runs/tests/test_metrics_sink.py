"""The training metrics sink: one request per logged step."""

import httpx
import pytest
from conftest import RecordingHandler

from prime_runs.exceptions import RetryableAPIError, UnauthorizedError
from prime_runs.sinks import RftMetricsSink
from prime_runs.sinks.base import SinkWriteError


def make_sink(make_platform_client, routes):
    handler = RecordingHandler(routes)
    sink = RftMetricsSink(make_platform_client(handler))
    sink.start("run-abc", {})
    return sink, handler


def test_each_metrics_dict_is_one_request(make_platform_client, rft_routes):
    sink, handler = make_sink(make_platform_client, rft_routes)

    sink.write([{"step": 1, "loss": 0.9}, {"step": 2, "loss": 0.8}])

    assert handler.bodies_for("/api/v1/rft/metrics") == [
        {"run_id": "run-abc", "metrics": {"step": 1, "loss": 0.9}},
        {"run_id": "run-abc", "metrics": {"step": 2, "loss": 0.8}},
    ]
    assert sink.steps_written == 2


def test_an_ambiguous_failure_is_replayed(make_platform_client, rft_routes, no_sleep):
    """The platform keeps one row per step, so a retry after a lost response
    cannot double anything, while a lost row is a hole in the curves for good."""
    calls = []

    def flaky(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(502)
        return httpx.Response(200, json={"data": {"status": "success"}})

    routes = dict(rft_routes)
    routes["POST /api/v1/rft/metrics"] = flaky
    sink, handler = make_sink(make_platform_client, routes)

    sink.write([{"step": 1}, {"step": 2}])

    assert len(calls) == 3
    assert sink.steps_written == 2


def test_a_persistent_transient_failure_loses_only_its_step(
    make_platform_client, rft_routes, no_sleep
):
    import json

    def down_for_step_one(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["metrics"]["step"] == 1:
            return httpx.Response(502)
        return httpx.Response(200, json={"data": {"status": "success"}})

    routes = dict(rft_routes)
    routes["POST /api/v1/rft/metrics"] = down_for_step_one
    sink, handler = make_sink(make_platform_client, routes)

    with pytest.raises(SinkWriteError) as info:
        sink.write([{"step": 1}, {"step": 2}])

    assert info.value.failed_records == 1
    assert isinstance(info.value.cause, RetryableAPIError)
    assert sink.steps_written == 1


def test_a_sink_wide_failure_stops_the_batch_and_counts_the_rest(make_platform_client, rft_routes):
    routes = dict(rft_routes)
    routes["POST /api/v1/rft/metrics"] = lambda request: httpx.Response(401)
    sink, handler = make_sink(make_platform_client, routes)

    with pytest.raises(SinkWriteError) as info:
        sink.write([{"step": 1}, {"step": 2}, {"step": 3}])

    assert info.value.failed_records == 3
    assert isinstance(info.value.cause, UnauthorizedError)
    assert len(handler.paths()) == 1


def test_a_record_that_is_not_a_mapping_is_counted_not_sent(make_platform_client, rft_routes):
    sink, handler = make_sink(make_platform_client, rft_routes)

    with pytest.raises(SinkWriteError) as info:
        sink.write(["loss=0.5", {"step": 1}])

    assert info.value.failed_records == 1
    assert isinstance(info.value.cause, TypeError)
    assert len(handler.bodies_for("/api/v1/rft/metrics")) == 1


def test_write_before_start_is_a_producer_bug(make_platform_client, rft_routes):
    sink = RftMetricsSink(make_platform_client(RecordingHandler(rft_routes)))

    with pytest.raises(RuntimeError, match="before start"):
        sink.write([{"step": 1}])
