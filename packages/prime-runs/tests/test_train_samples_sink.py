"""The training samples sink: step-keyed Parquet through presign -> PUT -> confirm."""

import httpx
import pytest
from _fakes import make_episode, make_trace, make_train_episode
from conftest import RecordingHandler, StorageHandler

from prime_runs.exceptions import APIError, RetryableAPIError, TransportError
from prime_runs.sinks import RftSamplesSink, training_step
from prime_runs.sinks.base import SinkWriteError


class Encoder:
    """Records what it was asked to encode; returns a recognizable payload."""

    def __init__(self, payload=b"parquet-bytes"):
        self.calls = []
        self.offsets = []
        self.payload = payload

    def __call__(self, episodes, run_id, step, sample_id_offset=0):
        self.calls.append((list(episodes), run_id, step))
        self.offsets.append(sample_id_offset)
        return self.payload


def make_sink(make_platform_client, routes, *, encoder=None, storage=None, **kwargs):
    handler = RecordingHandler(routes)
    storage = storage or StorageHandler()
    encoder = encoder or Encoder()
    sink = RftSamplesSink(
        make_platform_client(handler), encoder=encoder, upload_client=storage.client(), **kwargs
    )
    sink.start("run-abc", {})
    return sink, handler, storage, encoder


def test_each_step_on_the_cadence_is_one_upload(make_platform_client, rft_routes):
    sink, handler, storage, encoder = make_sink(make_platform_client, rft_routes)
    episodes = [
        make_train_episode("e1", step=10),
        make_train_episode("e2", step=20),
        make_train_episode("e3", step=10),
        make_train_episode("e4", step=11),  # off-cadence: not a loss
    ]

    sink.write(episodes)

    assert [(step, [e.id for e in eps]) for eps, _, step in encoder.calls] == [
        (10, ["e1", "e3"]),
        (20, ["e2"]),
    ]
    assert [body["step"] for body in handler.bodies_for("/api/v1/rft/samples/presign")] == [10, 20]
    assert [str(u.url) for u in storage.uploads] == [
        "http://storage.test/run-abc/step-10.parquet",
        "http://storage.test/run-abc/step-20.parquet",
    ]
    assert storage.uploads[0].content == b"parquet-bytes"
    assert storage.uploads[0].headers["content-type"] == "application/parquet"
    confirms = handler.bodies_for("/api/v1/rft/samples/confirm")
    assert confirms[0] == {
        "run_id": "run-abc",
        "step": 10,
        "s3_key": "runs/run-abc/step-10.parquet",
    }
    assert sink.steps_written == 2
    assert sink.sampled_out == 1
    assert sink.skipped == 0


def test_the_cadence_is_configurable(make_platform_client, rft_routes):
    sink, handler, storage, encoder = make_sink(make_platform_client, rft_routes, step_interval=1)

    sink.write([make_train_episode("e1", step=7)])

    assert [step for _, _, step in encoder.calls] == [7]
    with pytest.raises(ValueError):
        RftSamplesSink(sink._client, step_interval=0)


def test_records_without_training_work_have_no_row_here(make_platform_client, rft_routes):
    """Eval-work episodes, bare traces and JSON episodes reach Prime Traces
    only; here they are counted, not uploaded, and not a loss."""
    sink, handler, storage, encoder = make_sink(make_platform_client, rft_routes)

    sink.write(
        [
            make_train_episode("eval-work", step=10, work="eval"),
            make_episode("unstamped", [make_trace()]),
            make_trace(),
            {
                "id": "json-episode",
                "traces": [],
                "run": {"type": "train", "work": {"type": "train", "step": 10}},
            },
        ]
    )

    assert encoder.calls == []
    assert handler.paths() == []
    assert sink.skipped == 4


def test_nothing_to_encode_makes_no_request(make_platform_client, rft_routes):
    sink, handler, storage, encoder = make_sink(
        make_platform_client, rft_routes, encoder=Encoder(payload=None)
    )

    sink.write([make_train_episode("e1", step=10)])

    assert encoder.calls and handler.paths() == []
    assert sink.steps_written == 0


def test_a_failed_presign_loses_that_step_only(make_platform_client, rft_routes):
    """Presign is replayable (it only mints a URL), so a blip is retried; a
    step whose presign keeps failing is lost on its own, not the batch."""
    import json

    def flaky(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["step"] == 10:
            return httpx.Response(502)
        return rft_routes["POST /api/v1/rft/samples/presign"](request)

    routes = dict(rft_routes)
    routes["POST /api/v1/rft/samples/presign"] = flaky
    sink, handler, storage, encoder = make_sink(make_platform_client, routes)

    with pytest.raises(SinkWriteError) as info:
        sink.write(
            [
                make_train_episode("e1", step=10),
                make_train_episode("e2", step=10),
                make_train_episode("e3", step=20),
            ]
        )

    # Step 10 (two episodes) was lost; step 20 went through.
    assert info.value.failed_records == 2
    assert isinstance(info.value.cause, RetryableAPIError)
    assert sink.steps_written == 1
    assert [str(u.url) for u in storage.uploads] == ["http://storage.test/run-abc/step-20.parquet"]


def test_the_storage_put_is_retried_then_reported_as_transient(
    make_platform_client, rft_routes, no_sleep
):
    storage = StorageHandler(status_codes=[503, 503, 503])
    sink, handler, storage, encoder = make_sink(make_platform_client, rft_routes, storage=storage)

    with pytest.raises(SinkWriteError) as info:
        sink.write([make_train_episode("e1", step=10)])

    assert len(storage.uploads) == 3
    assert isinstance(info.value.cause, RetryableAPIError)
    assert handler.paths() == ["POST /api/v1/rft/samples/presign"]  # never confirmed


def test_a_storage_put_that_recovers_is_confirmed(make_platform_client, rft_routes, no_sleep):
    storage = StorageHandler(status_codes=[500, 200])
    sink, handler, storage, encoder = make_sink(make_platform_client, rft_routes, storage=storage)

    sink.write([make_train_episode("e1", step=10)])

    assert len(storage.uploads) == 2
    assert "POST /api/v1/rft/samples/confirm" in handler.paths()


def test_a_storage_rejection_is_not_retried(make_platform_client, rft_routes):
    storage = StorageHandler(status_codes=[403])
    sink, handler, storage, encoder = make_sink(make_platform_client, rft_routes, storage=storage)

    with pytest.raises(SinkWriteError) as info:
        sink.write([make_train_episode("e1", step=10)])

    assert len(storage.uploads) == 1
    assert isinstance(info.value.cause, APIError)
    assert not isinstance(info.value.cause, RetryableAPIError)


def test_a_dropped_connection_to_storage_is_transient(make_platform_client, rft_routes, no_sleep):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    storage = StorageHandler()
    storage.__call__ = refuse  # type: ignore[method-assign]
    upload = httpx.Client(transport=httpx.MockTransport(refuse))
    handler = RecordingHandler(rft_routes)
    sink = RftSamplesSink(make_platform_client(handler), encoder=Encoder(), upload_client=upload)
    sink.start("run-abc", {})

    with pytest.raises(SinkWriteError) as info:
        sink.write([make_train_episode("e1", step=10)])

    assert isinstance(info.value.cause, TransportError)


def test_an_encoder_failure_is_a_record_rejection(make_platform_client, rft_routes):
    def broken(episodes, run_id, step, sample_id_offset=0):
        raise ValueError("cannot serialize")

    sink, handler, storage, encoder = make_sink(make_platform_client, rft_routes, encoder=broken)

    with pytest.raises(SinkWriteError) as info:
        sink.write([make_train_episode("e1", step=10)])

    assert isinstance(info.value.cause, ValueError)
    assert handler.paths() == []


def test_a_presign_without_a_url_is_an_api_error(make_platform_client, rft_routes):
    routes = dict(rft_routes)
    routes["POST /api/v1/rft/samples/presign"] = {"data": {"expiresIn": 900}}
    sink, handler, storage, encoder = make_sink(make_platform_client, routes)

    with pytest.raises(SinkWriteError) as info:
        sink.write([make_train_episode("e1", step=10)])

    assert isinstance(info.value.cause, APIError)
    assert "presign" in str(info.value.cause)


def test_a_step_logged_again_gets_its_own_sample_id_range(make_platform_client, rft_routes):
    """Objects are additive per step on the platform (each presign mints a new
    key and the viewer unions them), and the viewer looks samples up by
    (step, sample_id): a straggler batch must not reuse the first object's ids."""
    from prime_runs.sinks.train_samples import SAMPLE_ID_STRIDE

    sink, handler, storage, encoder = make_sink(make_platform_client, rft_routes)

    sink.write([make_train_episode("e1", step=10), make_train_episode("e2", step=20)])
    sink.write([make_train_episode("e3", step=10)])  # dispatched at 10, shipped later

    assert [step for _, _, step in encoder.calls] == [10, 20, 10]
    assert encoder.offsets == [0, 0, SAMPLE_ID_STRIDE]
    presigned = [body["step"] for body in handler.bodies_for("/api/v1/rft/samples/presign")]
    assert presigned == [10, 20, 10]
    assert sink.steps_written == 3


def test_an_upload_that_failed_still_advances_the_sample_id_range(make_platform_client, rft_routes):
    """An attempt that failed late may still have landed an object; the next
    one for the step must not overlap it."""
    from prime_runs.sinks.train_samples import SAMPLE_ID_STRIDE

    storage = StorageHandler(status_codes=[403])
    sink, handler, storage, encoder = make_sink(make_platform_client, rft_routes, storage=storage)

    with pytest.raises(SinkWriteError):
        sink.write([make_train_episode("e1", step=10)])
    sink.write([make_train_episode("e2", step=10)])

    assert encoder.offsets == [0, SAMPLE_ID_STRIDE]


def test_a_confirm_whose_response_was_lost_is_replayed(make_platform_client, rft_routes, no_sleep):
    """Confirm validates the key and refreshes progress; a second confirm of
    the same key changes nothing, so an ambiguous failure is retried rather
    than costing the step."""
    attempts = []

    def flaky(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(504)
        return httpx.Response(200, json={"data": {"status": "success"}})

    routes = dict(rft_routes)
    routes["POST /api/v1/rft/samples/confirm"] = flaky
    sink, handler, storage, encoder = make_sink(make_platform_client, routes)

    sink.write([make_train_episode("e1", step=10)])

    assert len(attempts) == 2
    assert len(storage.uploads) == 1
    assert sink.steps_written == 1


def test_without_pyarrow_the_sink_turns_itself_off_quietly(
    make_platform_client, rft_routes, monkeypatch, caplog
):
    """No loss: the episodes still reach Prime Traces. Say how to turn it on."""
    monkeypatch.setattr("prime_runs.projection.parquet_available", lambda: False)
    handler = RecordingHandler(rft_routes)
    sink = RftSamplesSink(make_platform_client(handler))

    with caplog.at_level("WARNING"):
        sink.start("run-abc", {})

    assert sink.enabled is False
    assert "prime-runs[train]" in caplog.text
    sink.write([make_train_episode("e1", step=10)])
    assert handler.paths() == []


def test_close_releases_only_an_owned_upload_client(make_platform_client, rft_routes):
    handler = RecordingHandler(rft_routes)
    injected = StorageHandler().client()
    sink = RftSamplesSink(make_platform_client(handler), encoder=Encoder(), upload_client=injected)
    sink.close()
    assert not injected.is_closed

    owned = RftSamplesSink(make_platform_client(handler), encoder=Encoder())
    owned.start("run-abc", {})
    client = owned._upload()
    owned.close()
    assert client.is_closed


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (make_train_episode("e", step=12), 12),
        (make_train_episode("e", step=12, work="eval"), None),
        (make_episode("e"), None),
        ({"run": {"type": "train", "work": {"type": "train", "step": 3}}}, 3),
        ({"run": {"type": "eval"}}, None),
        ({"run": {"type": "train", "work": {"type": "train", "step": True}}}, None),
        ("not a record", None),
    ],
    ids=["train", "eval-work", "unstamped", "json", "json-eval", "bool-step", "string"],
)
def test_training_step_reads_the_episode_provenance(record, expected):
    assert training_step(record) == expected


def test_the_default_encoder_writes_the_viewer_table():
    """The real Parquet encoder, when pyarrow is installed: one row per episode
    with the RFT-only columns layered on the shared sample projection."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    import io
    import json

    from prime_runs.projection import episodes_to_parquet_bytes, train_sample_schema

    episodes = [
        make_train_episode("e1", step=10, idx=3, reward=1.0, advantage=0.25, env_id="gsm8k"),
        make_train_episode("e2", step=10, idx=4, reward=0.0, advantage=-0.25, env_id="gsm8k"),
    ]

    payload = episodes_to_parquet_bytes(episodes, "run-abc", 10)

    assert payload is not None
    table = pq.read_table(io.BytesIO(payload))
    assert table.schema.equals(train_sample_schema())
    rows = table.to_pylist()
    assert [row["run_id"] for row in rows] == ["run-abc", "run-abc"]
    assert [row["step"] for row in rows] == [10, 10]
    assert [row["problem_id"] for row in rows] == [3, 4]
    assert [row["sample_id"] for row in rows] == [0, 1]
    assert [row["reward"] for row in rows] == [1.0, 0.0]
    assert [row["advantage"] for row in rows] == [0.25, -0.25]
    assert [row["env_name"] for row in rows] == ["gsm8k", "gsm8k"]
    trajectory = json.loads(rows[0]["trajectory"])
    assert trajectory[-1]["advantage"] == 0.25
    assert rows[0]["num_output_tokens"] == 5
    info = json.loads(rows[0]["info"])
    assert info["native_wrapper"]["id"] == "e1"
    assert isinstance(pa.Table, type)


def test_the_default_encoder_numbers_rows_from_the_offset():
    pytest.importorskip("pyarrow")
    import io

    import pyarrow.parquet as pq

    from prime_runs.projection import episodes_to_parquet_bytes

    episodes = [make_train_episode("e1", step=10, idx=3), make_train_episode("e2", step=10, idx=4)]

    payload = episodes_to_parquet_bytes(episodes, "run-abc", 10, sample_id_offset=7)

    rows = pq.read_table(io.BytesIO(payload)).to_pylist()
    assert [row["sample_id"] for row in rows] == [7, 8]
    assert [row["problem_id"] for row in rows] == [3, 4]


def test_the_default_encoder_skips_episodes_without_a_trajectory():
    pytest.importorskip("pyarrow")
    from _fakes import Episode, Trace

    from prime_runs.projection import episodes_to_parquet_bytes

    bare = Episode(id="no-branches", traces=[Trace(id="t", branches=[])])
    assert episodes_to_parquet_bytes([bare], "run-abc", 10) is None
    assert episodes_to_parquet_bytes([], "run-abc", 10) is None


def test_a_forked_child_drops_the_inherited_storage_client(make_platform_client, rft_routes):
    """The socket belongs to the parent: the child must neither reuse it nor
    close it (a close would send close_notify on the parent's connection)."""
    handler = RecordingHandler(rft_routes)
    owned = RftSamplesSink(make_platform_client(handler), encoder=Encoder())
    owned.start("run-abc", {})
    inherited = owned._upload()

    owned.reset_after_fork()

    assert owned._upload_client is None
    assert not inherited.is_closed  # dropped, not closed
    assert owned._upload() is not inherited
    owned.close()

    injected = RftSamplesSink(
        make_platform_client(handler), encoder=Encoder(), upload_client=StorageHandler().client()
    )
    injected.start("run-abc", {})
    injected.reset_after_fork()
    with pytest.raises(SinkWriteError) as info:
        injected.write([make_train_episode("e1", step=10)])
    assert isinstance(info.value.cause, RuntimeError)
    assert "after a fork" in str(info.value.cause)


def test_the_sink_is_registered_for_fork_resets():
    from prime_runs import _fork
    from prime_runs.sinks import RftSamplesSink

    sink = RftSamplesSink(object(), encoder=Encoder())  # type: ignore[arg-type]

    assert sink in _fork._registry
