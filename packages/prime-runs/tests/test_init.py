"""``init()``: mode resolution, disabled runs, online runs."""

import os

import pytest
from _fakes import make_episode, make_trace
from conftest import RecordingHandler

import prime_runs as pr
from prime_runs.exceptions import ConfigurationError
from prime_runs.models import RunSpec, RunStatus

# -------------------------------------------------------------------- modes


def test_no_api_key_disables_the_run_and_says_so(caplog):
    with caplog.at_level("WARNING"):
        run = pr.init(name="local")

    assert run.mode == "disabled"
    assert "not be tracked" in caplog.text
    run.finish()


def test_the_mode_can_be_set_from_the_environment(monkeypatch):
    monkeypatch.setenv("PRIME_RUNS_MODE", "disabled")

    run = pr.init(name="local", api_key="test-key")

    assert run.mode == "disabled"
    run.finish()


def test_an_unknown_mode_is_rejected():
    with pytest.raises(ConfigurationError, match="not one of"):
        pr.init(mode="sideways")


def test_a_disabled_run_still_answers_every_call(tmp_path):
    run = pr.init(mode="disabled")

    run.log_traces([{"id": "t1"}])
    # No sinks, so nothing was queued and no uploader thread was started for it.
    assert run._worker._thread is None
    run.finish(summary={"avg_reward": 1.0})

    assert run.id.startswith("disabled-")
    assert run.url is None
    assert run.status is RunStatus.COMPLETED
    assert run.dropped_records == 0
    assert not list(tmp_path.iterdir())  # tmp_path is $HOME here


def test_online_without_an_api_key_is_a_configuration_error():
    with pytest.raises(ConfigurationError, match="needs an API key"):
        pr.init(mode="online", environments=["gsm8k"])


# ------------------------------------------------------------------- online


@pytest.fixture
def online(monkeypatch, make_platform_client, eval_routes):
    """``init(mode="online")`` wired to a MockTransport, traces sink off."""

    def _init(routes=None, **kwargs):
        handler = RecordingHandler(routes or eval_routes)
        monkeypatch.setattr(
            "prime_runs.run.PlatformClient", lambda **_: make_platform_client(handler)
        )
        monkeypatch.setattr("prime_runs.run.TracesSink", lambda **_: _NullSink())
        run = pr.init(
            name="test-run",
            environments=["gsm8k"],
            model="Qwen3-8B",
            framework="verifiers",
            api_key="test-key",
            **kwargs,
        )
        return run, handler

    return _init


class _NullSink:
    name = "traces"
    enabled = True

    def start(self, run_id, context):
        pass

    def write(self, records):
        pass

    def flush(self):
        pass

    def close(self):
        pass


def test_an_online_run_returns_the_platforms_id_and_viewer_url(online):
    run, _ = online()

    assert run.id == "eval-abc"
    assert run.url == "https://app.example/dashboard/evaluations/eval-abc"
    assert run.mode == "online"
    run.finish()


def test_eval_uploaders_keep_the_permanent_retirement(online):
    run, _ = online()

    assert run._worker._retire_cooldown is None
    run.finish()


def test_a_failed_create_closes_the_platform_client(monkeypatch):
    class FailingClient:
        def __init__(self):
            self.closed = False

        def post(self, *args, **kwargs):
            raise pr.APIError("create failed")

        def close(self):
            self.closed = True

    client = FailingClient()
    monkeypatch.setattr("prime_runs.run.PlatformClient", lambda **_: client)

    with pytest.raises(pr.APIError, match="create failed"):
        pr.init(
            name="test-run",
            environments=[{"id": "env-123"}],
            api_key="test-key",
            mode="online",
        )

    assert client.closed is True


def test_an_online_run_has_both_transports_by_default(
    monkeypatch, make_platform_client, eval_routes
):
    handler = RecordingHandler(eval_routes)
    monkeypatch.setattr("prime_runs.run.PlatformClient", lambda **_: make_platform_client(handler))
    monkeypatch.setattr("prime_traces.TracesClient", lambda **_: object())

    run = pr.init(name="test-run", environments=["gsm8k"], api_key="test-key")

    assert [sink.name for sink in run._worker.sinks] == ["traces", "eval_samples"]
    run.finish()


def test_episodes_stream_to_the_sample_table_while_the_run_is_going(online):
    run, handler = online()

    run.log_traces([make_episode("ep-1", [make_trace()])])
    run.flush()

    posted = handler.bodies_for("/api/v1/evaluations/eval-abc/samples")
    assert len(posted) == 1
    assert posted[0]["samples"][0]["sample_id"] == "ep-1"
    # Still running: the point is that the dashboard fills in as rollouts land.
    assert "POST /api/v1/evaluations/eval-abc/finalize" not in handler.paths()
    run.finish()


def test_the_end_to_end_shape_a_producer_writes(online):
    from prime_runs import metrics

    episodes = [make_episode(f"ep-{n}", [make_trace(idx=n, reward=float(n))]) for n in range(3)]

    run, handler = online()
    for episode in episodes:
        run.log_traces([episode])
    run.finish(summary=metrics.from_episodes(episodes))

    # Episodes queued while a request is in flight go out together, so the
    # request count depends on timing; every row landing exactly once does not.
    posted = handler.bodies_for("/api/v1/evaluations/eval-abc/samples")
    assert 1 <= len(posted) <= 3
    assert sorted(s["sample_id"] for body in posted for s in body["samples"]) == [
        "ep-0",
        "ep-1",
        "ep-2",
    ]
    finalize = handler.bodies_for("/api/v1/evaluations/eval-abc/finalize")[0]
    assert finalize["metrics"]["avg_reward"] == 1.0
    assert run.status is RunStatus.COMPLETED
    assert run.errors == []


# --------------------------------------------------------------------- fork


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is POSIX-only")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_a_forked_child_does_not_duplicate_the_parents_records_or_close_its_run(online):
    run, handler = online()
    run.log_traces([{"sample_id": f"parent-{n}"} for n in range(5)])

    pid = os.fork()
    if pid == 0:  # pragma: no cover - asserted through the child's exit code
        code = 0
        try:
            # ``handler`` is the child's copy: only what the child uploads lands here.
            before = len(handler.bodies_for("/api/v1/evaluations/eval-abc/samples"))
            run.log_traces([{"sample_id": "child-1"}])
            run.flush()
            run._on_process_exit()
            posted = handler.bodies_for("/api/v1/evaluations/eval-abc/samples")[before:]
            ids = [s["sample_id"] for body in posted for s in body["samples"]]
            if ids != ["child-1"]:
                code = 3
            if "POST /api/v1/evaluations/eval-abc/finalize" in handler.paths():
                code = 4
        except BaseException:
            code = 2
        finally:
            os._exit(code)

    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    run.finish()

    posted = handler.bodies_for("/api/v1/evaluations/eval-abc/samples")
    ids = [s["sample_id"] for body in posted for s in body["samples"]]
    assert sorted(ids) == [f"parent-{n}" for n in range(5)]
    assert handler.paths().count("POST /api/v1/evaluations/eval-abc/finalize") == 1


def test_records_reject_nonfinite_json_instead_of_sending_an_opaque_400(online):
    run, _ = online(on_error="raise")
    run.log_traces([{"sample_id": "bad", "reward": float("nan")}])

    with pytest.raises(ValueError, match="Out of range float values"):
        run.flush()
    run.finish()


# ----------------------------------------------------------------- training


@pytest.fixture
def online_train(monkeypatch, make_platform_client, rft_routes):
    """``init(kind="train", mode="online")`` wired to a MockTransport: traces
    sink off, sample uploads to an in-memory store, a recording encoder."""
    from conftest import StorageHandler

    from prime_runs.sinks import RftSamplesSink

    storage = StorageHandler()
    encoder_calls = []

    def encoder(episodes, run_id, step, sample_id_offset=0):
        encoder_calls.append((len(episodes), run_id, step))
        return b"parquet"

    def samples_sink(client, **kwargs):
        return RftSamplesSink(client, encoder=encoder, upload_client=storage.client(), **kwargs)

    def _init(routes=None, **kwargs):
        handler = RecordingHandler(routes or rft_routes)
        monkeypatch.setattr(
            "prime_runs.run.PlatformClient", lambda **_: make_platform_client(handler)
        )
        monkeypatch.setattr("prime_runs.run.TracesSink", lambda **_: _NullSink())
        monkeypatch.setattr("prime_runs.run.RftSamplesSink", samples_sink)
        # The RFT create response carries no viewer URL; the SDK builds one.
        monkeypatch.setenv("PRIME_FRONTEND_URL", "https://app.example")
        kwargs.setdefault("name", "test-run")
        kwargs.setdefault("model", "Qwen/Qwen3-8B")
        kwargs.setdefault("environments", ["primeintellect/vf-math"])
        kwargs.setdefault("team_id", "team-1")
        run = pr.init(kind="train", api_key="test-key", **kwargs)
        return run, handler, storage, encoder_calls

    return _init


def test_a_training_run_registers_with_the_rft_api(online_train):
    run, handler, _, _ = online_train(
        training=pr.TrainingSpec(max_steps=50, batch_size=32),
        config={"trainer": {"lr": 1e-6}},
    )

    assert run.kind == "train"
    assert run.id == "run-abc"
    assert run.url == "https://app.example/dashboard/training/run-abc"
    assert run.attached is False
    body = handler.bodies_for("/api/v1/rft/external-runs")[0]
    assert body["base_model"] == "Qwen/Qwen3-8B"
    assert body["max_steps"] == 50
    assert body["batch_size"] == 32
    assert body["environments"] == [{"id": "primeintellect/vf-math"}]
    assert body["run_config"] == {"trainer": {"lr": 1e-6}}
    assert body["team_id"] == "team-1"
    assert [sink.name for sink in run._worker.sinks] == ["traces", "rft_samples"]
    assert [sink.name for sink in run._metrics_worker.sinks] == ["rft_metrics"]
    run.finish()


def test_a_training_run_streams_metrics_and_step_samples(online_train):
    from _fakes import make_train_episode

    run, handler, storage, encoder_calls = online_train()

    run.log_metrics({"loss": 0.5}, step=10)
    run.log_episodes([make_train_episode("e1", step=10), make_train_episode("e2", step=10)])
    run.finish()

    metrics = handler.bodies_for("/api/v1/rft/metrics")
    assert len(metrics) == 1
    assert metrics[0]["run_id"] == "run-abc"
    assert metrics[0]["metrics"]["loss"] == 0.5
    assert metrics[0]["metrics"]["step"] == 10
    assert "_timestamp" in metrics[0]["metrics"]
    assert encoder_calls == [(2, "run-abc", 10)]
    assert len(storage.uploads) == 1
    assert handler.bodies_for("/api/v1/rft/samples/confirm")[0]["step"] == 10
    # Closed out through the idempotent finalize, not the status PUT.
    assert handler.bodies_for("/api/v1/rft/finalize") == [{"run_id": "run-abc", "exit_code": 0}]
    assert "PUT /api/v1/rft/external-runs/run-abc/status" not in handler.paths()
    assert run.status is RunStatus.COMPLETED
    assert run.errors == []


def test_a_failed_training_run_is_marked_failed_with_the_reason(online_train):
    run, handler, _, _ = online_train()

    with pytest.raises(ValueError):
        with run:
            raise ValueError("loss is NaN")

    body = handler.bodies_for("/api/v1/rft/external-runs/run-abc/status")[0]
    assert body == {"status": "failed", "error_message": "ValueError: loss is NaN"}
    assert "POST /api/v1/rft/finalize" not in handler.paths()


def test_training_uploaders_pause_rather_than_retire_after_an_outage(online_train):
    from prime_runs.run import TRAIN_RETIRE_COOLDOWN

    run, _, _, _ = online_train()

    assert run._worker._retire_cooldown == TRAIN_RETIRE_COOLDOWN == 300.0
    assert run._metrics_worker._retire_cooldown == TRAIN_RETIRE_COOLDOWN
    run.finish()


def test_attaching_to_a_managed_run_registers_nothing(online_train):
    run, handler, _, _ = online_train(id="run-managed")

    assert run.id == "run-managed"
    assert run.attached is True
    assert run.url == "https://app.example/dashboard/training/run-managed"
    assert "POST /api/v1/rft/external-runs" not in handler.paths()
    run.finish()
    # A clean exit still completes it.
    assert handler.bodies_for("/api/v1/rft/finalize") == [{"run_id": "run-managed", "exit_code": 0}]


def test_an_attached_run_leaves_failure_marking_to_the_launcher(online_train, caplog):
    run, handler, _, _ = online_train(id="run-managed")

    with caplog.at_level("INFO"):
        run.fail("boom")

    assert not any("status" in path for path in handler.paths())
    assert "POST /api/v1/rft/finalize" not in handler.paths()
    assert "leaving it to the launcher" in caplog.text
    assert run.status is RunStatus.FAILED


def test_a_training_run_needs_a_team(monkeypatch, make_platform_client, rft_routes):
    handler = RecordingHandler(rft_routes)
    monkeypatch.setattr("prime_runs.run.PlatformClient", lambda **_: make_platform_client(handler))

    with pytest.raises(ConfigurationError, match="team"):
        pr.init(kind="train", model="Qwen/Qwen3-8B", api_key="test-key", mode="online")
    assert handler.paths() == []


def test_a_team_outside_the_allowlist_surfaces_the_platforms_reason(
    monkeypatch, make_platform_client, rft_routes
):
    import httpx

    routes = dict(rft_routes)
    routes["POST /api/v1/rft/external-runs"] = lambda request: httpx.Response(
        403, json={"detail": "External training runs are not enabled for this team"}
    )
    handler = RecordingHandler(routes)
    monkeypatch.setattr("prime_runs.run.PlatformClient", lambda **_: make_platform_client(handler))

    with pytest.raises(pr.ForbiddenError, match="not enabled for this team"):
        pr.init(kind="train", model="m", team_id="team-1", api_key="test-key", mode="online")


def test_a_disabled_training_run_keeps_an_attached_id(tmp_path):
    run = pr.init(kind="train", mode="disabled", id="run-managed", model="m")

    assert run.id == "run-managed"
    assert run.kind == "train"
    run.log_metrics({"loss": 0.1}, step=1)
    assert run._metrics_worker._thread is None
    run.finish()
    assert run.status is RunStatus.COMPLETED


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "eval", "id": "eval-1"},
        {"kind": "eval", "training": pr.TrainingSpec()},
        {"kind": "serve"},
    ],
    ids=["eval-with-id", "eval-with-training", "unknown-kind"],
)
def test_kind_arguments_are_checked_before_anything_else(kwargs):
    with pytest.raises(ConfigurationError):
        pr.init(mode="disabled", **kwargs)


def test_training_records_are_stamped_with_the_train_run_type():
    from prime_runs.run import _sink_context
    from prime_runs.sinks.base import stamp_run

    spec = RunSpec(kind="train", model="m")
    context = _sink_context(spec)

    assert context["run_type"] == "train"
    assert stamp_run({"id": "t1"}, "run-abc", context["run_type"])["run"] == {
        "id": "run-abc",
        "type": "train",
    }
