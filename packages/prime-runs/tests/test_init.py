"""``init()``: mode resolution, disabled runs, online runs."""

import os

import pytest
from _fakes import make_episode, make_trace
from conftest import RecordingHandler

import prime_runs as pr
from prime_runs.exceptions import ConfigurationError
from prime_runs.models import RunStatus

# -------------------------------------------------------------------- modes


def test_no_api_key_disables_the_run_and_says_so(caplog):
    """Loudly, not silently: a user who forgot ``prime login`` must not believe
    the run was tracked."""
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
    """Same object shape, so producer code needs no branching — and nothing
    touches the network or the filesystem."""
    run = pr.init(mode="disabled")

    run.log_traces([{"id": "t1"}])
    run.finish(summary={"avg_reward": 1.0})

    assert run.id.startswith("disabled-")
    assert run.url is None
    assert run.status is RunStatus.COMPLETED
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
    """Traces is the system of record; the sample table is what the viewer reads."""
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


def test_finishing_an_online_run_finalizes_it_with_its_metrics(online):
    run, handler = online()

    run.finish(summary={"avg_reward": 0.9})

    body = handler.bodies_for("/api/v1/evaluations/eval-abc/finalize")[0]
    assert body["metrics"]["avg_reward"] == 0.9


def test_the_end_to_end_shape_a_producer_writes(online):
    """The whole surface, in the order verifiers calls it."""
    from prime_runs import metrics

    episodes = [make_episode(f"ep-{n}", [make_trace(idx=n, reward=float(n))]) for n in range(3)]

    run, handler = online()
    for episode in episodes:
        run.log_traces([episode])
    run.finish(summary=metrics.from_episodes(episodes))

    assert handler.paths().count("POST /api/v1/evaluations/eval-abc/samples") == 3
    finalize = handler.bodies_for("/api/v1/evaluations/eval-abc/finalize")[0]
    assert finalize["metrics"]["avg_reward"] == 1.0
    assert run.status is RunStatus.COMPLETED
    assert run.errors == []


def test_a_second_init_in_one_process_opens_its_own_run():
    first = pr.init(mode="disabled")
    first.finish()

    second = pr.init(mode="disabled")
    second.finish()

    assert second.id != first.id


# --------------------------------------------------------------------- fork


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is POSIX-only")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_a_forked_child_does_not_duplicate_the_parents_records_or_close_its_run(online):
    """At fork time the parent may have records in the upload queue. The child
    inherits a copy; writing them would upload every record twice, and the
    inherited atexit hook must not finalize the parent's run."""
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
