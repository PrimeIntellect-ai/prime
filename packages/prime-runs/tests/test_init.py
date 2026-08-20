"""``init()``: mode resolution, offline runs, online runs, rank handling."""

import json
import os

import pytest
from _fakes import make_episode, make_trace
from conftest import RecordingHandler

import prime_runs as pr
from prime_runs.exceptions import ConfigurationError
from prime_runs.models import RunStatus
from prime_runs.run import RUN_ID_ENV

# ------------------------------------------------------------------- offline


def test_an_offline_run_is_a_real_run(tmp_path):
    """The reason producers can delete their ``--no-push`` branch: same ID, same
    status, same calls — just a different destination."""
    run = pr.init(name="local", environments=["gsm8k"], mode="offline", dir=str(tmp_path))

    assert run.id.startswith("offline-")
    assert run.url == str((tmp_path / run.id).resolve())
    assert run.mode == "offline"

    run.log({"reward": 0.5}, step=1)
    run.log_traces([{"id": "t1"}])
    run.finish(summary={"avg_reward": 0.5})

    state = json.loads((tmp_path / run.id / "run.json").read_text())
    assert state["status"] == RunStatus.COMPLETED.value
    assert state["summary"]["avg_reward"] == 0.5
    assert state["spec"]["environments"] == [{"name": "gsm8k"}]


def test_offline_records_are_written_in_the_wire_format(tmp_path):
    """The archive is a deferred upload, not a debug dump: these bytes are what
    ``TracesClient.upload_file`` sends, with the run already stamped."""
    run = pr.init(mode="offline", dir=str(tmp_path))
    run.log_traces([{"id": "t1"}, {"id": "t2"}])
    run.finish()

    lines = (tmp_path / run.id / "records" / "trace.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]

    assert [record["id"] for record in records] == ["t1", "t2"]
    assert all(record["run"]["id"] == run.id for record in records)


def test_episodes_are_written_to_their_own_file(tmp_path):
    run = pr.init(mode="offline", dir=str(tmp_path))
    run.log_traces([make_episode("ep-1", [make_trace()])])
    run.finish()

    assert (tmp_path / run.id / "records" / "episode.jsonl").exists()


def test_offline_metrics_are_a_time_series(tmp_path):
    run = pr.init(mode="offline", dir=str(tmp_path))
    run.log({"loss": 2.0}, step=1)
    run.log({"loss": 1.0}, step=2)
    run.flush()
    run.finish()

    lines = (tmp_path / run.id / "metrics.jsonl").read_text().splitlines()
    assert [json.loads(line)["loss"] for line in lines] == [2.0, 1.0]


def test_a_record_that_already_names_a_run_is_left_alone(tmp_path):
    """Producers stamp the run themselves; two sources of truth for the run ID
    is how traces end up on the wrong run."""
    run = pr.init(mode="offline", dir=str(tmp_path))
    run.log_traces([{"id": "t1", "run": {"id": "someone-elses-run"}}])
    run.finish()

    record = json.loads((tmp_path / run.id / "records" / "trace.jsonl").read_text())
    assert record["run"]["id"] == "someone-elses-run"


# -------------------------------------------------------------------- modes


def test_no_api_key_degrades_to_offline_rather_than_skipping_the_run(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        run = pr.init(name="local", dir=str(tmp_path))

    assert run.mode == "offline"
    assert "offline" in caplog.text
    run.finish()


def test_the_mode_can_be_set_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("PRIME_RUNS_MODE", "disabled")

    run = pr.init(name="local", api_key="test-key", dir=str(tmp_path))

    assert run.mode == "disabled"
    run.finish()


def test_an_unknown_mode_is_rejected(tmp_path):
    with pytest.raises(ConfigurationError, match="not one of"):
        pr.init(mode="sideways", dir=str(tmp_path))


def test_a_disabled_run_still_answers_every_call(tmp_path):
    """Same object shape, so producer code needs no branching."""
    run = pr.init(mode="disabled", dir=str(tmp_path))

    run.log({"reward": 1.0}, step=1)
    run.log_traces([{"id": "t1"}])
    run.finish(summary={"avg_reward": 1.0})

    assert run.id
    assert run.status is RunStatus.COMPLETED
    assert not list(tmp_path.iterdir())


def test_training_runs_are_not_supported_yet(tmp_path):
    with pytest.raises(ConfigurationError, match="training runs"):
        pr.init(kind="train", api_key="test-key", environments=["gsm8k"])


# --------------------------------------------------------------------- rank


def test_a_non_primary_rank_with_no_run_to_join_records_nothing(monkeypatch, tmp_path):
    """Otherwise rank 3 creates a second run for the same job."""
    monkeypatch.setenv("RANK", "3")

    run = pr.init(name="local", api_key="test-key", environments=["gsm8k"])

    assert run.mode == "disabled"
    assert run.is_primary is False
    run.finish()


def test_a_non_primary_offline_rank_without_a_run_to_join_records_nothing(monkeypatch, tmp_path):
    """An offline rank must not create a run it is forbidden to finalize."""
    monkeypatch.setenv("RANK", "3")

    run = pr.init(mode="offline", dir=str(tmp_path))

    assert run.mode == "disabled"
    assert run.is_primary is False
    run.finish()
    assert not list(tmp_path.iterdir())


def test_a_run_id_in_the_environment_is_joined_not_recreated(monkeypatch, tmp_path):
    monkeypatch.setenv("DP_RANK", "2")
    monkeypatch.setenv(RUN_ID_ENV, "offline-shared")

    run = pr.init(mode="offline", dir=str(tmp_path))

    assert run.id == "offline-shared"
    assert run.is_primary is False
    run.finish()


def test_init_publishes_the_run_id_for_child_processes(tmp_path):
    """Forked workers and subprocess launchers join the run their parent opened
    instead of each opening their own."""
    run = pr.init(mode="offline", dir=str(tmp_path))

    assert os.environ[RUN_ID_ENV] == run.id
    run.finish()


# ------------------------------------------------------------------- online


@pytest.fixture
def online(monkeypatch, make_platform_client, eval_routes):
    """``init(mode="online")`` wired to a MockTransport."""

    def _init(routes=None, **kwargs):
        handler = RecordingHandler(routes or eval_routes)
        monkeypatch.setattr(
            "prime_runs.run.PlatformClient", lambda **_: make_platform_client(handler)
        )
        run = pr.init(
            name="test-run",
            environments=["gsm8k"],
            model="Qwen3-8B",
            framework="verifiers",
            api_key="test-key",
            traces=False,
            **kwargs,
        )
        return run, handler

    return _init


def test_samples_use_a_separate_client_from_run_finalization(
    monkeypatch, make_platform_client, eval_routes
):
    handler = RecordingHandler(eval_routes)

    class TrackingClient:
        def __init__(self):
            self._delegate = make_platform_client(handler)
            self.closed = False

        def get(self, *args, **kwargs):
            return self._delegate.get(*args, **kwargs)

        def post(self, *args, **kwargs):
            return self._delegate.post(*args, **kwargs)

        def put(self, *args, **kwargs):
            return self._delegate.put(*args, **kwargs)

        def close(self):
            self.closed = True

    clients = []

    def make_client(**kwargs):
        client = TrackingClient()
        clients.append(client)
        return client

    monkeypatch.setattr("prime_runs.run.PlatformClient", make_client)
    run = pr.init(
        name="test-run",
        environments=["gsm8k"],
        api_key="test-key",
        traces=False,
        handle_signals=False,
    )
    assert len(clients) == 2

    # Model UploadWorker.close() timing out: it intentionally leaves its sink
    # open, while lifecycle finalization still closes the backend transport.
    run._worker.close = lambda timeout=None: None
    run.finish()

    backend_client, samples_client = clients
    assert backend_client.closed is True
    assert samples_client.closed is False
    run._worker.sinks[0].close()
    assert samples_client.closed is True


def test_an_online_run_returns_the_platforms_id_and_viewer_url(online):
    run, _ = online()

    assert run.id == "eval-abc"
    assert run.url == "https://app.example/dashboard/evaluations/eval-abc"
    assert run.mode == "online"
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
    """The whole surface, in the order verifiers will call it."""
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


# ------------------------------------------------------- run id inheritance


def test_a_second_init_in_one_process_opens_its_own_run(tmp_path):
    """init() exports PRIME_RUN_ID for child processes. Reading our own export
    back would silently attach the second eval to the first, and it would never
    create or finalize a run of its own."""
    first = pr.init(mode="offline", dir=str(tmp_path))
    first.finish()

    second = pr.init(mode="offline", dir=str(tmp_path))
    second.finish()

    assert second.id != first.id
    assert (tmp_path / second.id / "run.json").exists()


def test_a_finished_run_stops_advertising_itself(tmp_path):
    run = pr.init(mode="offline", dir=str(tmp_path))
    assert os.environ[RUN_ID_ENV] == run.id

    run.finish()

    assert RUN_ID_ENV not in os.environ


def test_an_id_inherited_from_a_parent_process_is_joined(monkeypatch, tmp_path):
    """The env var without a matching PID belongs to an ancestor."""
    monkeypatch.setenv(RUN_ID_ENV, "offline-from-parent")

    run = pr.init(mode="offline", dir=str(tmp_path))

    assert run.id == "offline-from-parent"
    assert run.is_primary is False
    run.finish()


def test_an_explicit_id_is_a_resume_and_still_finalizes(online):
    """Resuming after a crash has to be able to close the run out; only an ID
    picked up from the environment belongs to someone else."""
    run, handler = online(id="eval-abc")

    run.finish(summary={"avg_reward": 1.0})

    assert "POST /api/v1/evaluations/eval-abc/finalize" in handler.paths()


def test_resuming_preserves_existing_config_and_summary(online, eval_routes):
    routes = dict(eval_routes)
    routes["GET /api/v1/evaluations/eval-abc"] = {
        **routes["GET /api/v1/evaluations/eval-abc"],
        "metadata": {"before_crash": True, "overridden": "old"},
        "metrics": {"old_reward": 0.5, "overridden": "old"},
    }

    run, handler = online(
        routes=routes,
        id="eval-abc",
        config={"overridden": "new"},
        summary={"overridden": "new"},
    )
    run.update_config({"after_resume": True})
    run.log({"new_reward": 1.0})
    run.finish()

    update = handler.bodies_for("/api/v1/evaluations/eval-abc")[0]
    assert update["metadata"] == {
        "before_crash": True,
        "overridden": "new",
        "after_resume": True,
    }
    assert update["metrics"] == {
        "old_reward": 0.5,
        "overridden": "new",
        "new_reward": 1.0,
    }


def test_init_forwards_the_finish_timeout(tmp_path):
    run = pr.init(mode="offline", dir=str(tmp_path), finish_timeout=0.25)

    assert run._finish_timeout == 0.25
    run.finish()


def test_an_id_inherited_from_the_environment_does_not_finalize(monkeypatch, online):
    monkeypatch.setenv(RUN_ID_ENV, "eval-abc")

    run, handler = online()
    run.finish()

    assert "POST /api/v1/evaluations/eval-abc/finalize" not in handler.paths()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is POSIX-only")
# Forking a threaded process is exactly the situation under test — hosted evals
# do it, and the uploader thread is why the SDK needs a fork hook at all.
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_a_forked_child_joins_the_run_without_duplicating_the_parents_records(tmp_path):
    """The end-to-end shape hosted evals actually hit.

    At fork time the parent has records in the upload queue and bytes in the
    sink's write buffer. The child inherits copies of both; writing them would
    put every one of those records in the file twice, and opening its own run
    would split one job across two.
    """
    import json

    run = pr.init(mode="offline", dir=str(tmp_path), handle_signals=False)
    run.log_traces([{"id": f"parent-{n}"} for n in range(5)])

    pid = os.fork()
    if pid == 0:  # pragma: no cover - asserted through the child's exit code
        code = 0
        try:
            child = pr.init(mode="offline", dir=str(tmp_path), handle_signals=False)
            if child.id != run.id:
                code = 1
            if child.is_primary:
                code = 3
            child.log_traces([{"id": "child-1"}])
            child.finish()
            # The original handle is inherited too. Its atexit callback must be
            # harmless in the child: the parent still owns this lifecycle.
            run._on_process_exit()
            state = json.loads((tmp_path / run.id / "run.json").read_text())
            if state["status"] != RunStatus.RUNNING.value:
                code = 4
        except BaseException:
            code = 2
        finally:
            os._exit(code)

    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0, "the child did not join the parent's run"
    run.finish()

    lines = (tmp_path / run.id / "records" / "trace.jsonl").read_text().splitlines()
    ids = [json.loads(line)["id"] for line in lines]

    assert sorted(ids) == sorted([f"parent-{n}" for n in range(5)] + ["child-1"])


def test_offline_records_are_on_disk_before_any_flush(tmp_path):
    """Nothing may sit in a process-local write buffer.

    A buffered writer holds records in memory until it decides to flush, and a
    fork copies that buffer — after which both processes eventually write it and
    every buffered record lands in the file twice. Reading the file back through
    a separate handle, with no flush and no close, is what proves the buffer is
    not there to be copied.
    """
    run = pr.init(mode="offline", dir=str(tmp_path))
    run.log_traces([{"id": "t1"}])
    run.flush()

    path = tmp_path / run.id / "records" / "trace.jsonl"
    assert path.read_text().count('"t1"') == 1

    run.log_traces([{"id": "t2"}])
    run.flush()
    assert path.read_text().count('"t2"') == 1

    run.finish()
    assert len(path.read_text().splitlines()) == 2


def test_offline_records_reject_nonfinite_json_instead_of_writing_invalid_jsonl(tmp_path):
    run = pr.init(
        mode="offline",
        dir=str(tmp_path),
        handle_signals=False,
        on_error="raise",
    )
    run.log_traces([{"id": "bad", "reward": float("nan")}])

    with pytest.raises(ValueError, match="Out of range float values"):
        run.flush()

    path = tmp_path / run.id / "records" / "trace.jsonl"
    assert path.read_text() == ""
    run.finish()
