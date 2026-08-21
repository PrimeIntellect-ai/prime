"""``init()``: mode resolution, offline runs, online runs."""

import json
import os

import pytest
from _fakes import make_episode, make_trace
from conftest import RecordingHandler

import prime_runs as pr
from prime_runs.exceptions import ConfigurationError
from prime_runs.models import RunStatus

# ------------------------------------------------------------------- offline


def test_an_offline_run_is_a_real_run(tmp_path):
    run = pr.init(name="local", environments=["gsm8k"], mode="offline", dir=str(tmp_path))

    assert run.id.startswith("offline-")
    assert run.url == str((tmp_path / run.id).resolve())
    assert run.mode == "offline"

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
    assert all(record["run"] == {"id": run.id, "type": "eval"} for record in records)


def test_episodes_are_written_to_their_own_file(tmp_path):
    run = pr.init(mode="offline", dir=str(tmp_path))
    run.log_traces([make_episode("ep-1", [make_trace()])])
    run.finish()

    assert (tmp_path / run.id / "records" / "episode.jsonl").exists()


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

    run.log_traces([{"id": "t1"}])
    run.finish(summary={"avg_reward": 1.0})

    assert run.id
    assert run.status is RunStatus.COMPLETED
    assert not list(tmp_path.iterdir())


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


def test_a_second_init_in_one_process_opens_its_own_run(tmp_path):
    first = pr.init(mode="offline", dir=str(tmp_path))
    first.finish()

    second = pr.init(mode="offline", dir=str(tmp_path))
    second.finish()

    assert second.id != first.id
    assert (tmp_path / second.id / "run.json").exists()


# --------------------------------------------------------------------- fork


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is POSIX-only")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_a_forked_child_does_not_duplicate_the_parents_records_or_close_its_run(tmp_path):
    """At fork time the parent has records in the upload queue. The child
    inherits a copy; writing them would put every record in the file twice,
    and the inherited atexit hook must not finalize the parent's run."""
    run = pr.init(mode="offline", dir=str(tmp_path))
    run.log_traces([{"id": f"parent-{n}"} for n in range(5)])

    pid = os.fork()
    if pid == 0:  # pragma: no cover - asserted through the child's exit code
        code = 0
        try:
            run.log_traces([{"id": "child-1"}])
            run.flush()
            run._on_process_exit()
            state = json.loads((tmp_path / run.id / "run.json").read_text())
            if state["status"] != RunStatus.RUNNING.value:
                code = 4
        except BaseException:
            code = 2
        finally:
            os._exit(code)

    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    run.finish()

    lines = (tmp_path / run.id / "records" / "trace.jsonl").read_text().splitlines()
    ids = [json.loads(line)["id"] for line in lines]

    assert sorted(ids) == sorted([f"parent-{n}" for n in range(5)] + ["child-1"])


def test_offline_records_are_on_disk_before_any_flush(tmp_path):
    """Nothing may sit in a process-local write buffer a fork could copy."""
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
    run = pr.init(mode="offline", dir=str(tmp_path), on_error="raise")
    run.log_traces([{"id": "bad", "reward": float("nan")}])

    with pytest.raises(ValueError, match="Out of range float values"):
        run.flush()

    path = tmp_path / run.id / "records" / "trace.jsonl"
    assert path.read_text() == ""
    run.finish()
