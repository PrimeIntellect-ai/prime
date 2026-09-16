"""Exercise exclusive routing through init, the worker, and real Traces HTTP calls."""

import httpx
import pytest
from _fakes import make_episode, make_trace, make_train_episode
from conftest import FakeSink, RecordingHandler
from prime_traces import TracesClient

import prime_runs as pr


@pytest.fixture
def uploads(monkeypatch):
    calls = []
    replies = []

    def handle(request):
        calls.append(request)
        if replies:
            return replies.pop(0)
        return httpx.Response(200, json={"upload_id": "a" * 64, "status": "committed"})

    monkeypatch.setattr(
        "prime_runs.sinks.traces.TracesClient",
        lambda **kwargs: TracesClient(transport=httpx.MockTransport(handle), **kwargs),
    )
    return calls, replies


@pytest.fixture
def open_run(monkeypatch, make_platform_client, eval_routes, rft_routes, uploads):
    def create(kind="eval", **kwargs):
        handler = RecordingHandler(eval_routes if kind == "eval" else rft_routes)
        monkeypatch.setattr(
            "prime_runs.run.PlatformClient", lambda **_: make_platform_client(handler)
        )
        legacy_training = FakeSink(name="rft_samples")
        monkeypatch.setattr("prime_runs.run.RftSamplesSink", lambda _: legacy_training)
        run = pr.init(
            kind=kind,
            model="model",
            environments=["gsm8k"],
            api_key="test-key",
            team_id="team-1",
            **kwargs,
        )
        return run, handler, legacy_training

    return create


def denial(code="service_not_enabled", status=403):
    return httpx.Response(status, json={"error": {"code": code, "message": code}})


@pytest.mark.parametrize("kind", ["eval", "train"])
def test_beta_uploads_only_to_traces_but_keeps_run_lifecycle_and_metrics(open_run, uploads, kind):
    calls, _ = uploads
    run, platform, legacy = open_run(kind)
    episode = make_episode() if kind == "eval" else make_train_episode(step=10)
    with run:
        run.log_episodes([episode])
        run.flush()
        run.log_episodes([episode])
        if kind == "train":
            run.log_metrics({"loss": 0.5}, step=10)

    assert len(calls) == 2
    assert all(request.headers["X-Prime-Team-ID"] == "team-1" for request in calls)
    assert not any("/samples" in path for path in platform.paths())
    assert legacy.started == legacy.batches == []
    assert run.failed_records == {}
    assert run.errors == []
    if kind == "eval":
        finalized = platform.bodies_for("/api/v1/evaluations/eval-abc/finalize")[0]
        assert finalized["metrics"]["prime_runs"] == {"traces_episodes_written": 2}
    else:
        assert legacy.closed
        assert platform.bodies_for("/api/v1/rft/metrics")[0]["metrics"]["loss"] == 0.5
        assert platform.bodies_for("/api/v1/rft/finalize")


@pytest.mark.parametrize("kind", ["eval", "train"])
def test_no_beta_access_routes_first_and_later_batches_to_legacy(open_run, uploads, kind):
    calls, replies = uploads
    replies.append(denial())
    run, platform, legacy = open_run(kind)
    with run:
        for index in range(2):
            episode = (
                make_episode(f"e{index}")
                if kind == "eval"
                else make_train_episode(f"e{index}", step=10)
            )
            run.log_episodes([episode])
            run.flush()

    assert len(calls) == 1
    assert run.failed_records == {}
    assert run.errors == []
    if kind == "eval":
        bodies = platform.bodies_for("/api/v1/evaluations/eval-abc/samples")
        assert [body["samples"][0]["sample_id"] for body in bodies] == ["e0", "e1"]
        assert run.summary["prime_runs"]["traces_episodes_written"] == 0
    else:
        assert len(legacy.started) == 1
        assert len(legacy.batches) == 2


@pytest.mark.parametrize(
    "code,status",
    [("forbidden", 403), ("unauthenticated", 401), ("invalid_trace", 400), ("busy", 503)],
)
def test_upload_failures_never_enable_legacy_fallback(open_run, uploads, code, status):
    _, replies = uploads
    replies.extend(denial(code, status) for _ in range(20))
    run, platform, _ = open_run()
    with run:
        for _ in range(4):
            run.log_episodes([make_episode()])
            run.flush()

    assert run.failed_records == {"traces": 4}
    assert run.summary["prime_runs"]["traces_episodes_written"] == 0
    assert not any("/samples" in path for path in platform.paths())


def test_access_revoked_after_a_commit_does_not_split_the_run(open_run, uploads):
    _, replies = uploads
    run, platform, _ = open_run()
    with run:
        run.log_episodes([make_episode("e1")])
        run.flush()
        replies.append(denial())
        run.log_episodes([make_episode("e2")])

    assert run.failed_records == {"traces": 1}
    assert run.summary["prime_runs"]["traces_episodes_written"] == 1
    assert not any("/samples" in path for path in platform.paths())


@pytest.mark.parametrize("summary_source", ["update", "finish"])
def test_counts_preserve_sibling_metrics_and_override_only_the_owned_field(
    open_run, summary_source
):
    run, platform, _ = open_run()
    summary = {
        "prime_runs": {
            "custom_metric": 1,
            "nested": {"reward": 0.5},
            "traces_episodes_written": 999,
        }
    }
    with run:
        run.log_traces([make_trace()])
        run.log_episodes([{"id": "empty", "traces": []}, make_episode("e1")])
        if summary_source == "update":
            run.update_summary(summary)
        else:
            run.finish(summary=summary)

    expected = {
        "custom_metric": 1,
        "nested": {"reward": 0.5},
        "traces_episodes_written": 1,
    }
    assert run.summary["prime_runs"] == expected
    assert summary["prime_runs"]["traces_episodes_written"] == 999
    for path in ["/api/v1/evaluations/eval-abc", "/api/v1/evaluations/eval-abc/finalize"]:
        assert platform.bodies_for(path)[0]["metrics"]["prime_runs"] == expected


@pytest.mark.parametrize("value", [None, 1, "custom", [1]])
def test_non_mapping_prime_runs_summary_is_replaced_with_receipt_count(open_run, value):
    run, _, _ = open_run()
    with run:
        run.log_episodes([make_episode()])
        run.update_summary({"prime_runs": value})

    assert run.summary["prime_runs"] == {"traces_episodes_written": 1}


def test_strict_error_policy_surfaces_failure_without_falling_back(open_run, uploads):
    _, replies = uploads
    replies.append(denial("forbidden"))
    run, platform, _ = open_run(on_error="raise")
    run.log_episodes([make_episode()])
    with pytest.raises(pr.ForbiddenError):
        run.finish()

    assert run.finished
    assert not any("/samples" in path for path in platform.paths())


def test_partial_upload_retains_committed_count_and_never_duplicates_to_legacy(
    monkeypatch, open_run, uploads
):
    _, replies = uploads
    replies.extend(
        [
            httpx.Response(200, json={"upload_id": "a" * 64, "status": "committed"}),
            denial(),
        ]
    )
    upload_records = TracesClient.upload_records

    def one_episode_per_batch(self, records, **kwargs):
        return upload_records(self, records, target_batch_bytes=1, **kwargs)

    monkeypatch.setattr(TracesClient, "upload_records", one_episode_per_batch)
    run, platform, _ = open_run()
    with run:
        run.log_episodes([make_episode("e1"), make_episode("e2")])

    assert run.summary["prime_runs"]["traces_episodes_written"] == 1
    assert run.failed_records
    assert not any("/samples" in path for path in platform.paths())


def test_hosted_attachment_reports_receipts_without_overwriting_launcher_config(open_run):
    run, platform, _ = open_run(id="eval-abc")
    with run:
        run.log_episodes([make_episode()])

    assert "POST /api/v1/evaluations/" not in platform.paths()
    update = platform.bodies_for("/api/v1/evaluations/eval-abc")[0]
    assert "metadata" not in update
    assert update["metrics"]["prime_runs"]["traces_episodes_written"] == 1
