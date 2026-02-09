import json
from pathlib import Path

import pytest
import typer
from prime_cli.commands import evals


def test_logs_cmd_routes_to_display(monkeypatch):
    captured: dict[str, object] = {}

    def fake_display(eval_id: str, tail: int, follow: bool) -> None:
        captured["eval_id"] = eval_id
        captured["tail"] = tail
        captured["follow"] = follow

    monkeypatch.setattr(evals, "_display_logs", fake_display)

    evals.logs_cmd(eval_id="eval-123", tail=50, follow=True)

    assert captured == {"eval_id": "eval-123", "tail": 50, "follow": True}


def test_stop_cmd_routes_to_hosted_stop(monkeypatch):
    captured: dict[str, str] = {}

    def fake_stop(eval_id: str) -> None:
        captured["eval_id"] = eval_id

    monkeypatch.setattr(evals, "stop_hosted_evaluation", fake_stop)

    evals.stop_cmd(eval_id="eval-999")

    assert captured["eval_id"] == "eval-999"


def test_pull_cmd_writes_verifiers_files(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(evals, "_fetch_eval_status", lambda eval_id: {"status": "COMPLETED"})

    class _APIStub:
        def get(self, endpoint: str):
            assert endpoint == "/evaluations/eval-abc/export"
            return {
                "metadata": {"env_id": "primeintellect/wordle", "model": "openai/gpt-4.1-mini"},
                "results": [{"example_id": 1, "reward": 1.0}],
            }

    monkeypatch.setattr(evals, "APIClient", lambda: _APIStub())

    out_dir = tmp_path / "pulled"
    evals.pull_cmd(eval_id="eval-abc", output_dir=str(out_dir), force=False)

    metadata = json.loads((out_dir / "metadata.json").read_text())
    results_lines = (out_dir / "results.jsonl").read_text().strip().splitlines()
    results = [json.loads(line) for line in results_lines]

    assert metadata["env_id"] == "primeintellect/wordle"
    assert metadata["model"] == "openai/gpt-4.1-mini"
    assert results == [{"example_id": 1, "reward": 1.0}]


def test_pull_cmd_rejects_non_completed(monkeypatch):
    monkeypatch.setattr(evals, "_fetch_eval_status", lambda eval_id: {"status": "RUNNING"})

    with pytest.raises(typer.Exit) as exc_info:
        evals.pull_cmd(eval_id="eval-running", output_dir=None, force=False)

    assert exc_info.value.exit_code == 1


def test_display_logs_follow_exits_nonzero_on_terminal_failure(monkeypatch):
    monkeypatch.setattr(
        evals,
        "_fetch_eval_status",
        lambda eval_id: {
            "status": "FAILED",
            "evaluation_id": eval_id,
            "error_message": "boom",
        },
    )
    monkeypatch.setattr(evals, "_fetch_logs", lambda eval_id: "")

    with pytest.raises(typer.Exit) as exc_info:
        evals._display_logs("eval-failed", tail=100, follow=True)

    assert exc_info.value.exit_code == 1


def test_display_logs_follow_sleeps_once_per_iteration(monkeypatch):
    monkeypatch.setattr(
        evals,
        "_fetch_eval_status",
        lambda eval_id: {
            "status": "RUNNING",
            "evaluation_id": eval_id,
            "total_samples": 0,
        },
    )
    monkeypatch.setattr(evals, "_fetch_logs", lambda eval_id: "")

    sleep_calls: list[int] = []

    def fake_sleep(seconds: int) -> None:
        sleep_calls.append(seconds)
        raise KeyboardInterrupt

    monkeypatch.setattr(evals.time, "sleep", fake_sleep)

    evals._display_logs("eval-running", tail=100, follow=True)

    assert sleep_calls == [5]
