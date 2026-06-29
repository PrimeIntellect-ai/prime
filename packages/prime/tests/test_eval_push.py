import json

import pytest
from click.testing import CliRunner
from prime_cli.commands.evals import (
    _push_samples_with_progress,
    _push_single_eval,
)
from prime_cli.main import app
from typing_extensions import cast

runner = CliRunner()


def test_push_eval_rejects_public_with_eval_id(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    (tmp_path / "metadata.json").write_text(json.dumps({"env": "gsm8k", "model": "gpt-4"}))
    (tmp_path / "results.jsonl").write_text("")

    result = runner.invoke(
        app,
        ["eval", "push", ".", "--eval-id", "eval-123", "--public"],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )

    assert result.exit_code == 1, result.output
    assert "cannot be used with --eval-id" in result.output


def test_push_eval_forwards_name_override(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    captured = {}

    def fake_push_single_eval(config_path, env_slug, run_id, eval_id, is_public, name):
        captured.update(
            {
                "config_path": config_path,
                "env_slug": env_slug,
                "run_id": run_id,
                "eval_id": eval_id,
                "is_public": is_public,
                "name": name,
            }
        )
        return "eval-123"

    monkeypatch.setattr("prime_cli.commands.evals._push_single_eval", fake_push_single_eval)

    (tmp_path / "metadata.json").write_text(json.dumps({"env": "gsm8k", "model": "gpt-4"}))
    (tmp_path / "results.jsonl").write_text("")

    result = runner.invoke(
        app,
        ["eval", "push", ".", "--eval-id", "eval-123", "--name", "custom eval"],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "config_path": ".",
        "env_slug": None,
        "run_id": None,
        "eval_id": "eval-123",
        "is_public": False,
        "name": "custom eval",
    }


def test_push_samples_with_progress_supports_old_prime_evals_client(monkeypatch):
    calls = []

    class DummyClient:
        def push_samples(self, evaluation_id, samples):
            calls.append((evaluation_id, samples))

    class DummyConsole:
        is_terminal = True

    class UnexpectedProgress:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("old prime-evals clients should skip progress callbacks")

    monkeypatch.setattr("prime_cli.commands.evals.console", DummyConsole())
    monkeypatch.setattr("prime_cli.commands.evals.Progress", UnexpectedProgress)

    samples = [{"example_id": "1"}]
    _push_samples_with_progress(DummyClient(), "eval-123", samples)

    assert calls == [("eval-123", samples)]


def test_push_eval_cli_supports_old_prime_evals_client(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    captured = {}

    class TerminalConsole:
        is_terminal = True

        def print(self, *_args, **_kwargs):
            return None

    class OldEvalsClient:
        def __init__(self, _api_client):
            return None

        def create_evaluation(self, **kwargs):
            captured["create"] = kwargs
            return {"evaluation_id": "eval-123"}

        def push_samples(self, evaluation_id, samples):
            captured["push"] = (evaluation_id, samples)

        def finalize_evaluation(self, evaluation_id, metrics=None):
            captured["finalize"] = (evaluation_id, metrics)
            return {}

    monkeypatch.setattr("prime_cli.commands.evals.console", TerminalConsole())
    monkeypatch.setattr("prime_cli.commands.evals.APIClient", lambda: object())
    monkeypatch.setattr("prime_cli.commands.evals.EvalsClient", OldEvalsClient)

    (tmp_path / "metadata.json").write_text(
        json.dumps({"env": "owner/gsm8k", "model": "gpt-4", "avg_reward": 1.0})
    )
    (tmp_path / "results.jsonl").write_text(json.dumps({"id": 1, "reward": 1.0}) + "\n")

    result = runner.invoke(
        app,
        ["eval", "push", "."],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )

    assert result.exit_code == 0, result.output
    assert captured["create"]["environments"] == [{"slug": "owner/gsm8k"}]
    assert captured["push"] == ("eval-123", [{"id": 1, "reward": 1.0, "example_id": 1}])
    assert captured["finalize"] == ("eval-123", {"reward": 1.0})


def test_push_samples_with_progress_skips_callback_when_signature_is_uninspectable(monkeypatch):
    calls = []

    class DummyClient:
        def push_samples(self, evaluation_id, samples):
            calls.append((evaluation_id, samples))

    class DummyConsole:
        is_terminal = True

    class UnexpectedProgress:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("uninspectable clients should skip progress callbacks")

    monkeypatch.setattr("prime_cli.commands.evals.console", DummyConsole())
    monkeypatch.setattr("prime_cli.commands.evals.Progress", UnexpectedProgress)
    monkeypatch.setattr(
        "prime_cli.commands.evals.inspect.signature",
        lambda _callable: (_ for _ in ()).throw(ValueError("no signature")),
    )

    samples = [{"example_id": "1"}]
    _push_samples_with_progress(DummyClient(), "eval-123", samples)

    assert calls == [("eval-123", samples)]


class TestPushSingleEval:
    def test_create_evaluation_defaults_to_private(self, tmp_path, monkeypatch):
        metadata = {"env": "owner/gsm8k", "model": "gpt-4"}
        (tmp_path / "metadata.json").write_text(json.dumps(metadata))
        (tmp_path / "results.jsonl").write_text("")

        captured = {}

        class DummyEvalsClient:
            def __init__(self, _api_client):
                pass

            def create_evaluation(self, **kwargs):
                captured.update(kwargs)
                return {"evaluation_id": "eval-123"}

            def finalize_evaluation(self, evaluation_id, metrics=None):
                captured["finalized_evaluation_id"] = evaluation_id
                captured["finalized_metrics"] = metrics

        monkeypatch.setattr("prime_cli.commands.evals.APIClient", lambda: object())
        monkeypatch.setattr("prime_cli.commands.evals.EvalsClient", DummyEvalsClient)

        eval_id = _push_single_eval(str(tmp_path), None, None, None)

        assert eval_id == "eval-123"
        assert captured["is_public"] is False
        assert captured["environments"] == [{"slug": "owner/gsm8k"}]

    def test_create_evaluation_requires_pushed_environment(self, tmp_path, capsys):
        metadata = {"env": "gsm8k", "model": "gpt-4"}
        (tmp_path / "metadata.json").write_text(json.dumps(metadata))
        (tmp_path / "results.jsonl").write_text("")

        with pytest.raises(SystemExit) as exc_info:
            _push_single_eval(str(tmp_path), None, None, None)

        assert cast(SystemExit, exc_info.value).code == 1
        output = capsys.readouterr().out
        assert "Evaluation uploads require a pushed environment" in output
        assert "prime env push gsm8k" in output
        assert "--env <owner>/gsm8k" in output

    def test_create_evaluation_passes_public_flag(self, tmp_path, monkeypatch):
        metadata = {"env": "owner/gsm8k", "model": "gpt-4"}
        (tmp_path / "metadata.json").write_text(json.dumps(metadata))
        (tmp_path / "results.jsonl").write_text("")

        captured = {}

        class DummyEvalsClient:
            def __init__(self, _api_client):
                pass

            def create_evaluation(self, **kwargs):
                captured.update(kwargs)
                return {"evaluation_id": "eval-123"}

            def finalize_evaluation(self, evaluation_id, metrics=None):
                captured["finalized_evaluation_id"] = evaluation_id
                captured["finalized_metrics"] = metrics

        monkeypatch.setattr("prime_cli.commands.evals.APIClient", lambda: object())
        monkeypatch.setattr("prime_cli.commands.evals.EvalsClient", DummyEvalsClient)

        eval_id = _push_single_eval(str(tmp_path), None, None, None, is_public=True)

        assert eval_id == "eval-123"
        assert captured["is_public"] is True

    def test_create_evaluation_prefers_explicit_name_override(self, tmp_path, monkeypatch):
        metadata = {"env": "owner/gsm8k", "model": "gpt-4", "eval_name": "metadata name"}
        (tmp_path / "metadata.json").write_text(json.dumps(metadata))
        (tmp_path / "results.jsonl").write_text("")

        captured = {}

        class DummyEvalsClient:
            def __init__(self, _api_client):
                pass

            def create_evaluation(self, **kwargs):
                captured.update(kwargs)
                return {"evaluation_id": "eval-123"}

            def finalize_evaluation(self, evaluation_id, metrics=None):
                captured["finalized_evaluation_id"] = evaluation_id
                captured["finalized_metrics"] = metrics

        monkeypatch.setattr("prime_cli.commands.evals.APIClient", lambda: object())
        monkeypatch.setattr("prime_cli.commands.evals.EvalsClient", DummyEvalsClient)

        eval_id = _push_single_eval(str(tmp_path), None, None, None, name="explicit override")

        assert eval_id == "eval-123"
        assert captured["name"] == "explicit override"

    def test_update_evaluation_prefers_explicit_name_override(self, tmp_path, monkeypatch):
        metadata = {"env": "gsm8k", "model": "gpt-4", "eval_name": "metadata name"}
        (tmp_path / "metadata.json").write_text(json.dumps(metadata))
        (tmp_path / "results.jsonl").write_text("")

        captured = {}

        class DummyEvalsClient:
            def __init__(self, _api_client):
                pass

            def get_evaluation(self, evaluation_id):
                captured["checked_evaluation_id"] = evaluation_id
                return {"evaluation_id": evaluation_id}

            def update_evaluation(self, **kwargs):
                captured.update(kwargs)

            def finalize_evaluation(self, evaluation_id, metrics=None):
                captured["finalized_evaluation_id"] = evaluation_id
                captured["finalized_metrics"] = metrics

        monkeypatch.setattr("prime_cli.commands.evals.APIClient", lambda: object())
        monkeypatch.setattr("prime_cli.commands.evals.EvalsClient", DummyEvalsClient)

        eval_id = _push_single_eval(str(tmp_path), None, None, "eval-123", name="explicit override")

        assert eval_id == "eval-123"
        assert captured["checked_evaluation_id"] == "eval-123"
        assert captured["name"] == "explicit override"

    def test_push_single_eval_prints_returned_viewer_url(self, tmp_path, monkeypatch, capsys):
        metadata = {"env": "owner/gsm8k", "model": "gpt-4"}
        (tmp_path / "metadata.json").write_text(json.dumps(metadata))
        (tmp_path / "results.jsonl").write_text("")

        class DummyEvalsClient:
            def __init__(self, _api_client):
                pass

            def create_evaluation(self, **_kwargs):
                return {"evaluation_id": "eval-123"}

            def finalize_evaluation(self, evaluation_id, metrics=None):
                assert evaluation_id == "eval-123"
                assert metrics == {}
                return {
                    "viewer_url": "https://app.primeintellect.ai/dashboard/evaluations/eval-123"
                }

        monkeypatch.setattr("prime_cli.commands.evals.APIClient", lambda: object())
        monkeypatch.setattr("prime_cli.commands.evals.EvalsClient", DummyEvalsClient)
        monkeypatch.setattr(
            "prime_cli.commands.evals.get_eval_viewer_url",
            lambda _eval_id: "https://fallback.example/eval-123",
        )

        _push_single_eval(str(tmp_path), None, None, None)

        output = capsys.readouterr().out
        assert "https://app.primeintellect.ai/dashboard/evaluations/eval-123" in output
        assert "https://fallback.example/eval-123" not in output

    def test_push_single_eval_falls_back_to_configured_viewer_url(
        self, tmp_path, monkeypatch, capsys
    ):
        metadata = {"env": "owner/gsm8k", "model": "gpt-4"}
        (tmp_path / "metadata.json").write_text(json.dumps(metadata))
        (tmp_path / "results.jsonl").write_text("")

        class DummyEvalsClient:
            def __init__(self, _api_client):
                pass

            def create_evaluation(self, **_kwargs):
                return {"evaluation_id": "eval-123"}

            def finalize_evaluation(self, evaluation_id, metrics=None):
                assert evaluation_id == "eval-123"
                assert metrics == {}
                return {}

        monkeypatch.setattr("prime_cli.commands.evals.APIClient", lambda: object())
        monkeypatch.setattr("prime_cli.commands.evals.EvalsClient", DummyEvalsClient)
        monkeypatch.setattr(
            "prime_cli.commands.evals.get_eval_viewer_url",
            lambda _eval_id: "https://fallback.example/eval-123",
        )

        _push_single_eval(str(tmp_path), None, None, None)

        output = capsys.readouterr().out
        assert "https://fallback.example/eval-123" in output

    def test_push_single_eval_warns_for_skipped_artifact_rows(self, tmp_path, monkeypatch, capsys):
        metadata = {"env": "owner/gsm8k", "model": "gpt-4"}
        (tmp_path / "metadata.json").write_text(json.dumps(metadata))
        (tmp_path / "results.jsonl").write_text('{"id": 1, "reward": 1.0}\nnot json\n')

        class DummyEvalsClient:
            def __init__(self, _api_client):
                pass

            def create_evaluation(self, **_kwargs):
                return {"evaluation_id": "eval-123"}

            def push_samples(self, evaluation_id, samples):
                assert evaluation_id == "eval-123"
                assert samples == [{"id": 1, "reward": 1.0, "example_id": 1}]

            def finalize_evaluation(self, evaluation_id, metrics=None):
                assert evaluation_id == "eval-123"
                assert metrics == {}
                return {}

        monkeypatch.setattr("prime_cli.commands.evals.APIClient", lambda: object())
        monkeypatch.setattr("prime_cli.commands.evals.EvalsClient", DummyEvalsClient)

        _push_single_eval(str(tmp_path), None, None, None)

        output = capsys.readouterr().out
        assert "Warning: Skipped 1 invalid lines in results.jsonl" in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
