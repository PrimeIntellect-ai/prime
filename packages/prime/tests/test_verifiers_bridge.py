import json
from typing import Any, cast

import pytest
import typer
from prime_cli.commands.env import compute_content_hash
from prime_cli.core import APIClient
from prime_cli.verifiers_bridge import (
    ResolvedEnvironment,
    _compute_local_content_hash,
    _fetch_remote_env_details,
    _resolve_local_env_display,
    run_eval_passthrough,
)


def _create_env(env_dir):
    env_dir.mkdir(parents=True)
    (env_dir / "pyproject.toml").write_text('[project]\nname = "simpleqa"\nversion = "0.1.0"\n')
    (env_dir / "README.md").write_text("# simpleqa\n")
    (env_dir / "simpleqa.py").write_text("def load_environment():\n    return None\n")
    package_dir = env_dir / "prompts"
    package_dir.mkdir()
    (package_dir / "system.txt").write_text("answer carefully\n")


def test_compute_local_content_hash_matches_env_push_hash(tmp_path):
    env_dir = tmp_path / "simpleqa"
    _create_env(env_dir)

    assert _compute_local_content_hash(env_dir) == compute_content_hash(env_dir)


def test_fetch_remote_env_details_uses_versions_endpoint_for_content_hash():
    class DummyClient:
        def get(self, endpoint):
            if endpoint == "/environmentshub/alice/simpleqa/@latest":
                return {
                    "data": {
                        "name": "simpleqa",
                        "sha256": "artifact-sha-not-source-hash",
                    }
                }
            if endpoint == "/environmentshub/alice/simpleqa/versions":
                return {
                    "data": {
                        "versions": [
                            {
                                "version": "0.1.4 (latest)",
                                "sha256": "a" * 64,
                            }
                        ]
                    }
                }
            raise AssertionError(f"Unexpected endpoint: {endpoint}")

    details = _fetch_remote_env_details(cast(Any, DummyClient()), "alice", "simpleqa")

    assert details is not None
    assert details["sha256"] == "artifact-sha-not-source-hash"
    assert details["latest_version"] == {
        "semantic_version": "0.1.4",
        "content_hash": "a" * 64,
    }


def test_resolve_local_env_display_recognizes_in_sync_metadata_connected_env(tmp_path, monkeypatch):
    env_dir = tmp_path / "simpleqa"
    _create_env(env_dir)

    prime_dir = env_dir / ".prime"
    prime_dir.mkdir()
    (prime_dir / ".env-metadata.json").write_text(
        json.dumps({"owner": "alice", "name": "simpleqa", "environment_id": "env-123"})
    )

    remote_hash = compute_content_hash(env_dir)
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge._fetch_remote_env_details",
        lambda client, owner_slug, env_name, version="latest": {
            "latest_version": {"content_hash": remote_hash}
        },
    )

    env_display_id, platform_slug, platform_url, recommend_push, push_reason = (
        _resolve_local_env_display("simpleqa", env_dir, APIClient(api_key="test-key"))
    )

    assert env_display_id == "alice/simpleqa"
    assert platform_slug == "alice/simpleqa"
    assert platform_url is not None
    assert recommend_push is False
    assert push_reason is None


class DummyConfig:
    api_key = "test-api-key"
    inference_url = None
    team_id = None


class DummyPlugin:
    eval_module = "verifiers.cli.commands.eval"

    def build_module_command(self, module: str, args: list[str]) -> list[str]:
        return [module, *args]


def _setup_eval_passthrough(monkeypatch):
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.load_verifiers_prime_plugin", lambda console: DummyPlugin()
    )
    monkeypatch.setattr("prime_cli.verifiers_bridge.Config", lambda: DummyConfig())
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge._add_default_inference_and_key_args",
        lambda args, config: (list(args), {}, "openai/gpt-4.1-mini", None),
    )
    monkeypatch.setattr("prime_cli.verifiers_bridge._validate_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge._preflight_inference_billing",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge._build_job_id", lambda *args, **kwargs: "job-123"
    )
    monkeypatch.setattr("prime_cli.verifiers_bridge._run_command", lambda *args, **kwargs: None)


def test_run_eval_requires_publish_before_upload_when_local_env_is_ahead(monkeypatch, capsys):
    _setup_eval_passthrough(monkeypatch)
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge._prepare_single_environment",
        lambda *args, **kwargs: ResolvedEnvironment(
            original="simpleqa",
            env_name="simpleqa",
            install_mode="local",
            env_display_id="simpleqa (local - ahead of alice/simpleqa)",
            platform_slug="alice/simpleqa",
            platform_url="https://app.primeintellect.ai/dashboard/environments/alice/simpleqa",
            recommend_push=True,
            push_reason="ahead",
            local_env_path=None,
        ),
    )
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("typer.confirm", lambda *args, **kwargs: True)

    push_calls = []

    def fake_publish_environment_for_eval(resolved):
        push_calls.append(resolved.platform_slug)

    monkeypatch.setattr(
        "prime_cli.verifiers_bridge._publish_environment_for_eval",
        fake_publish_environment_for_eval,
    )
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge._resolve_environment_reference",
        lambda env_name, env_dir_path: ResolvedEnvironment(
            original=env_name,
            env_name=env_name,
            install_mode="local",
            env_display_id="alice/simpleqa",
            upstream_slug="alice/simpleqa",
            platform_slug="alice/simpleqa",
            platform_url="https://app.primeintellect.ai/dashboard/environments/alice/simpleqa",
            recommend_push=False,
            local_env_path=None,
        ),
    )

    captured = {}

    def fake_push_eval_results_to_hub(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.push_eval_results_to_hub",
        fake_push_eval_results_to_hub,
    )

    run_eval_passthrough(
        environment="simpleqa",
        passthrough_args=[],
        skip_upload=False,
        env_path=None,
    )

    assert push_calls == ["alice/simpleqa"]
    assert captured["env_name"] == "simpleqa"
    assert captured["model"] == "openai/gpt-4.1-mini"
    assert captured["job_id"] == "job-123"
    assert captured["upstream_slug"] == "alice/simpleqa"

    output = capsys.readouterr().out
    assert "Cannot push evaluation results:" in output
    assert "reproducible" in output
    assert output.count("Environment URL:") == 1


def test_run_eval_fails_when_local_env_diverges_in_non_interactive_mode(monkeypatch, capsys):
    _setup_eval_passthrough(monkeypatch)
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge._prepare_single_environment",
        lambda *args, **kwargs: ResolvedEnvironment(
            original="simpleqa",
            env_name="simpleqa",
            install_mode="local",
            env_display_id="simpleqa (local only)",
            recommend_push=True,
            push_reason="local_only",
            local_env_path=None,
        ),
    )
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    with pytest.raises(typer.Exit) as exc_info:
        run_eval_passthrough(
            environment="simpleqa",
            passthrough_args=[],
            skip_upload=False,
            env_path=None,
        )

    assert cast(typer.Exit, exc_info.value).exit_code == 1
    output = capsys.readouterr().out
    assert "Cannot push evaluation results:" in output
    assert "reproducible" in output
    assert "Publish the current local version with:" in output
