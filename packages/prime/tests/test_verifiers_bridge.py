import json

import pytest
import typer
from prime_cli.commands.env import compute_content_hash
from prime_cli.verifiers_bridge import (
    ResolvedEnvironment,
    _compute_local_content_hash,
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
        _resolve_local_env_display("simpleqa", env_dir, object())
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


def test_run_eval_uploads_even_when_local_env_is_ahead(monkeypatch, capsys):
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

    assert captured["env_name"] == "simpleqa"
    assert captured["model"] == "openai/gpt-4.1-mini"
    assert captured["job_id"] == "job-123"
    assert captured["upstream_slug"] == "alice/simpleqa"

    output = capsys.readouterr().out
    assert "tracked upstream anyway" in output


def test_run_eval_errors_when_upload_is_required_but_no_upstream_exists(monkeypatch, capsys):
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
        ),
    )
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.find_environment_metadata", lambda **kwargs: None
    )

    with pytest.raises(typer.Exit) as exc_info:
        run_eval_passthrough(
            environment="simpleqa",
            passthrough_args=[],
            skip_upload=False,
            env_path=None,
        )

    assert exc_info.value.exit_code == 1
    output = capsys.readouterr().out
    assert "Failed to push results to hub:" in output
    assert "no upstream environment found" in output
