from pathlib import Path

import pytest
import typer
from prime_cli.client import APIError
from prime_cli.commands.evals import (
    _create_hosted_evaluations,
    _load_hosted_eval_configs,
    _print_eval_status,
    _resolve_hosted_environment,
)
from prime_cli.main import app
from prime_cli.utils.hosted_eval import (
    HostedEvalConfig,
    clean_logs,
    filter_progress_bars,
    strip_ansi,
)
from prime_cli.verifiers_bridge import ResolvedEnvironment
from typer.testing import CliRunner

runner = CliRunner()


class TestLogCleaning:
    def test_strip_ansi_basic(self):
        assert strip_ansi("\x1b[31mRed text\x1b[0m") == "Red text"

    def test_filter_progress_bars_keeps_100_percent(self):
        text = "Progress: 100%|██████████| 10/10 [00:01<00:00]"
        result = filter_progress_bars(text)
        assert "100%" in result

    def test_filter_progress_bars_drops_partial_updates(self):
        text = "Progress: 50%|█████     | 5/10 [00:01<00:01]"
        assert filter_progress_bars(text) == ""

    def test_clean_logs_removes_ansi_and_progress(self):
        text = (
            "\x1b[32mStarting evaluation\x1b[0m\n"
            "Progress: 50%|█████     | 5/10 [00:01<00:01]\n"
            "\x1b[1mProgress: 100%|██████████| 10/10 [00:02<00:00]\x1b[0m\n"
            "\x1b[32m✓ Evaluation complete\x1b[0m"
        )
        result = clean_logs(text)
        assert "Starting evaluation" in result
        assert "✓ Evaluation complete" in result
        assert "100%" in result
        assert "50%" not in result
        assert "\x1b" not in result

    def test_clean_logs_hides_status_messages(self):
        assert clean_logs("The hosted eval is still initializing") == ""


def test_eval_list_shows_hosted_checkbox(monkeypatch):
    class DummyConfig:
        team_id = None

    class DummyClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def list_evaluations(self, **_kwargs):
            return {
                "evaluations": [
                    {
                        "evaluation_id": "hosted-1",
                        "environment_names": ["aime2024"],
                        "model_name": "openai/gpt-4.1-mini",
                        "status": "RUNNING",
                        "is_hosted": True,
                        "metadata": {"num_examples": 3, "rollouts_per_example": 2},
                    },
                    {
                        "evaluation_id": "local-1",
                        "environment_names": ["gsm8k"],
                        "model_name": "openai/gpt-4.1-mini",
                        "status": "COMPLETED",
                        "is_hosted": False,
                        "metadata": {"num_examples": 5, "rollouts_per_example": 1},
                    },
                ],
                "total": 2,
            }

    monkeypatch.setattr("prime_cli.commands.evals.Config", lambda: DummyConfig())
    monkeypatch.setattr("prime_cli.commands.evals.APIClient", lambda: object())
    monkeypatch.setattr("prime_cli.commands.evals.EvalsClient", DummyClient)

    result = runner.invoke(
        app,
        ["eval", "list"],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )

    assert result.exit_code == 0, result.output
    assert "Type" in result.output
    assert "HOSTED" in result.output
    assert "LOCAL" in result.output












def test_create_hosted_evaluation_adds_team_id_to_payload(monkeypatch):
    captured = {}

    class DummyConfig:
        team_id = "team-123"

    class DummyAPIClient:
        def __init__(self):
            self.config = DummyConfig()

        def post(self, endpoint, json=None):
            captured["endpoint"] = endpoint
            captured["json"] = json
            return {"evaluation_id": "eval-123"}

    monkeypatch.setattr("prime_cli.commands.evals.APIClient", DummyAPIClient)

    result = _create_hosted_evaluations(
        HostedEvalConfig(
            environment_id="env-123",
            inference_model="openai/gpt-4.1-mini",
            num_examples=5,
            rollouts_per_example=3,
        )
    )

    assert result["evaluation_id"] == "eval-123"
    assert captured["endpoint"] == "/hosted-evaluations"
    assert captured["json"]["team_id"] == "team-123"
    assert captured["json"]["eval_config"]["allow_sandbox_access"] is True
    assert captured["json"]["eval_config"]["allow_instances_access"] is False
    assert captured["json"]["eval_config"]["allow_tunnel_access"] is True


def test_create_hosted_evaluation_includes_sampling_args_in_payload(monkeypatch):
    captured = {}

    class DummyConfig:
        team_id = None

    class DummyAPIClient:
        def __init__(self):
            self.config = DummyConfig()

        def post(self, endpoint, json=None):
            captured["endpoint"] = endpoint
            captured["json"] = json
            return {"evaluation_id": "eval-123"}

    monkeypatch.setattr("prime_cli.commands.evals.APIClient", DummyAPIClient)

    _create_hosted_evaluations(
        HostedEvalConfig(
            environment_id="env-123",
            inference_model="openai/gpt-4.1-mini",
            num_examples=5,
            rollouts_per_example=3,
            sampling_args={
                "temperature": 0.2,
                "extra_body": {
                    "provider": {
                        "order": ["azure"],
                        "allow_fallbacks": False,
                        "require_parameters": True,
                    }
                },
            },
        )
    )

    assert captured["endpoint"] == "/hosted-evaluations"
    assert captured["json"]["eval_config"]["sampling_args"] == {
        "temperature": 0.2,
        "extra_body": {
            "provider": {
                "order": ["azure"],
                "allow_fallbacks": False,
                "require_parameters": True,
            }
        },
    }


def test_create_hosted_evaluation_includes_extra_env_kwargs_in_payload(monkeypatch):
    captured = {}

    class DummyConfig:
        team_id = None

    class DummyAPIClient:
        def __init__(self):
            self.config = DummyConfig()

        def post(self, endpoint, json=None):
            captured["endpoint"] = endpoint
            captured["json"] = json
            return {"evaluation_id": "eval-123"}

    monkeypatch.setattr("prime_cli.commands.evals.APIClient", DummyAPIClient)

    _create_hosted_evaluations(
        HostedEvalConfig(
            environment_id="env-123",
            inference_model="openai/gpt-4.1-mini",
            num_examples=5,
            rollouts_per_example=3,
            extra_env_kwargs={
                "task_library": "ronig",
                "keep_remote_artifacts": True,
            },
        )
    )

    assert captured["endpoint"] == "/hosted-evaluations"
    assert captured["json"]["eval_config"]["extra_env_kwargs"] == {
        "task_library": "ronig",
        "keep_remote_artifacts": True,
    }


def test_create_hosted_evaluation_includes_hosted_runtime_args_in_payload(monkeypatch):
    captured = {}

    class DummyConfig:
        team_id = None

    class DummyAPIClient:
        def __init__(self):
            self.config = DummyConfig()

        def post(self, endpoint, json=None):
            captured["endpoint"] = endpoint
            captured["json"] = json
            return {"evaluation_id": "eval-123"}

    monkeypatch.setattr("prime_cli.commands.evals.APIClient", DummyAPIClient)

    _create_hosted_evaluations(
        HostedEvalConfig(
            environment_id="env-123",
            inference_model="openai/gpt-4.1-mini",
            num_examples=5,
            rollouts_per_example=3,
            max_concurrent=64,
            max_retries=3,
            state_columns=["turn", "timing"],
            independent_scoring=True,
            verbose=True,
            headers=["X-Test: one", "X-Second: two"],
        )
    )

    assert captured["endpoint"] == "/hosted-evaluations"
    assert captured["json"]["eval_config"]["max_concurrent"] == 64
    assert captured["json"]["eval_config"]["max_retries"] == 3
    assert captured["json"]["eval_config"]["state_columns"] == ["turn", "timing"]
    assert captured["json"]["eval_config"]["independent_scoring"] is True
    assert captured["json"]["eval_config"]["verbose"] is True
    assert captured["json"]["eval_config"]["headers"] == [
        "X-Test: one",
        "X-Second: two",
    ]


def test_create_hosted_evaluation_includes_api_base_url_and_key_var_in_payload(monkeypatch):
    captured = {}

    class DummyConfig:
        team_id = None

    class DummyAPIClient:
        def __init__(self):
            self.config = DummyConfig()

        def post(self, endpoint, json=None):
            captured["endpoint"] = endpoint
            captured["json"] = json
            return {"evaluation_id": "eval-123"}

    monkeypatch.setattr("prime_cli.commands.evals.APIClient", DummyAPIClient)

    _create_hosted_evaluations(
        HostedEvalConfig(
            environment_id="env-123",
            inference_model="openai/gpt-4.1-mini",
            num_examples=5,
            rollouts_per_example=3,
            api_base_url="https://api.openai.com/v1",
            api_key_var="OPENAI_API_KEY",
        )
    )

    assert captured["endpoint"] == "/hosted-evaluations"
    assert captured["json"]["eval_config"]["api_base_url"] == "https://api.openai.com/v1"
    assert captured["json"]["eval_config"]["api_key_var"] == "OPENAI_API_KEY"


def test_create_hosted_evaluation_includes_tunnel_access_in_payload(monkeypatch):
    captured = {}

    class DummyConfig:
        team_id = None

    class DummyAPIClient:
        def __init__(self):
            self.config = DummyConfig()

        def post(self, endpoint, json=None):
            captured["endpoint"] = endpoint
            captured["json"] = json
            return {"evaluation_id": "eval-123"}

    monkeypatch.setattr("prime_cli.commands.evals.APIClient", DummyAPIClient)

    _create_hosted_evaluations(
        HostedEvalConfig(
            environment_id="env-123",
            inference_model="openai/gpt-4.1-mini",
            num_examples=5,
            rollouts_per_example=3,
            allow_tunnel_access=True,
        )
    )

    assert captured["endpoint"] == "/hosted-evaluations"
    assert captured["json"]["eval_config"]["allow_tunnel_access"] is True


def test_create_hosted_evaluation_accepts_plural_ids_response(monkeypatch):
    class DummyConfig:
        team_id = None

    class DummyAPIClient:
        def __init__(self):
            self.config = DummyConfig()

        def post(self, endpoint, json=None):
            return {"evaluation_ids": ["eval-123", "eval-456"]}

    monkeypatch.setattr("prime_cli.commands.evals.APIClient", DummyAPIClient)

    result = _create_hosted_evaluations(
        HostedEvalConfig(
            environment_id="env-123",
            inference_model="openai/gpt-4.1-mini",
            num_examples=5,
            rollouts_per_example=3,
        ),
        environment_ids=["env-123", "env-456"],
    )

    assert result["evaluation_ids"] == ["eval-123", "eval-456"]
























def test_hosted_eval_config_accepts_id_alias(tmp_path):
    config_path = tmp_path / "eval.toml"
    config_path.write_text(
        """
model = "openai/gpt-4.1-mini"

[[eval]]
id = "gsm8k"
""".strip()
    )

    loaded = _load_hosted_eval_configs(str(config_path))[0]

    assert loaded["env_id"] == "gsm8k"


def test_eval_run_hosted_endpoint_id_uses_default_endpoints_path_from_cwd(monkeypatch, tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "eval.toml"
    config_path.write_text(
        """
endpoint_id = "test-endpoint"

[[eval]]
env_id = "gsm8k"
""".strip()
    )

    captured = {}

    def fake_resolve_endpoints_file(path):
        captured["endpoints_path"] = path
        return Path(path)

    def fake_load_endpoints(path):
        return {
            "test-endpoint": [
                {
                    "model": "openai/gpt-4.1-mini",
                    "url": "https://api.example.com/v1",
                    "key": "PRIME_API_KEY",
                }
            ]
        }

    monkeypatch.setattr(
        "verifiers.utils.eval_utils.resolve_endpoints_file",
        fake_resolve_endpoints_file,
    )
    monkeypatch.setattr("verifiers.utils.eval_utils.load_endpoints", fake_load_endpoints)

    loaded = _load_hosted_eval_configs(str(config_path))[0]

    assert loaded["model"] == "openai/gpt-4.1-mini"
    assert captured["endpoints_path"] == "./configs/endpoints.toml"


def test_hosted_eval_config_allows_eval_endpoint_id_to_override_top_level_model(
    monkeypatch, tmp_path
):
    config_path = tmp_path / "eval.toml"
    config_path.write_text(
        """
model = "openai/gpt-4.1-mini"

[[eval]]
env_id = "gsm8k"
endpoint_id = "test-endpoint"
""".strip()
    )

    monkeypatch.setattr(
        "verifiers.utils.eval_utils.resolve_endpoints_file",
        lambda path: Path(path),
    )
    monkeypatch.setattr(
        "verifiers.utils.eval_utils.load_endpoints",
        lambda path: {
            "test-endpoint": [
                {
                    "model": "anthropic/claude-sonnet-4",
                    "url": "https://api.example.com/v1",
                    "key": "PRIME_API_KEY",
                }
            ]
        },
    )

    loaded = _load_hosted_eval_configs(str(config_path))[0]

    assert loaded["model"] == "anthropic/claude-sonnet-4"


def test_hosted_eval_config_allows_eval_model_to_override_top_level_endpoint_id(
    monkeypatch, tmp_path
):
    config_path = tmp_path / "eval.toml"
    config_path.write_text(
        """
endpoint_id = "test-endpoint"

[[eval]]
env_id = "gsm8k"
model = "anthropic/claude-sonnet-4"
""".strip()
    )

    monkeypatch.setattr(
        "verifiers.utils.eval_utils.resolve_endpoints_file",
        lambda path: (_ for _ in ()).throw(AssertionError("should not resolve endpoint_id")),
    )
    monkeypatch.setattr(
        "verifiers.utils.eval_utils.load_endpoints",
        lambda path: (_ for _ in ()).throw(AssertionError("should not load endpoints")),
    )

    loaded = _load_hosted_eval_configs(str(config_path))[0]

    assert loaded["model"] == "anthropic/claude-sonnet-4"


def test_eval_run_local_toml_passthrough(monkeypatch, tmp_path):
    captured = {}
    config_path = tmp_path / "eval.toml"
    config_path.write_text(
        """
model = "openai/gpt-4.1-mini"

[[eval]]
env_id = "gsm8k"
""".strip()
    )

    def fake_run_eval_passthrough(environment, passthrough_args, skip_upload, env_path):
        captured["environment"] = environment
        captured["passthrough_args"] = passthrough_args
        captured["skip_upload"] = skip_upload
        captured["env_path"] = env_path

    monkeypatch.setattr("prime_cli.commands.evals.run_eval_passthrough", fake_run_eval_passthrough)

    result = runner.invoke(
        app,
        ["eval", "run", str(config_path), "--skip-upload"],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "environment": str(config_path),
        "passthrough_args": [],
        "skip_upload": True,
        "env_path": None,
    }


def test_eval_run_local_sampling_args_passthrough(monkeypatch):
    captured = {}

    def fake_run_eval_passthrough(environment, passthrough_args, skip_upload, env_path):
        captured["environment"] = environment
        captured["passthrough_args"] = passthrough_args
        captured["skip_upload"] = skip_upload
        captured["env_path"] = env_path

    monkeypatch.setattr("prime_cli.commands.evals.run_eval_passthrough", fake_run_eval_passthrough)

    result = runner.invoke(
        app,
        ["eval", "run", "gsm8k", "--sampling-args", '{"temperature":0.2}'],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "environment": "gsm8k",
        "passthrough_args": ["--sampling-args", '{"temperature":0.2}'],
        "skip_upload": False,
        "env_path": None,
    }






def test_eval_run_rejects_hosted_only_flags_without_hosted():
    result = runner.invoke(
        app,
        ["eval", "run", "gsm8k", "--follow"],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )
    assert result.exit_code == 1
    assert "hosted-only options require `--hosted`" in result.output


def test_eval_run_rejects_explicit_poll_interval_without_hosted():
    result = runner.invoke(
        app,
        ["eval", "run", "gsm8k", "--poll-interval", "10"],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )
    assert result.exit_code == 1
    assert "hosted-only options require `--hosted`" in result.output


def test_eval_run_rejects_tunnel_access_without_hosted():
    result = runner.invoke(
        app,
        ["eval", "run", "gsm8k", "--allow-tunnel-access"],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )
    assert result.exit_code == 1
    assert "hosted-only options require `--hosted`" in result.output






def test_resolve_hosted_environment_warns_when_local_code_differs(monkeypatch, capsys):
    monkeypatch.setattr(
        "prime_cli.commands.evals._resolve_environment_reference",
        lambda environment, env_dir_path: ResolvedEnvironment(
            original=environment,
            env_name="gsm8k",
            install_mode="local",
            env_display_id="gsm8k (local - ahead of primeintellect/gsm8k)",
            platform_slug="primeintellect/gsm8k",
            recommend_push=True,
            push_reason="ahead",
        ),
    )

    class DummyAPIClient:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, endpoint):
            assert endpoint == "/environmentshub/primeintellect/gsm8k/@latest"
            return {"data": {"id": "env-123"}}

    monkeypatch.setattr("prime_cli.commands.evals.APIClient", DummyAPIClient)

    assert _resolve_hosted_environment("gsm8k", env_dir_path=None, env_path=None) == (
        "primeintellect/gsm8k",
        "env-123",
    )

    output = capsys.readouterr().out
    assert "Local environment code differs from the latest published version of" in output
    assert "Hosted evaluations always use the latest published version of" in output
    assert "primeintellect/gsm8k" in output
    assert "Using hosted environment primeintellect/gsm8k@latest" in output


def test_resolve_hosted_environment_requires_a_published_environment(monkeypatch, capsys):
    monkeypatch.setattr(
        "prime_cli.commands.evals._resolve_environment_reference",
        lambda environment, env_dir_path: ResolvedEnvironment(
            original=environment,
            env_name="gsm8k",
            install_mode="local",
            env_display_id="gsm8k (local only)",
            recommend_push=True,
            push_reason="local_only",
        ),
    )

    with pytest.raises(typer.Exit) as exc_info:
        _resolve_hosted_environment("gsm8k", env_dir_path=None, env_path=None)

    assert exc_info.value.exit_code == 1
    output = capsys.readouterr().out
    assert "hosted evaluations require an upstream environment on the platform" in output
    assert "publish the local environment with `prime env push`" in output


def test_resolve_hosted_environment_requires_a_pushed_hub_version(monkeypatch, capsys):
    monkeypatch.setattr(
        "prime_cli.commands.evals._resolve_environment_reference",
        lambda environment, env_dir_path: ResolvedEnvironment(
            original=environment,
            env_name="gsm8k",
            install_mode="local",
            env_display_id="gsm8k (local - ahead of primeintellect/gsm8k)",
            platform_slug="primeintellect/gsm8k",
            recommend_push=True,
            push_reason="ahead",
        ),
    )

    class DummyAPIClient:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, endpoint):
            raise APIError("404 not found")

    monkeypatch.setattr("prime_cli.commands.evals.APIClient", DummyAPIClient)

    with pytest.raises(typer.Exit) as exc_info:
        _resolve_hosted_environment("gsm8k", env_dir_path=None, env_path=None)

    assert exc_info.value.exit_code == 1
    output = capsys.readouterr().out
    assert "hosted evaluations require an environment that is published to the" in output
    assert "platform" in output
    assert "Publish primeintellect/gsm8k with `prime env push` first." in output




def test_eval_stop_command_calls_cancel_endpoint(monkeypatch):
    captured = {}

    def fake_patch(self, endpoint, json=None, params=None):
        captured["endpoint"] = endpoint
        return {"message": "Evaluation cancelled", "evaluation_id": "eval-123"}

    monkeypatch.setattr("prime_cli.commands.evals.APIClient.patch", fake_patch)

    result = runner.invoke(
        app,
        ["eval", "stop", "eval-123"],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )

    assert result.exit_code == 0, result.output
    assert captured == {"endpoint": "/hosted-evaluations/eval-123/cancel"}
    assert "Evaluation cancelled" in result.output
    assert "dashboard/evaluations/eval-123" in result.output


def test_print_eval_status_prefers_returned_viewer_url(monkeypatch, capsys):
    called = {"used": False}

    monkeypatch.setattr(
        "prime_cli.commands.evals.get_eval_viewer_url",
        lambda eval_id: called.__setitem__("used", True) or f"fallback/{eval_id}",
    )

    _print_eval_status(
        {
            "status": "RUNNING",
            "evaluation_id": "eval-123",
            "viewer_url": "http://localhost:3000/dashboard/evaluations/eval-123",
            "error_message": None,
        }
    )

    output = capsys.readouterr().out
    assert "http://localhost:3000/dashboard/evaluations/eval-123" in output
    assert called["used"] is False


def test_print_eval_status_falls_back_to_eval_id_when_viewer_url_missing(monkeypatch, capsys):
    monkeypatch.setattr(
        "prime_cli.commands.evals.get_eval_viewer_url",
        lambda eval_id: f"fallback/{eval_id}",
    )

    _print_eval_status(
        {
            "status": "CANCELLED",
            "evaluation_id": "eval-123",
            "viewer_url": None,
            "error_message": "Stopped",
        }
    )

    output = capsys.readouterr().out
    assert "Status: CANCELLED" in output
    assert "Error:" in output
    assert "fallback/eval-123" in output


def test_eval_logs_command_follows_incremental_output(monkeypatch):
    statuses = iter(
        [
            {"status": "RUNNING", "evaluation_id": "eval-123"},
            {"status": "COMPLETED", "evaluation_id": "eval-123", "total_samples": 2},
        ]
    )
    logs = iter(
        [
            "Line 1\nLine 2",
            "Line 1\nLine 2\nLine 3",
        ]
    )

    monkeypatch.setattr("prime_cli.commands.evals.time.sleep", lambda _: None)
    monkeypatch.setattr(
        "prime_cli.commands.evals._fetch_eval_status",
        lambda _client, _: next(statuses),
    )
    monkeypatch.setattr("prime_cli.commands.evals._fetch_logs", lambda _client, _: next(logs))
    monkeypatch.setattr(
        "prime_cli.commands.evals.get_eval_viewer_url",
        lambda eval_id: f"https://app.primeintellect.ai/dashboard/evaluations/{eval_id}",
    )

    result = runner.invoke(
        app,
        ["eval", "logs", "eval-123", "--follow"],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )

    assert result.exit_code == 0, result.output
    assert "Line 1" in result.output
    assert "Line 2" in result.output
    assert "Line 3" in result.output
    assert result.output.count("Line 1") == 1
    assert "Status: COMPLETED" in result.output


def test_eval_logs_command_emits_waiting_status_when_logs_are_unchanged(monkeypatch):
    statuses = iter(
        [
            {"status": "RUNNING", "evaluation_id": "eval-123"},
            {"status": "RUNNING", "evaluation_id": "eval-123"},
            {"status": "COMPLETED", "evaluation_id": "eval-123", "total_samples": 1},
        ]
    )
    logs = iter(
        [
            "Line 1",
            "Line 1",
            "Line 1",
        ]
    )

    monkeypatch.setattr("prime_cli.commands.evals.time.sleep", lambda _: None)
    monkeypatch.setattr(
        "prime_cli.commands.evals.HOSTED_LOGS_STATUS_UPDATE_EVERY_POLLS",
        1,
    )
    monkeypatch.setattr(
        "prime_cli.commands.evals._fetch_eval_status",
        lambda _client, _: next(statuses),
    )
    monkeypatch.setattr("prime_cli.commands.evals._fetch_logs", lambda _client, _: next(logs))
    monkeypatch.setattr(
        "prime_cli.commands.evals.get_eval_viewer_url",
        lambda eval_id: f"https://app.primeintellect.ai/dashboard/evaluations/{eval_id}",
    )

    result = runner.invoke(
        app,
        ["eval", "logs", "eval-123", "--follow"],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )

    assert result.exit_code == 0, result.output
    assert result.output.count("Line 1") == 1
    assert "Evaluation status: RUNNING (waiting for logs...)" in result.output
    assert "Status: COMPLETED" in result.output


def test_eval_run_hosted_raises_not_implemented():
    # The platform backend isn't v1-aware yet, so `--hosted` is gated: the run config is parsed
    # (machinery preserved) and then a NotImplementedError is raised before any submission.
    result = runner.invoke(
        app,
        ["eval", "run", "gsm8k", "--hosted", "-m", "openai/gpt-4.1-mini", "-n", "2"],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, NotImplementedError)
    assert "not supported yet" in str(result.exception)
    assert "openai/gpt-4.1-mini" in str(result.exception)
