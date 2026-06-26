import pytest
from click.testing import CliRunner
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

runner = CliRunner()
CLI_ENV = {"PRIME_DISABLE_VERSION_CHECK": "1"}


class TestLogCleaning:
    def test_strip_ansi_basic(self):
        assert strip_ansi("\x1b[31mRed text\x1b[0m") == "Red text"

    def test_filter_progress_bars_keeps_100_percent(self):
        text = "Progress: 100%|██████████| 10/10 [00:01<00:00]"
        assert "100%" in filter_progress_bars(text)

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


@pytest.fixture
def submit_capture(monkeypatch):
    captured = []

    def resolve(environment, *, env_path):
        return environment.split("@", 1)[0], f"env-{environment.rsplit('/', 1)[-1]}"

    def create(config, environment_ids):
        captured.append((config, environment_ids))
        return {"evaluation_id": f"eval-{len(captured)}"}

    monkeypatch.setattr("prime_cli.commands.evals._resolve_hosted_environment", resolve)
    monkeypatch.setattr("prime_cli.commands.evals._create_hosted_evaluations", create)
    return captured


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

    result = runner.invoke(app, ["eval", "list"], env=CLI_ENV)

    assert result.exit_code == 0, result.output
    assert "HOSTED" in result.output
    assert "LOCAL" in result.output


def test_eval_run_delegates_untouched_arguments(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "prime_cli.commands.evals.exec_verifiers_process",
        lambda name, args, plain=False: captured.update(name=name, args=args, plain=plain),
    )

    result = runner.invoke(
        app,
        ["eval", "run", "--taskset.id", "owner/tasks", "--harness.runtime.type", "prime"],
        env=CLI_ENV,
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "name": "eval",
        "args": ["--taskset.id", "owner/tasks", "--harness.runtime.type", "prime"],
        "plain": False,
    }


def test_eval_submit_uses_typed_defaults(submit_capture):
    result = runner.invoke(
        app,
        ["eval", "submit", "primeintellect/gsm8k", "--model", "openai/gpt-4.1-mini"],
        env=CLI_ENV,
    )

    assert result.exit_code == 0, result.output
    config, environment_ids = submit_capture[0]
    assert config.env_id == "primeintellect/gsm8k"
    assert config.model == "openai/gpt-4.1-mini"
    assert config.num_examples == 5
    assert config.rollouts_per_example == 3
    assert config.allow_sandbox_access is True
    assert config.allow_instances_access is False
    assert config.allow_tunnel_access is True
    assert environment_ids == ["env-gsm8k"]
    assert "prime eval logs eval-1 -f" in result.output


def test_eval_submit_parses_runtime_options(submit_capture):
    result = runner.invoke(
        app,
        [
            "eval",
            "submit",
            "primeintellect/gsm8k",
            "--max-concurrent",
            "100",
            "--max-retries",
            "3",
            "--state-columns",
            "turn",
            "timing",
            "--independent-scoring",
            "--verbose",
            "--header",
            "X-Test: one",
            "--sampling-args",
            '{"max_tokens":4096,"temperature":0.2}',
            "--extra-env-kwargs",
            '{"task_library":"ronig"}',
            "--api-base-url",
            "https://api.openai.com/v1",
            "--api-key-var",
            "OPENAI_API_KEY",
        ],
        env=CLI_ENV,
    )

    assert result.exit_code == 0, result.output
    config, _ = submit_capture[0]
    assert config.max_concurrent == 100
    assert config.max_retries == 3
    assert config.state_columns == ["turn", "timing"]
    assert config.independent_scoring is True
    assert config.verbose is True
    assert config.headers == ["X-Test: one"]
    assert config.sampling_args == {"max_tokens": 4096, "temperature": 0.2}
    assert config.extra_env_kwargs == {"task_library": "ronig"}
    assert config.api_base_url == "https://api.openai.com/v1"
    assert config.api_key_var == "OPENAI_API_KEY"


def test_eval_submit_rejects_invalid_header(submit_capture):
    result = runner.invoke(
        app,
        ["eval", "submit", "primeintellect/gsm8k", "--header", "not-a-header"],
        env=CLI_ENV,
    )

    assert result.exit_code == 1
    assert "headers must use 'Name: Value'" in result.output
    assert submit_capture == []


def test_create_hosted_evaluation_builds_payload(monkeypatch):
    captured = {}

    class DummyConfig:
        team_id = "team-123"

    class DummyAPIClient:
        def __init__(self):
            self.config = DummyConfig()

        def post(self, endpoint, json=None):
            captured.update(endpoint=endpoint, json=json)
            return {"evaluation_id": "eval-123"}

    monkeypatch.setattr("prime_cli.commands.evals.APIClient", DummyAPIClient)
    config = HostedEvalConfig(
        env_id="owner/gsm8k",
        model="openai/gpt-4.1-mini",
        sampling_args={"temperature": 0.2},
        extra_env_kwargs={"split": "test"},
        max_concurrent=64,
        headers=["X-Test: one"],
    )

    result = _create_hosted_evaluations(config, ["env-123", "env-456"])

    assert result == {"evaluation_id": "eval-123"}
    assert captured["endpoint"] == "/hosted-evaluations"
    assert captured["json"]["team_id"] == "team-123"
    assert captured["json"]["environment_ids"] == ["env-123", "env-456"]
    assert captured["json"]["inference_model"] == "openai/gpt-4.1-mini"
    assert captured["json"]["eval_config"]["sampling_args"] == {"temperature": 0.2}
    assert captured["json"]["eval_config"]["extra_env_kwargs"] == {"split": "test"}
    assert captured["json"]["eval_config"]["max_concurrent"] == 64
    assert captured["json"]["eval_config"]["headers"] == ["X-Test: one"]


def test_load_hosted_eval_configs_supports_shared_defaults(tmp_path):
    path = tmp_path / "hosted.toml"
    path.write_text(
        'model = "openai/gpt-4.1-mini"\n'
        "num_examples = 8\n"
        "[[eval]]\n"
        'env_id = "owner/gsm8k"\n'
        "[[eval]]\n"
        'env_id = "owner/math500"\n'
        "num_examples = 12\n",
        encoding="utf-8",
    )

    configs = _load_hosted_eval_configs(path)

    assert [config.env_id for config in configs] == ["owner/gsm8k", "owner/math500"]
    assert [config.num_examples for config in configs] == [8, 12]
    assert all(config.model == "openai/gpt-4.1-mini" for config in configs)


def test_load_hosted_eval_configs_rejects_unknown_fields(tmp_path, capsys):
    path = tmp_path / "hosted.toml"
    path.write_text('env_id = "owner/gsm8k"\ndebug = true\n', encoding="utf-8")

    with pytest.raises(SystemExit):
        _load_hosted_eval_configs(path)

    assert "Extra inputs are not permitted" in capsys.readouterr().out


def test_eval_submit_groups_targets_with_identical_settings(submit_capture, tmp_path):
    path = tmp_path / "hosted.toml"
    path.write_text(
        'model = "openai/gpt-4.1-mini"\n'
        "[[eval]]\n"
        'env_id = "owner/gsm8k"\n'
        "[[eval]]\n"
        'env_id = "owner/math500"\n',
        encoding="utf-8",
    )

    result = runner.invoke(app, ["eval", "submit", str(path)], env=CLI_ENV)

    assert result.exit_code == 0, result.output
    assert len(submit_capture) == 1
    _, environment_ids = submit_capture[0]
    assert environment_ids == ["env-gsm8k", "env-math500"]


def test_eval_submit_keeps_different_settings_separate(submit_capture, tmp_path):
    path = tmp_path / "hosted.toml"
    path.write_text(
        "[[eval]]\n"
        'env_id = "owner/gsm8k"\n'
        "num_examples = 5\n"
        "[[eval]]\n"
        'env_id = "owner/math500"\n'
        "num_examples = 10\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["eval", "submit", str(path)], env=CLI_ENV)

    assert result.exit_code == 0, result.output
    assert [call[1] for call in submit_capture] == [["env-gsm8k"], ["env-math500"]]


def test_eval_submit_follow_streams_single_evaluation(submit_capture, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "prime_cli.commands.evals._display_logs",
        lambda eval_id, tail, follow, poll_interval: captured.update(
            eval_id=eval_id, tail=tail, follow=follow, poll_interval=poll_interval
        ),
    )

    result = runner.invoke(
        app,
        ["eval", "submit", "owner/gsm8k", "--follow", "--poll-interval", "0.25"],
        env=CLI_ENV,
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "eval_id": "eval-1",
        "tail": 1000,
        "follow": True,
        "poll_interval": 0.25,
    }


def test_resolve_hosted_environment_uses_explicit_slug(monkeypatch, capsys):
    class DummyAPIClient:
        def get(self, endpoint):
            assert endpoint == "/environmentshub/owner/gsm8k/@latest"
            return {"data": {"id": "env-123"}}

    monkeypatch.setattr("prime_cli.commands.evals.APIClient", DummyAPIClient)

    resolved = _resolve_hosted_environment("owner/gsm8k", env_path=None)

    assert resolved == ("owner/gsm8k", "env-123")
    assert "owner/gsm8k@latest" in capsys.readouterr().out


def test_resolve_hosted_environment_rejects_version(capsys):
    with pytest.raises(SystemExit):
        _resolve_hosted_environment("owner/gsm8k@1.2.3", env_path=None)

    assert "only unversioned slugs" in capsys.readouterr().out


def test_resolve_hosted_environment_uses_local_metadata(monkeypatch, tmp_path):
    env_path = tmp_path / "gsm8k"
    (env_path / ".prime").mkdir(parents=True)
    (env_path / ".prime" / ".env-metadata.json").write_text(
        '{"owner":"owner","name":"gsm8k"}', encoding="utf-8"
    )

    class DummyAPIClient:
        def get(self, endpoint):
            assert endpoint == "/environmentshub/owner/gsm8k/@latest"
            return {"data": {"id": "env-123"}}

    monkeypatch.setattr("prime_cli.commands.evals.APIClient", DummyAPIClient)

    assert _resolve_hosted_environment("gsm8k", env_path=str(env_path)) == (
        "owner/gsm8k",
        "env-123",
    )


def test_resolve_hosted_environment_requires_upstream(capsys):
    with pytest.raises(SystemExit):
        _resolve_hosted_environment("not-published", env_path=None)

    assert "require an upstream environment" in capsys.readouterr().out


def test_eval_submit_reports_resolve_api_errors(monkeypatch):
    monkeypatch.setattr(
        "prime_cli.commands.evals._resolve_hosted_environment",
        lambda environment, env_path=None: (_ for _ in ()).throw(APIError("lookup failed")),
    )

    result = runner.invoke(app, ["eval", "submit", "owner/gsm8k"], env=CLI_ENV)

    assert result.exit_code == 1
    assert "lookup failed" in result.output


def test_eval_stop_command_calls_cancel_endpoint(monkeypatch):
    captured = {}

    def fake_patch(self, endpoint, json=None, params=None):
        captured["endpoint"] = endpoint
        return {"message": "Evaluation cancelled", "evaluation_id": "eval-123"}

    monkeypatch.setattr("prime_cli.commands.evals.APIClient.patch", fake_patch)

    result = runner.invoke(app, ["eval", "stop", "eval-123"], env=CLI_ENV)

    assert result.exit_code == 0, result.output
    assert captured == {"endpoint": "/hosted-evaluations/eval-123/cancel"}
    assert "Evaluation cancelled" in result.output


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
        }
    )

    assert "http://localhost:3000/dashboard/evaluations/eval-123" in capsys.readouterr().out
    assert called["used"] is False


def test_print_eval_status_falls_back_to_eval_id(monkeypatch, capsys):
    monkeypatch.setattr(
        "prime_cli.commands.evals.get_eval_viewer_url",
        lambda eval_id: f"fallback/{eval_id}",
    )

    _print_eval_status(
        {
            "status": "CANCELLED",
            "evaluation_id": "eval-123",
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
    logs = iter(["Line 1\nLine 2", "Line 1\nLine 2\nLine 3"])
    monkeypatch.setattr("prime_cli.commands.evals.time.sleep", lambda _: None)
    monkeypatch.setattr(
        "prime_cli.commands.evals._fetch_eval_status", lambda _client, _: next(statuses)
    )
    monkeypatch.setattr("prime_cli.commands.evals._fetch_logs", lambda _client, _: next(logs))

    result = runner.invoke(app, ["eval", "logs", "eval-123", "--follow"], env=CLI_ENV)

    assert result.exit_code == 0, result.output
    assert result.output.count("Line 1") == 1
    assert "Line 3" in result.output
    assert "Status: COMPLETED" in result.output


def test_eval_logs_command_emits_waiting_status(monkeypatch):
    statuses = iter(
        [
            {"status": "RUNNING", "evaluation_id": "eval-123"},
            {"status": "RUNNING", "evaluation_id": "eval-123"},
            {"status": "COMPLETED", "evaluation_id": "eval-123", "total_samples": 1},
        ]
    )
    logs = iter(["Line 1", "Line 1", "Line 1"])
    monkeypatch.setattr("prime_cli.commands.evals.time.sleep", lambda _: None)
    monkeypatch.setattr("prime_cli.commands.evals.HOSTED_LOGS_STATUS_UPDATE_EVERY_POLLS", 1)
    monkeypatch.setattr(
        "prime_cli.commands.evals._fetch_eval_status", lambda _client, _: next(statuses)
    )
    monkeypatch.setattr("prime_cli.commands.evals._fetch_logs", lambda _client, _: next(logs))

    result = runner.invoke(app, ["eval", "logs", "eval-123", "--follow"], env=CLI_ENV)

    assert result.exit_code == 0, result.output
    assert result.output.count("Line 1") == 1
    assert "Evaluation status: RUNNING (waiting for logs...)" in result.output
    assert "Status: COMPLETED" in result.output
