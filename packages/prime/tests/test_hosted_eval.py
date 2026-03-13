from prime_cli.client import APIError
from prime_cli.main import app
from prime_cli.utils.hosted_eval import clean_logs, filter_progress_bars, strip_ansi
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


def test_eval_run_hosted_invokes_hosted_runner(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        "prime_cli.commands.evals._resolve_hosted_environment",
        lambda environment, env_dir_path=None, env_path=None: (
            "primeintellect/gsm8k",
            "env-123",
        ),
    )

    def fake_run_hosted_evaluation(config):
        captured["environment_id"] = config.environment_id
        captured["inference_model"] = config.inference_model
        captured["num_examples"] = config.num_examples
        captured["rollouts_per_example"] = config.rollouts_per_example

        from prime_cli.utils.hosted_eval import EvalStatus, HostedEvalResult

        return HostedEvalResult(
            evaluation_id="eval-123",
            status=EvalStatus.PENDING,
            total_samples=14,
            avg_score=0.75,
            min_score=0.5,
            max_score=1.0,
            error_message=None,
            logs="done",
        )

    monkeypatch.setattr(
        "prime_cli.commands.evals._create_hosted_evaluation",
        fake_run_hosted_evaluation,
    )

    result = runner.invoke(
        app,
        ["eval", "run", "primeintellect/gsm8k", "--hosted", "-m", "openai/gpt-4.1-mini"],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )

    assert result.exit_code == 0, result.output
    assert captured["environment_id"] == "env-123"
    assert captured["inference_model"] == "openai/gpt-4.1-mini"
    assert captured["num_examples"] == 5
    assert captured["rollouts_per_example"] == 3
    assert "Hosted evaluation started" in result.output
    assert "prime eval logs eval-123 -f" in result.output


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

    from prime_cli.commands.evals import _create_hosted_evaluation
    from prime_cli.utils.hosted_eval import HostedEvalConfig

    result = _create_hosted_evaluation(
        HostedEvalConfig(
            environment_id="env-123",
            inference_model="openai/gpt-4.1-mini",
            num_examples=5,
            rollouts_per_example=3,
        )
    )

    assert result.evaluation_id == "eval-123"
    assert captured["endpoint"] == "/hosted-evaluations"
    assert captured["json"]["team_id"] == "team-123"


def test_eval_run_hosted_follow_streams_logs(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        "prime_cli.commands.evals._resolve_hosted_environment",
        lambda environment, env_dir_path=None, env_path=None: (
            "primeintellect/gsm8k",
            "env-123",
        ),
    )

    def fake_run_hosted_evaluation(config):
        captured["num_examples"] = config.num_examples
        captured["rollouts_per_example"] = config.rollouts_per_example

        from prime_cli.utils.hosted_eval import EvalStatus, HostedEvalResult

        return HostedEvalResult(
            evaluation_id="eval-123",
            status=EvalStatus.PENDING,
            total_samples=0,
            avg_score=None,
            min_score=None,
            max_score=None,
        )

    monkeypatch.setattr(
        "prime_cli.commands.evals._create_hosted_evaluation",
        fake_run_hosted_evaluation,
    )

    def fake_display_logs(eval_id, tail, follow, poll_interval=5.0):
        captured["display_eval_id"] = eval_id
        captured["display_tail"] = tail
        captured["display_follow"] = follow
        captured["display_poll_interval"] = poll_interval

    monkeypatch.setattr("prime_cli.commands.evals._display_logs", fake_display_logs)

    result = runner.invoke(
        app,
        [
            "eval",
            "run",
            "primeintellect/gsm8k",
            "--hosted",
            "--follow",
            "-n",
            "7",
            "-r",
            "2",
        ],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )

    assert result.exit_code == 0, result.output
    assert captured["num_examples"] == 7
    assert captured["rollouts_per_example"] == 2
    assert captured["display_eval_id"] == "eval-123"
    assert captured["display_tail"] == 1000
    assert captured["display_follow"] is True
    assert captured["display_poll_interval"] == 10.0


def test_eval_run_hosted_supports_single_eval_toml(monkeypatch, tmp_path):
    captured = {}
    config_path = tmp_path / "eval.toml"
    config_path.write_text(
        """
model = "openai/gpt-4.1-mini"
num_examples = 7
rollouts_per_example = 2

[[eval]]
env_id = "gsm8k"
env_args = { split = "test" }
""".strip()
    )

    def fake_resolve(environment, env_dir_path=None, env_path=None):
        captured["environment"] = environment
        captured["env_dir_path"] = env_dir_path
        captured["env_path"] = env_path
        return ("primeintellect/gsm8k", "env-123")

    def fake_run_hosted_evaluation(config):
        captured["environment_id"] = config.environment_id
        captured["inference_model"] = config.inference_model
        captured["num_examples"] = config.num_examples
        captured["rollouts_per_example"] = config.rollouts_per_example
        captured["env_args"] = config.env_args

        from prime_cli.utils.hosted_eval import EvalStatus, HostedEvalResult

        return HostedEvalResult(
            evaluation_id="eval-123",
            status=EvalStatus.PENDING,
            total_samples=0,
            avg_score=None,
            min_score=None,
            max_score=None,
        )

    monkeypatch.setattr("prime_cli.commands.evals._resolve_hosted_environment", fake_resolve)
    monkeypatch.setattr(
        "prime_cli.commands.evals._create_hosted_evaluation",
        fake_run_hosted_evaluation,
    )

    result = runner.invoke(
        app,
        ["eval", "run", str(config_path), "--hosted"],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "environment": "gsm8k",
        "env_dir_path": None,
        "env_path": None,
        "environment_id": "env-123",
        "inference_model": "openai/gpt-4.1-mini",
        "num_examples": 7,
        "rollouts_per_example": 2,
        "env_args": {"split": "test"},
    }


def test_eval_run_hosted_rejects_multi_eval_toml(tmp_path):
    config_path = tmp_path / "eval.toml"
    config_path.write_text(
        """
[[eval]]
env_id = "gsm8k"

[[eval]]
env_id = "math500"
""".strip()
    )

    result = runner.invoke(
        app,
        ["eval", "run", str(config_path), "--hosted"],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )
    assert result.exit_code == 1
    assert "exactly one" in result.output
    assert "eval" in result.output


def test_eval_run_hosted_rejects_unsupported_toml_fields(tmp_path):
    config_path = tmp_path / "eval.toml"
    config_path.write_text(
        """
resume = true

[[eval]]
env_id = "gsm8k"
""".strip()
    )

    result = runner.invoke(
        app,
        ["eval", "run", str(config_path), "--hosted"],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )
    assert result.exit_code == 1
    assert "does not support" in result.output
    assert "`resume`" in result.output


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


def test_eval_run_hosted_passes_env_dir_path_to_resolver(monkeypatch):
    captured = {}

    def fake_resolve(environment, env_dir_path=None, env_path=None):
        captured["environment"] = environment
        captured["env_dir_path"] = env_dir_path
        captured["env_path"] = env_path
        return ("primeintellect/gsm8k", "env-123")

    def fake_run_hosted_evaluation(config):
        from prime_cli.utils.hosted_eval import EvalStatus, HostedEvalResult

        return HostedEvalResult(
            evaluation_id="eval-123",
            status=EvalStatus.PENDING,
            total_samples=0,
            avg_score=None,
            min_score=None,
            max_score=None,
        )

    monkeypatch.setattr("prime_cli.commands.evals._resolve_hosted_environment", fake_resolve)
    monkeypatch.setattr(
        "prime_cli.commands.evals._create_hosted_evaluation",
        fake_run_hosted_evaluation,
    )

    result = runner.invoke(
        app,
        [
            "eval",
            "run",
            "gsm8k",
            "--hosted",
            "--env-dir-path",
            "/tmp/custom-envs",
        ],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "environment": "gsm8k",
        "env_dir_path": "/tmp/custom-envs",
        "env_path": None,
    }


def test_eval_run_hosted_passes_env_path_to_resolver(monkeypatch):
    captured = {}

    def fake_resolve(environment, env_dir_path=None, env_path=None):
        captured["environment"] = environment
        captured["env_dir_path"] = env_dir_path
        captured["env_path"] = env_path
        return ("primeintellect/gsm8k", "env-123")

    def fake_run_hosted_evaluation(config):
        from prime_cli.utils.hosted_eval import EvalStatus, HostedEvalResult

        return HostedEvalResult(
            evaluation_id="eval-123",
            status=EvalStatus.PENDING,
            total_samples=0,
            avg_score=None,
            min_score=None,
            max_score=None,
        )

    monkeypatch.setattr("prime_cli.commands.evals._resolve_hosted_environment", fake_resolve)
    monkeypatch.setattr(
        "prime_cli.commands.evals._create_hosted_evaluation",
        fake_run_hosted_evaluation,
    )

    result = runner.invoke(
        app,
        [
            "eval",
            "run",
            "gsm8k",
            "--hosted",
            "--env-path",
            "/tmp/local-env",
        ],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "environment": "gsm8k",
        "env_dir_path": None,
        "env_path": "/tmp/local-env",
    }


def test_eval_run_hosted_reports_resolve_api_errors(monkeypatch):
    monkeypatch.setattr(
        "prime_cli.commands.evals._resolve_hosted_environment",
        lambda environment, env_dir_path=None, env_path=None: (_ for _ in ()).throw(
            APIError("lookup failed")
        ),
    )

    result = runner.invoke(
        app,
        ["eval", "run", "gsm8k", "--hosted"],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )

    assert result.exit_code == 1
    assert "lookup failed" in result.output


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
    from prime_cli.commands.evals import _print_eval_status

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
    from prime_cli.commands.evals import _print_eval_status

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
