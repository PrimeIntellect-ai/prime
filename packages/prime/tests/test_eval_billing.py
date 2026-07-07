import json
from pathlib import Path

import httpx
import pytest
import toml
import typer
from prime_cli.api.inference import (
    InferenceAPIError,
    InferenceClient,
    InferencePaymentRequiredError,
)
from prime_cli.verifiers_bridge import (
    ResolvedEnvironment,
    run_eval_passthrough,
    run_gepa_passthrough,
)
from typing_extensions import cast


class DummyConfig:
    api_key = "test-api-key"
    inference_url = "https://api.pinference.ai/api/v1"
    team_id = None


class DummyPlugin:
    eval_module = "verifiers.v1.cli.eval.main"
    gepa_module = "verifiers.scripts.gepa"

    def build_module_command(self, module: str, args: list[str]) -> list[str]:
        return [module, *args]


def test_inference_client_maps_402_to_billing_error(monkeypatch):
    client = InferenceClient.__new__(InferenceClient)
    client.inference_url = "https://api.pinference.ai/api/v1"

    request = httpx.Request("GET", f"{client.inference_url}/models")
    response = httpx.Response(
        402,
        request=request,
        json={
            "error": {
                "message": (
                    "Insufficient balance (including overdraft). Please add funds to continue."
                ),
                "type": "insufficient_quota",
                "code": "insufficient_funds",
                "param": None,
            },
            "request_id": "req-123",
            "inference_id": "inf-123",
        },
    )

    class DummyHTTPClient:
        def get(self, url):
            return response

    client._client = DummyHTTPClient()

    with pytest.raises(
        InferencePaymentRequiredError,
        match="Payment required\\. Insufficient balance",
    ):
        client.list_models()


@pytest.mark.parametrize(
    "error_message",
    [
        (
            "Payment required. Please check your billing status at "
            "https://app.primeintellect.ai/dashboard/billing"
        ),
        "Insufficient balance (including overdraft). Please add funds to continue.",
    ],
)
def test_eval_run_blocks_when_inference_billing_is_missing(monkeypatch, error_message):
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.load_verifiers_prime_plugin", lambda: DummyPlugin()
    )
    monkeypatch.setattr("prime_cli.verifiers_bridge.Config", lambda: DummyConfig())

    class DummyInferenceClient:
        def __init__(self, timeout=None):
            self.timeout = timeout

        def retrieve_model(self, model):
            return {"id": model}

        def chat_completion(self, payload, stream=False):
            raise InferencePaymentRequiredError(error_message)

    monkeypatch.setattr("prime_cli.verifiers_bridge.InferenceClient", DummyInferenceClient)

    with pytest.raises(typer.Exit) as exc_info:
        run_eval_passthrough(
            environment="single_turn_math",
            passthrough_args=["-m", "deepseek/deepseek-chat"],
            skip_upload=False,
            env_path=None,
        )

    assert cast(typer.Exit, exc_info.value).exit_code == 1


def test_eval_preflight_omits_max_tokens(monkeypatch):
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.load_verifiers_prime_plugin", lambda: DummyPlugin()
    )
    monkeypatch.setattr("prime_cli.verifiers_bridge.Config", lambda: DummyConfig())
    seen_payloads = []
    seen_timeouts = []

    class DummyInferenceClient:
        def __init__(self, timeout=None):
            seen_timeouts.append(timeout)

        def retrieve_model(self, model):
            return {"id": model}

        def chat_completion(self, payload, stream=False):
            seen_payloads.append(payload)
            return {"id": "cmpl-123", "choices": []}

    monkeypatch.setattr("prime_cli.verifiers_bridge.InferenceClient", DummyInferenceClient)
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge._prepare_v1_environment",
        lambda *args, **kwargs: (_ for _ in ()).throw(typer.Exit(0)),
    )

    with pytest.raises(typer.Exit) as exc_info:
        run_eval_passthrough(
            environment="single_turn_math",
            passthrough_args=["-m", "openai/gpt-4.1-mini"],
            skip_upload=False,
            env_path=None,
        )

    assert cast(typer.Exit, exc_info.value).exit_code == 0
    assert seen_payloads == [
        {
            "model": "openai/gpt-4.1-mini",
            "messages": [{"role": "user", "content": "Reply with OK."}],
        }
    ]
    assert len(seen_timeouts) == 2
    assert all(timeout.read == 300.0 for timeout in seen_timeouts)


def test_eval_run_uses_v1_client_config(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.load_verifiers_prime_plugin", lambda: DummyPlugin()
    )
    monkeypatch.setattr("prime_cli.verifiers_bridge.Config", lambda: DummyConfig())

    class ExplodingInferenceClient:
        def __init__(self, timeout=None):
            raise AssertionError("endpoint registry aliases should skip Prime Inference preflight")

    captured = {}
    monkeypatch.setattr("prime_cli.verifiers_bridge.InferenceClient", ExplodingInferenceClient)
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge._prepare_v1_environment",
        lambda env_name, env_dir: (
            captured.update(prepared=(env_name, env_dir))
            or ResolvedEnvironment(
                original=env_name,
                env_name=env_name,
                install_mode="local",
            )
        ),
    )
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge._run_command",
        lambda command, env=None: captured.update({"command": command, "env": env}),
    )

    run_eval_passthrough(
        environment="goblin-questions",
        passthrough_args=[
            "-m",
            "gpt-4.1-mini",
            "--client.base-url",
            "https://api.openai.com/v1",
            "--client.headers",
            '{"X-Test": "yes"}',
            "--env-dir-path",
            str(tmp_path / "custom-envs"),
            "-n",
            "1",
        ],
        skip_upload=True,
        env_path=None,
    )

    command = captured["command"]
    assert command[:3] == ["verifiers.v1.cli.eval.main", "goblin-questions", "-m"]
    assert "gpt-4.1-mini" in command
    assert "--api-base-url" not in command
    assert "--api-key-var" not in command
    assert "--env-dir-path" not in command
    assert command.count("--client.headers") == 1
    header_idx = len(command) - 1 - command[::-1].index("--client.headers")
    headers = json.loads(command[header_idx + 1])
    assert headers["X-Test"] == "yes"
    assert headers["X-PI-Job-Id"].startswith("goblin_questions_gpt_4.1_mini_")
    assert captured["prepared"] == ("goblin-questions", str(tmp_path / "custom-envs"))
    assert captured["env"]["PRIME_API_KEY"] == "test-api-key"
    assert captured["env"]["PRIME_INFERENCE_URL"] == DummyConfig.inference_url
    assert "PRIME_TEAM_ID" not in captured["env"]


def test_eval_resume_forwards_only_resume_arguments(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.load_verifiers_prime_plugin", lambda: DummyPlugin()
    )
    monkeypatch.setattr("prime_cli.verifiers_bridge.Config", lambda: DummyConfig())
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge._validate_model",
        lambda *_args: (_ for _ in ()).throw(AssertionError("resume must skip model validation")),
    )
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge._preflight_inference_billing",
        lambda *_args: (_ for _ in ()).throw(AssertionError("resume must skip billing preflight")),
    )
    captured = {}
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge._run_command",
        lambda command, env=None: captured.update(command=command, env=env),
    )
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.push_eval_results_to_hub",
        lambda **kwargs: captured.update(upload=kwargs),
    )

    output_dir = tmp_path / "previous-run"
    output_dir.mkdir()
    (output_dir / "config.toml").write_text(
        """model = "openai/gpt-4.1-mini"

[taskset]
id = "wiki-search"

[client.headers]
X-PI-Job-Id = "existing-job-id"
X-Prime-Eval-Env-Display = "primeintellect/wiki-search"
""",
        encoding="utf-8",
    )
    (output_dir / "results.jsonl").write_text(
        json.dumps({"id": 0, "reward": 1.0}) + "\n",
        encoding="utf-8",
    )
    run_eval_passthrough(
        environment=str(output_dir),
        passthrough_args=["--resume", str(output_dir)],
        skip_upload=False,
        env_path=None,
    )

    assert captured["command"] == [
        "verifiers.v1.cli.eval.main",
        "--resume",
        str(output_dir),
    ]
    assert captured["env"]["PRIME_API_KEY"] == "test-api-key"
    assert captured["env"]["PRIME_INFERENCE_URL"] == DummyConfig.inference_url
    upload_kwargs = captured["upload"]
    assert upload_kwargs["env_name"] == "wiki-search"
    assert upload_kwargs["job_id"] == "existing-job-id"
    assert upload_kwargs["env_path"] is None
    assert upload_kwargs["upstream_slug"] == "primeintellect/wiki-search"
    upload = upload_kwargs["upload"]
    assert upload.model_name == "openai/gpt-4.1-mini"
    assert upload.env == "wiki-search"
    assert upload.results == [{"id": 0, "reward": 1.0, "example_id": 0}]
    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["num_examples"] == 1
    assert metadata["rollouts_per_example"] == 1


def test_eval_run_missing_endpoint_registry_falls_back_to_configured_inference(
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.load_verifiers_prime_plugin", lambda: DummyPlugin()
    )
    monkeypatch.setattr("prime_cli.verifiers_bridge.Config", lambda: DummyConfig())

    def fail_load_endpoints(_path):
        raise AssertionError("missing endpoint registry should not be loaded")

    monkeypatch.setattr("verifiers.utils.eval_utils.load_endpoints", fail_load_endpoints)

    seen_payloads = []

    class DummyInferenceClient:
        def __init__(self, timeout=None):
            pass

        def retrieve_model(self, model):
            return {"id": model}

        def chat_completion(self, payload, stream=False):
            seen_payloads.append(payload)
            return {"id": "cmpl-123", "choices": []}

    monkeypatch.setattr("prime_cli.verifiers_bridge.InferenceClient", DummyInferenceClient)
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge._prepare_v1_environment",
        lambda *args, **kwargs: (_ for _ in ()).throw(typer.Exit(0)),
    )

    with pytest.raises(typer.Exit) as exc_info:
        run_eval_passthrough(
            environment="single_turn_math",
            passthrough_args=["-m", "openai/gpt-4.1-mini"],
            skip_upload=False,
            env_path=None,
        )

    assert cast(typer.Exit, exc_info.value).exit_code == 0
    assert seen_payloads == [
        {
            "model": "openai/gpt-4.1-mini",
            "messages": [{"role": "user", "content": "Reply with OK."}],
        }
    ]


def test_gepa_run_provider_short_flag_does_not_override_env_dir_path(monkeypatch):
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.load_verifiers_prime_plugin", lambda: DummyPlugin()
    )
    monkeypatch.setattr("prime_cli.verifiers_bridge.Config", lambda: DummyConfig())
    captured = {}
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge._prepare_single_environment",
        lambda environment, env_dir_path: (
            captured.update({"environment": environment, "env_dir_path": env_dir_path})
            or ResolvedEnvironment(
                original=environment,
                env_name=environment,
                install_mode="local",
            )
        ),
    )
    monkeypatch.setattr("prime_cli.verifiers_bridge._run_command", lambda command, env=None: None)
    # the provider table lives in verifiers; the venv's verifiers may predate it
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge._provider_endpoint",
        lambda provider: ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    )

    run_gepa_passthrough(
        environment_or_config="single_turn_math",
        passthrough_args=["-p", "openai", "-m", "gpt-4.1-mini"],
    )

    assert captured == {
        "environment": "single_turn_math",
        "env_dir_path": "./environments",
    }


def test_eval_run_continues_when_model_validation_times_out(monkeypatch):
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.load_verifiers_prime_plugin", lambda: DummyPlugin()
    )
    monkeypatch.setattr("prime_cli.verifiers_bridge.Config", lambda: DummyConfig())

    seen_model_timeouts = []

    class DummyInferenceClient:
        def __init__(self, timeout=None):
            seen_model_timeouts.append(timeout)

        def retrieve_model(self, model):
            raise httpx.ReadTimeout("timed out")

        def chat_completion(self, payload, stream=False):
            return {"id": "cmpl-123", "choices": []}

    monkeypatch.setattr("prime_cli.verifiers_bridge.InferenceClient", DummyInferenceClient)
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge._prepare_v1_environment",
        lambda *args, **kwargs: (_ for _ in ()).throw(typer.Exit(0)),
    )

    with pytest.raises(typer.Exit) as exc_info:
        run_eval_passthrough(
            environment="single_turn_math",
            passthrough_args=["-m", "moonshotai/kimi-k2.5"],
            skip_upload=False,
            env_path=None,
        )

    assert exc_info.value.exit_code == 0
    assert len(seen_model_timeouts) == 2
    assert seen_model_timeouts[0].read == 300.0


def test_eval_run_continues_when_billing_preflight_times_out(monkeypatch):
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.load_verifiers_prime_plugin", lambda: DummyPlugin()
    )
    monkeypatch.setattr("prime_cli.verifiers_bridge.Config", lambda: DummyConfig())

    seen_billing_timeouts = []

    class DummyInferenceClient:
        def __init__(self, timeout=None):
            seen_billing_timeouts.append(timeout)

        def retrieve_model(self, model):
            return {"id": model}

        def chat_completion(self, payload, stream=False):
            raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr("prime_cli.verifiers_bridge.InferenceClient", DummyInferenceClient)
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge._prepare_v1_environment",
        lambda *args, **kwargs: (_ for _ in ()).throw(typer.Exit(0)),
    )

    with pytest.raises(typer.Exit) as exc_info:
        run_eval_passthrough(
            environment="single_turn_math",
            passthrough_args=["-m", "moonshotai/kimi-k2.5"],
            skip_upload=False,
            env_path=None,
        )

    assert exc_info.value.exit_code == 0
    assert len(seen_billing_timeouts) == 2
    assert seen_billing_timeouts[1].read == 300.0


@pytest.mark.parametrize("prefix", ["", "@"])
def test_eval_config_uses_v1_taskset(monkeypatch, tmp_path, prefix):
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.load_verifiers_prime_plugin", lambda: DummyPlugin()
    )
    monkeypatch.setattr("prime_cli.verifiers_bridge.Config", lambda: DummyConfig())
    monkeypatch.setattr("prime_cli.verifiers_bridge._validate_model", lambda *args: None)
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge._preflight_inference_billing",
        lambda *args: None,
    )

    config_path = tmp_path / "reverse-text.toml"
    config_path.write_text(
        """
model = "openai/gpt-4.1-mini"

[taskset]
id = "primeintellect/wordle"
""".strip(),
        encoding="utf-8",
    )

    prepared = []
    commands = []

    def fake_prepare(env_reference, env_dir_path):
        prepared.append((env_reference, env_dir_path))
        return ResolvedEnvironment(
            original=env_reference,
            env_name=env_reference.split("/")[-1],
            install_mode="hub",
        )

    def fake_run_command(command, env=None):
        commands.append(command)

    monkeypatch.setattr("prime_cli.verifiers_bridge._prepare_v1_environment", fake_prepare)
    monkeypatch.setattr("prime_cli.verifiers_bridge._run_command", fake_run_command)

    run_eval_passthrough(
        environment=f"{prefix}{config_path}",
        passthrough_args=[],
        skip_upload=True,
        env_path=None,
    )

    assert prepared == [("primeintellect/wordle", "./environments")]
    assert commands
    # pydantic-config only accepts a root config file as two tokens: `@ path`;
    # the hub slug is installed and pinned to the local name in a config copy
    assert commands[0][1] == "@"
    assert commands[0][2] != str(config_path)
    assert toml.load(commands[0][2])["taskset"]["id"] == "wordle"
    header_idx = commands[0].index("--client.headers")
    headers = json.loads(commands[0][header_idx + 1])
    assert headers["X-PI-Job-Id"].startswith("wordle_openai_gpt_4.1_mini_")


@pytest.mark.parametrize("from_config", [False, True])
def test_eval_dry_run_skips_inference_preflight(monkeypatch, tmp_path, from_config):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.load_verifiers_prime_plugin", lambda: DummyPlugin()
    )
    monkeypatch.setattr("prime_cli.verifiers_bridge.Config", lambda: DummyConfig())
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge._validate_model",
        lambda *_args: (_ for _ in ()).throw(AssertionError("dry-run must not make API calls")),
    )
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge._preflight_inference_billing",
        lambda *_args: (_ for _ in ()).throw(AssertionError("dry-run must not make API calls")),
    )
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge._prepare_v1_environment",
        lambda environment, _env_dir: ResolvedEnvironment(
            original=environment,
            env_name=environment,
            install_mode="local",
        ),
    )
    captured = {}
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge._run_command",
        lambda command, env=None: captured.update(command=command),
    )

    if from_config:
        config_path = tmp_path / "eval.toml"
        config_path.write_text('dry_run = true\n\n[taskset]\nid = "gsm8k-v1"\n', encoding="utf-8")
        environment = str(config_path)
        args = []
    else:
        environment = "gsm8k-v1"
        args = ["--dry-run", "true"]

    run_eval_passthrough(
        environment=environment,
        passthrough_args=args,
        skip_upload=True,
        env_path=None,
    )

    assert captured["command"][0] == "verifiers.v1.cli.eval.main"


def test_eval_config_preserves_output_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.load_verifiers_prime_plugin", lambda: DummyPlugin()
    )
    monkeypatch.setattr("prime_cli.verifiers_bridge.Config", lambda: DummyConfig())
    monkeypatch.setattr("prime_cli.verifiers_bridge._validate_model", lambda *args: None)
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge._preflight_inference_billing", lambda *args: None
    )
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge._prepare_v1_environment",
        lambda *_: ResolvedEnvironment(
            original="gsm8k-v1", env_name="gsm8k-v1", install_mode="local"
        ),
    )

    output_dir = tmp_path / "configured-output"
    config_path = tmp_path / "eval.toml"
    config_path.write_text(
        f'output_dir = "{output_dir}"\n\n[taskset]\nid = "gsm8k-v1"\n',
        encoding="utf-8",
    )
    captured = {}

    def fake_run(command, env=None):
        captured["command"] = command
        output_dir.mkdir()
        # a real run writes its resolved config.toml alongside the results
        (output_dir / "config.toml").write_text(
            'model = "openai/gpt-4.1-mini"\n\n[taskset]\nid = "gsm8k-v1"\n', encoding="utf-8"
        )
        (output_dir / "results.jsonl").write_text(
            json.dumps(
                {"task": {"idx": 0, "prompt": None}, "nodes": [], "rewards": {"correct": 1.0}}
            )
            + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr("prime_cli.verifiers_bridge._run_command", fake_run)

    run_eval_passthrough(
        environment=str(config_path),
        passthrough_args=[],
        skip_upload=True,
        env_path=None,
    )

    assert "--output-dir" not in captured["command"]
    assert (output_dir / "metadata.json").exists()


@pytest.mark.parametrize(
    ("task_ids", "expected_rollouts"),
    [
        ([0, 1], 1),
        ([0, 0, 1], None),
    ],
)
def test_eval_writes_v1_metadata(monkeypatch, tmp_path, task_ids, expected_rollouts):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.load_verifiers_prime_plugin", lambda: DummyPlugin()
    )
    monkeypatch.setattr("prime_cli.verifiers_bridge.Config", lambda: DummyConfig())
    monkeypatch.setattr("prime_cli.verifiers_bridge._validate_model", lambda *args: None)
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge._preflight_inference_billing",
        lambda *args: None,
    )

    monkeypatch.setattr(
        "prime_cli.verifiers_bridge._prepare_v1_environment",
        lambda *_args: ResolvedEnvironment(
            original="wiki-search", env_name="wiki-search", install_mode="local"
        ),
    )

    captured = {}

    def fake_run(command, env=None):
        captured["command"] = command
        output_dir = Path(command[command.index("--output-dir") + 1])
        output_dir.mkdir(parents=True)
        # a real run writes its resolved config.toml alongside the results
        (output_dir / "config.toml").write_text(
            'model = "openai/gpt-4.1-mini"\n\n[taskset]\nid = "wiki-search"\n', encoding="utf-8"
        )
        rewards = [1.0, 0.5, 0.75]
        rows = [
            {
                "task": {"idx": task_id, "prompt": None},
                "nodes": [],
                "rewards": {"correct": rewards[index]},
            }
            for index, task_id in enumerate(task_ids)
        ]
        (output_dir / "results.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    monkeypatch.setattr("prime_cli.verifiers_bridge._run_command", fake_run)

    run_eval_passthrough(
        environment="wiki-search",
        passthrough_args=["-m", "openai/gpt-4.1-mini"],
        skip_upload=True,
        env_path=None,
    )

    (metadata_path,) = tmp_path.glob("outputs/evals/*/*/metadata.json")
    metadata = json.loads(metadata_path.read_text())
    base_url_index = captured["command"].index("--client.base-url")
    assert captured["command"][base_url_index + 1] == DummyConfig.inference_url
    assert metadata["num_examples"] == 2
    assert metadata["avg_reward"] == 0.75
    if expected_rollouts is None:
        assert "rollouts_per_example" not in metadata
    else:
        assert metadata["rollouts_per_example"] == expected_rollouts


def test_inference_client_uses_custom_timeout(monkeypatch):
    monkeypatch.setattr("prime_cli.api.inference.Config", lambda: DummyConfig())

    client = InferenceClient(timeout=httpx.Timeout(connect=5.0, read=7.0, write=9.0, pool=11.0))

    assert client._client.timeout.read == 7.0


def test_streaming_error_reads_response_before_formatting(monkeypatch):
    client = InferenceClient.__new__(InferenceClient)
    client.inference_url = "https://api.pinference.ai/api/v1"

    request = httpx.Request("POST", f"{client.inference_url}/chat/completions")
    response = httpx.Response(
        500,
        request=request,
        stream=httpx.ByteStream(b"upstream boom"),
    )

    class DummyStreamResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            raise httpx.HTTPStatusError("boom", request=request, response=response)

    class DummyHTTPClient:
        def stream(self, method, url, json):
            return DummyStreamResponse()

    client._client = DummyHTTPClient()

    with pytest.raises(InferenceAPIError, match="POST .* 500 upstream boom"):
        next(client.chat_completion({"model": "x", "messages": []}, stream=True))

    assert response.text == "upstream boom"
