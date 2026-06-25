from pathlib import Path

import httpx
import pytest
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
    eval_module = "verifiers.cli.commands.eval"
    gepa_module = "verifiers.cli.commands.gepa"

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
        "prime_cli.verifiers_bridge.load_verifiers_prime_plugin", lambda console: DummyPlugin()
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
        "prime_cli.verifiers_bridge.load_verifiers_prime_plugin", lambda console: DummyPlugin()
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
        "prime_cli.verifiers_bridge._prepare_single_environment",
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


def test_eval_run_model_alias_uses_local_endpoint_registry(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "endpoints.toml").write_text(
        "\n".join(
            [
                "[[endpoint]]",
                'endpoint_id = "gpt-4.1-mini"',
                'model = "gpt-4.1-mini"',
                'url = "https://api.openai.com/v1"',
                'key = "OPENAI_API_KEY"',
                'type = "openai_chat_completions"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.load_verifiers_prime_plugin", lambda console: DummyPlugin()
    )
    monkeypatch.setattr("prime_cli.verifiers_bridge.Config", lambda: DummyConfig())

    class ExplodingInferenceClient:
        def __init__(self, timeout=None):
            raise AssertionError("endpoint registry aliases should skip Prime Inference preflight")

    captured = {}
    monkeypatch.setattr("prime_cli.verifiers_bridge.InferenceClient", ExplodingInferenceClient)
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge._prepare_single_environment",
        lambda *_args, **_kwargs: ResolvedEnvironment(
            original="goblin-questions",
            env_name="goblin-questions",
            install_mode="local",
        ),
    )

    def fake_run_command(command, env=None):
        # Read the temp v1 eval config (passed via `@ <file>`) before the bridge unlinks it.
        config_path = command[command.index("@") + 1]
        captured.update(
            {"command": command, "env": env, "config": Path(config_path).read_text()}
        )

    monkeypatch.setattr("prime_cli.verifiers_bridge._run_command", fake_run_command)

    run_eval_passthrough(
        environment="goblin-questions",
        passthrough_args=["-m", "gpt-4.1-mini", "-n", "1"],
        skip_upload=True,
        env_path=None,
    )

    command = captured["command"]
    assert command[1] == "-c"
    assert "verifiers.v1.cli.eval" in command[2]
    config = captured["config"]
    # The endpoint registry alias resolves the model to its base_url (and skips the Prime
    # Inference preflight, which would raise above).
    assert 'base_url = "https://api.openai.com/v1"' in config
    assert 'id = "goblin-questions"' in config
    assert 'model = "gpt-4.1-mini"' in config
    assert captured["env"]["PRIME_API_KEY"] == "test-api-key"


def test_eval_run_missing_endpoint_registry_falls_back_to_configured_inference(
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.load_verifiers_prime_plugin", lambda console: DummyPlugin()
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
        "prime_cli.verifiers_bridge._prepare_single_environment",
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


def test_eval_run_provider_short_flag_does_not_override_env_dir_path(monkeypatch):
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.load_verifiers_prime_plugin", lambda console: DummyPlugin()
    )
    monkeypatch.setattr("prime_cli.verifiers_bridge.Config", lambda: DummyConfig())
    captured = {}
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge._prepare_single_environment",
        lambda _plugin, environment, env_dir_path: captured.update(
            {"environment": environment, "env_dir_path": env_dir_path}
        )
        or ResolvedEnvironment(
            original=environment,
            env_name=environment,
            install_mode="local",
        ),
    )
    monkeypatch.setattr("prime_cli.verifiers_bridge._run_command", lambda command, env=None: None)

    run_eval_passthrough(
        environment="single_turn_math",
        passthrough_args=["-p", "openai", "-m", "gpt-4.1-mini"],
        skip_upload=True,
        env_path=None,
    )

    assert captured == {
        "environment": "single_turn_math",
        "env_dir_path": "./environments",
    }


def test_eval_run_empty_passthrough_arg_does_not_override_env_dir_path(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.load_verifiers_prime_plugin", lambda console: DummyPlugin()
    )
    monkeypatch.setattr("prime_cli.verifiers_bridge.Config", lambda: DummyConfig())
    captured = {}
    envs_path = str(tmp_path / "envs")
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge._prepare_single_environment",
        lambda _plugin, environment, env_dir_path: captured.update(
            {"environment": environment, "env_dir_path": env_dir_path}
        )
        or ResolvedEnvironment(
            original=environment,
            env_name=environment,
            install_mode="local",
        ),
    )
    monkeypatch.setattr("prime_cli.verifiers_bridge._run_command", lambda command, env=None: None)

    run_eval_passthrough(
        environment="single_turn_math",
        passthrough_args=["-p", "openai", "", envs_path, "-m", "gpt-4.1-mini"],
        skip_upload=True,
        env_path=None,
    )

    assert captured == {
        "environment": "single_turn_math",
        "env_dir_path": "./environments",
    }


def test_eval_run_uses_long_env_dir_path_without_treating_it_as_provider(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.load_verifiers_prime_plugin", lambda console: DummyPlugin()
    )
    monkeypatch.setattr("prime_cli.verifiers_bridge.Config", lambda: DummyConfig())

    class DummyInferenceClient:
        def __init__(self, timeout=None):
            pass

        def retrieve_model(self, model):
            return {"id": model}

        def chat_completion(self, payload, stream=False):
            return {"id": "cmpl-123", "choices": []}

    captured = {}
    envs_path = str(tmp_path / "envs")
    monkeypatch.setattr("prime_cli.verifiers_bridge.InferenceClient", DummyInferenceClient)
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge._prepare_single_environment",
        lambda _plugin, environment, env_dir_path: captured.update(
            {"environment": environment, "env_dir_path": env_dir_path}
        )
        or ResolvedEnvironment(
            original=environment,
            env_name=environment,
            install_mode="local",
        ),
    )
    monkeypatch.setattr("prime_cli.verifiers_bridge._run_command", lambda command, env=None: None)

    run_eval_passthrough(
        environment="single_turn_math",
        passthrough_args=["--env-dir-path", envs_path, "-m", "openai/gpt-4.1-mini"],
        skip_upload=True,
        env_path=None,
    )

    assert captured == {
        "environment": "single_turn_math",
        "env_dir_path": envs_path,
    }


def test_gepa_run_provider_short_flag_does_not_override_env_dir_path(monkeypatch):
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.load_verifiers_prime_plugin", lambda console: DummyPlugin()
    )
    monkeypatch.setattr("prime_cli.verifiers_bridge.Config", lambda: DummyConfig())
    captured = {}
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge._prepare_single_environment",
        lambda _plugin, environment, env_dir_path: captured.update(
            {"environment": environment, "env_dir_path": env_dir_path}
        )
        or ResolvedEnvironment(
            original=environment,
            env_name=environment,
            install_mode="local",
        ),
    )
    monkeypatch.setattr("prime_cli.verifiers_bridge._run_command", lambda command, env=None: None)

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
        "prime_cli.verifiers_bridge.load_verifiers_prime_plugin", lambda console: DummyPlugin()
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
        "prime_cli.verifiers_bridge._prepare_single_environment",
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
        "prime_cli.verifiers_bridge.load_verifiers_prime_plugin", lambda console: DummyPlugin()
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
        "prime_cli.verifiers_bridge._prepare_single_environment",
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


def test_eval_provider_short_flag_is_not_treated_as_env_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.load_verifiers_prime_plugin", lambda console: DummyPlugin()
    )
    monkeypatch.setattr("prime_cli.verifiers_bridge.Config", lambda: DummyConfig())
    monkeypatch.setattr("prime_cli.verifiers_bridge._validate_model", lambda *args: None)
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge._preflight_inference_billing",
        lambda *args: None,
    )

    config_path = tmp_path / "eval.toml"
    config_path.write_text(
        """
[[eval]]
env_id = "wiki-search"
""".strip(),
        encoding="utf-8",
    )

    prepared = []

    def fake_prepare(_plugin, env_reference, env_dir_path):
        prepared.append((env_reference, env_dir_path))

    monkeypatch.setattr("prime_cli.verifiers_bridge._prepare_single_environment", fake_prepare)
    monkeypatch.setattr("prime_cli.verifiers_bridge._run_command", lambda *args, **kwargs: None)

    run_eval_passthrough(
        environment=str(config_path),
        passthrough_args=["-p", "openai"],
        skip_upload=True,
        env_path=None,
    )

    assert prepared == [("wiki-search", "./environments")]


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
