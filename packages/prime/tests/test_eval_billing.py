import httpx
import pytest
from prime_cli.api.inference import (
    InferenceAPIError,
    InferenceClient,
    InferencePaymentRequiredError,
)
from prime_cli.verifiers_bridge import ResolvedEnvironment, run_gepa_passthrough


class DummyConfig:
    api_key = "test-api-key"
    inference_url = "https://api.pinference.ai/api/v1"
    team_id = None


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


def test_gepa_run_provider_short_flag_does_not_override_env_dir_path(monkeypatch):
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.build_verifiers_command",
        lambda name, args: [name, *args],
    )
    monkeypatch.setattr("prime_cli.verifiers_bridge.Config", lambda: DummyConfig())
    captured = {}
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge._prepare_single_environment",
        lambda environment, env_dir_path: captured.update(
            {"environment": environment, "env_dir_path": env_dir_path}
        )
        or ResolvedEnvironment(
            original=environment,
            env_name=environment,
            install_mode="local",
        ),
    )
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge._run_command",
        lambda command, env=None: None,
    )

    run_gepa_passthrough(
        environment_or_config="single_turn_math",
        passthrough_args=["-p", "openai", "-m", "gpt-4.1-mini"],
    )

    assert captured == {
        "environment": "single_turn_math",
        "env_dir_path": "./environments",
    }


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
