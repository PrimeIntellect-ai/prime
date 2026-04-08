import httpx
import pytest
import typer
from prime_cli.api.inference import (
    InferenceAPIError,
    InferenceClient,
    InferencePaymentRequiredError,
    InferenceTimeoutError,
)
from prime_cli.verifiers_bridge import run_eval_passthrough


class DummyConfig:
    api_key = "test-api-key"
    inference_url = "https://api.pinference.ai/api/v1"
    team_id = None


class DummyPlugin:
    eval_module = "verifiers.cli.commands.eval"

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
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.InferenceClient.retrieve_model",
        lambda self, model, timeout=None: {"id": model},
    )

    def fake_chat_completion(self, payload, stream=False, timeout=None):
        raise InferencePaymentRequiredError(error_message)

    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.InferenceClient.chat_completion",
        fake_chat_completion,
    )

    with pytest.raises(typer.Exit) as exc_info:
        run_eval_passthrough(
            environment="single_turn_math",
            passthrough_args=["-m", "deepseek/deepseek-chat"],
            skip_upload=False,
            env_path=None,
        )

    assert exc_info.value.exit_code == 1


def test_eval_preflight_omits_max_tokens(monkeypatch):
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.load_verifiers_prime_plugin", lambda console: DummyPlugin()
    )
    monkeypatch.setattr("prime_cli.verifiers_bridge.Config", lambda: DummyConfig())
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.InferenceClient.retrieve_model",
        lambda self, model, timeout=None: {"id": model},
    )

    seen_payloads = []
    seen_timeouts = []

    def fake_chat_completion(self, payload, stream=False, timeout=None):
        seen_payloads.append(payload)
        seen_timeouts.append(timeout)
        return {"id": "cmpl-123", "choices": []}

    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.InferenceClient.chat_completion",
        fake_chat_completion,
    )
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

    assert exc_info.value.exit_code == 0
    assert seen_payloads == [
        {
            "model": "openai/gpt-4.1-mini",
            "messages": [{"role": "user", "content": "Reply with OK."}],
        }
    ]
    assert len(seen_timeouts) == 1
    assert seen_timeouts[0].read == 1800.0


def test_eval_run_continues_when_model_validation_times_out(monkeypatch):
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.load_verifiers_prime_plugin", lambda console: DummyPlugin()
    )
    monkeypatch.setattr("prime_cli.verifiers_bridge.Config", lambda: DummyConfig())

    seen_model_timeouts = []

    def fake_retrieve_model(self, model, timeout=None):
        seen_model_timeouts.append(timeout)
        raise InferenceTimeoutError("GET https://api.pinference.ai/api/v1/models/foo timed out")

    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.InferenceClient.retrieve_model",
        fake_retrieve_model,
    )
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.InferenceClient.chat_completion",
        lambda self, payload, stream=False, timeout=None: {"id": "cmpl-123", "choices": []},
    )
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
    assert len(seen_model_timeouts) == 1
    assert seen_model_timeouts[0].read == 1800.0


def test_eval_run_continues_when_billing_preflight_times_out(monkeypatch):
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.load_verifiers_prime_plugin", lambda console: DummyPlugin()
    )
    monkeypatch.setattr("prime_cli.verifiers_bridge.Config", lambda: DummyConfig())
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.InferenceClient.retrieve_model",
        lambda self, model, timeout=None: {"id": model},
    )

    seen_billing_timeouts = []

    def fake_chat_completion(self, payload, stream=False, timeout=None):
        seen_billing_timeouts.append(timeout)
        raise InferenceTimeoutError(
            "POST https://api.pinference.ai/api/v1/chat/completions timed out"
        )

    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.InferenceClient.chat_completion",
        fake_chat_completion,
    )
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
    assert len(seen_billing_timeouts) == 1
    assert seen_billing_timeouts[0].read == 1800.0


def test_inference_client_maps_timeout_to_inference_timeout_error():
    client = InferenceClient.__new__(InferenceClient)
    client.inference_url = "https://api.pinference.ai/api/v1"

    class DummyHTTPClient:
        def get(self, url, **kwargs):
            raise httpx.ReadTimeout("timed out")

    client._client = DummyHTTPClient()

    with pytest.raises(InferenceTimeoutError, match="GET .* timed out"):
        client.retrieve_model("moonshotai/kimi-k2.5")


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
