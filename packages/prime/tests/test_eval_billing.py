import httpx
import pytest
import typer
from prime_cli.api.inference import (
    InferenceAPIError,
    InferenceClient,
    InferencePaymentRequiredError,
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

    assert exc_info.value.exit_code == 1


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

    assert exc_info.value.exit_code == 0
    assert seen_payloads == [
        {
            "model": "openai/gpt-4.1-mini",
            "messages": [{"role": "user", "content": "Reply with OK."}],
        }
    ]
    assert len(seen_timeouts) == 2
    assert all(timeout.read == 300.0 for timeout in seen_timeouts)


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
