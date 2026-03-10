import httpx
import pytest
import typer
from prime_cli.api.inference import (
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
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.InferenceClient.retrieve_model",
        lambda self, model: {"id": model},
    )

    def fake_chat_completion(self, payload, stream=False):
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
