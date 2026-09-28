import httpx
import pytest
from prime_cli.api.inference import InferenceClient, InferencePaymentRequiredError


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
