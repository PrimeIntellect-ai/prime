"""Wire-contract tests for hosted evaluation SDK methods."""

import asyncio
from typing import Any, Dict, Optional

import pytest

from prime_evals.core import APIClient, APIError
from prime_evals.evals import AsyncEvalsClient, EvalsClient


class RecordingClient:
    """Fake API client matching the injected-client contract (request + config)."""

    def __init__(self, team_id: Optional[str] = None):
        self.config = type("Config", (), {"team_id": team_id})()
        self.calls = []

    def request(self, method, endpoint, params=None, json=None, timeout=None):
        self.calls.append({"method": method, "endpoint": endpoint, "json": json})
        return {"evaluation_id": "eval-123", "evaluation_ids": ["eval-123"]}


def _eval_config() -> Dict[str, Any]:
    # Independent production inference credentials stay caller-owned and
    # must reach the wire verbatim, separate from platform auth.
    return {
        "num_examples": 5,
        "rollouts_per_example": 3,
        "api_base_url": "https://api.inference.example/v1",
        "api_key_var": "PRODUCTION_INFERENCE_KEY",
        "custom_secrets": {"PRODUCTION_INFERENCE_KEY": "sk-prod"},
        "headers": ["X-Custom-Header: custom-value"],
    }


def test_create_hosted_evaluation_posts_expected_payload():
    client = RecordingClient()
    evals = EvalsClient(client)
    eval_config = _eval_config()

    response = evals.create_hosted_evaluation(
        ["env-1", "env-2"],
        "openai/gpt-4.1-mini",
        eval_config,
        name="my-eval",
    )

    assert response == {"evaluation_id": "eval-123", "evaluation_ids": ["eval-123"]}
    assert len(client.calls) == 1, "create is non-idempotent: exactly one POST"
    call = client.calls[0]
    assert call["method"] == "POST"
    assert call["endpoint"] == "/hosted-evaluations"
    assert call["json"] == {
        "environment_ids": ["env-1", "env-2"],
        "inference_model": "openai/gpt-4.1-mini",
        "eval_config": eval_config,
        "name": "my-eval",
    }


def test_create_hosted_evaluation_omits_unset_optional_fields():
    client = RecordingClient(team_id=None)
    evals = EvalsClient(client)

    response = evals.create_hosted_evaluation(["env-1"], "openai/gpt-4.1-mini", {})

    assert response == {"evaluation_id": "eval-123", "evaluation_ids": ["eval-123"]}
    assert client.calls[0]["json"] == {
        "environment_ids": ["env-1"],
        "inference_model": "openai/gpt-4.1-mini",
        "eval_config": {},
    }


def test_create_hosted_evaluation_team_id_falls_back_to_config():
    client = RecordingClient(team_id="team-123")
    evals = EvalsClient(client)

    evals.create_hosted_evaluation(["env-1"], "openai/gpt-4.1-mini", {})

    assert client.calls[0]["json"]["team_id"] == "team-123"


def test_create_hosted_evaluation_explicit_team_id_overrides_config():
    client = RecordingClient(team_id="team-123")
    evals = EvalsClient(client)

    evals.create_hosted_evaluation(["env-1"], "openai/gpt-4.1-mini", {}, team_id="team-456")

    assert client.calls[0]["json"]["team_id"] == "team-456"


def test_create_hosted_evaluation_propagates_client_errors():
    class FailingClient(RecordingClient):
        def request(self, method, endpoint, params=None, json=None, timeout=None):
            raise APIError("HTTP 402: insufficient funds")

    with pytest.raises(APIError, match="insufficient funds"):
        EvalsClient(FailingClient()).create_hosted_evaluation(["env-1"], "m", {})


def test_cancel_hosted_evaluation_uses_patch_and_returns_raw_response():
    client = RecordingClient()
    evals = EvalsClient(client)

    response = evals.cancel_hosted_evaluation("eval-123")

    assert response == {"evaluation_id": "eval-123", "evaluation_ids": ["eval-123"]}
    assert client.calls == [
        {"method": "PATCH", "endpoint": "/hosted-evaluations/eval-123/cancel", "json": None}
    ]


class RecordingAsyncClient(RecordingClient):
    async def request(self, method, endpoint, params=None, json=None, timeout=None):
        return super().request(method, endpoint, params=params, json=json, timeout=timeout)


def _async_client(team_id: Optional[str] = None) -> AsyncEvalsClient:
    client = AsyncEvalsClient(api_key="test-key")
    client.client = RecordingAsyncClient(team_id=team_id)
    return client


def test_async_create_hosted_evaluation_matches_sync_contract():
    client = _async_client(team_id="team-123")
    eval_config = _eval_config()

    response = asyncio.run(
        client.create_hosted_evaluation(
            ["env-1", "env-2"],
            "openai/gpt-4.1-mini",
            eval_config,
            name="my-eval",
        )
    )

    assert response == {"evaluation_id": "eval-123", "evaluation_ids": ["eval-123"]}
    calls = client.client.calls
    assert len(calls) == 1
    assert calls[0]["method"] == "POST"
    assert calls[0]["endpoint"] == "/hosted-evaluations"
    assert calls[0]["json"] == {
        "environment_ids": ["env-1", "env-2"],
        "inference_model": "openai/gpt-4.1-mini",
        "eval_config": eval_config,
        "name": "my-eval",
        "team_id": "team-123",
    }


def test_async_cancel_hosted_evaluation_uses_patch():
    client = _async_client()

    response = asyncio.run(client.cancel_hosted_evaluation("eval-123"))

    assert response == {"evaluation_id": "eval-123", "evaluation_ids": ["eval-123"]}
    assert client.client.calls == [
        {"method": "PATCH", "endpoint": "/hosted-evaluations/eval-123/cancel", "json": None}
    ]


def test_request_wraps_malformed_json_in_api_error(monkeypatch):
    class MalformedResponse:
        def raise_for_status(self):
            return None

        def json(self):
            raise ValueError("Expecting value")

    class MalformedHttpClient:
        def request(self, method, url, **kwargs):
            return MalformedResponse()

    api_client = APIClient(api_key="test-key")
    monkeypatch.setattr(api_client, "client", MalformedHttpClient())

    with pytest.raises(APIError, match="Invalid JSON in API response"):
        api_client.request("GET", "/evaluations/eval-123")
