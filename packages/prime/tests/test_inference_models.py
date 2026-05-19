from __future__ import annotations

from typing import Any, Dict

import httpx
import pytest
from prime_cli.api.inference import InferenceAPIError, InferenceClient
from prime_cli.main import app
from typer.testing import CliRunner

TEST_ENV: Dict[str, str] = {
    "COLUMNS": "200",
    "LINES": "50",
    "NO_COLOR": "1",
    "PRIME_DISABLE_VERSION_CHECK": "1",
}


class NoKeyConfig:
    api_key = ""
    inference_url = "https://api.pinference.ai/api/v1"
    team_id = "team-1"


def test_inference_client_requires_api_key_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("prime_cli.api.inference.Config", lambda: NoKeyConfig())

    with pytest.raises(InferenceAPIError, match="No API key"):
        InferenceClient()


def test_inference_client_can_list_models_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("prime_cli.api.inference.Config", lambda: NoKeyConfig())
    created: Dict[str, Any] = {}

    class DummyHTTPClient:
        def get(self, url: str) -> httpx.Response:
            created["url"] = url
            request = httpx.Request("GET", url)
            return httpx.Response(200, request=request, json={"object": "list", "data": []})

    def fake_http_client(**kwargs: Any) -> DummyHTTPClient:
        created["headers"] = kwargs["headers"]
        return DummyHTTPClient()

    monkeypatch.setattr("prime_cli.api.inference.httpx.Client", fake_http_client)

    client = InferenceClient(require_auth=False)

    assert client.list_models() == {"object": "list", "data": []}
    assert created["url"] == "https://api.pinference.ai/api/v1/models"
    assert "Authorization" not in created["headers"]
    assert "X-Prime-Team-ID" not in created["headers"]


def test_models_command_uses_optional_auth_client(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_kwargs: Dict[str, Any] = {}

    class DummyModelsClient:
        def __init__(self, **kwargs: Any) -> None:
            seen_kwargs.update(kwargs)

        def list_models(self) -> Dict[str, Any]:
            return {"object": "list", "data": [{"id": "qwen/qwen3-8b"}]}

    monkeypatch.setattr("prime_cli.commands.inference.InferenceClient", DummyModelsClient)

    result = CliRunner().invoke(
        app,
        ["inference", "models", "--output", "json"],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    assert seen_kwargs["require_auth"] is False
    assert '"id": "qwen/qwen3-8b"' in result.output
