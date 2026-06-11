import json
from pathlib import Path

import httpx
import pytest

from prime_sandboxes.core.client import APIClient, AsyncAPIClient


class RecordingTransport(httpx.BaseTransport):
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, request=request, json={"ok": True})


class AsyncRecordingTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, request=request, json={"ok": True})


@pytest.fixture
def temp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in (
        "PRIME_API_KEY",
        "PRIME_API_BASE_URL",
        "PRIME_BASE_URL",
        "PRIME_SANDBOX_BASE_URL",
        "PRIME_SANDBOX_INGRESS_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def write_config(home: Path, config: dict[str, str]) -> None:
    config_dir = home / ".prime"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(json.dumps(config))


def test_sync_client_routes_only_sandbox_endpoints_to_sandbox_base_url(
    temp_home: Path,
) -> None:
    write_config(
        temp_home,
        {
            "api_key": "test-key",
            "base_url": "https://api.example",
            "sandbox_base_url": "https://sandbox.example",
        },
    )
    transport = RecordingTransport()
    client = APIClient()
    client.client = httpx.Client(transport=transport)

    assert client.request("GET", "/sandbox") == {"ok": True}
    assert client.request("GET", "sandbox/sbx-1") == {"ok": True}
    assert client.request("GET", "sandboxed") == {"ok": True}
    assert client.request("GET", "/template/registry-credentials") == {"ok": True}

    assert [str(request.url) for request in transport.requests] == [
        "https://sandbox.example/api/v1/sandbox",
        "https://sandbox.example/api/v1/sandbox/sbx-1",
        "https://api.example/api/v1/sandboxed",
        "https://api.example/api/v1/template/registry-credentials",
    ]


def test_sync_client_sandbox_base_url_env_var_overrides_saved_config(
    temp_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_config(
        temp_home,
        {
            "api_key": "test-key",
            "base_url": "https://api.example",
            "sandbox_base_url": "https://saved-sandbox.example",
        },
    )
    monkeypatch.setenv("PRIME_SANDBOX_BASE_URL", "https://env-sandbox.example/api/v1")
    transport = RecordingTransport()
    client = APIClient()
    client.client = httpx.Client(transport=transport)

    assert client.request("GET", "/sandbox/sbx-1") == {"ok": True}

    assert str(transport.requests[0].url) == "https://env-sandbox.example/api/v1/sandbox/sbx-1"


@pytest.mark.asyncio
async def test_async_client_routes_sandbox_endpoints_to_sandbox_base_url(
    temp_home: Path,
) -> None:
    write_config(
        temp_home,
        {
            "api_key": "test-key",
            "base_url": "https://api.example",
            "sandbox_base_url": "https://sandbox.example/api/v1",
        },
    )
    transport = AsyncRecordingTransport()
    client = AsyncAPIClient()
    client.client = httpx.AsyncClient(transport=transport)

    assert await client.request("GET", "/sandbox/sbx-1/auth") == {"ok": True}

    assert str(transport.requests[0].url) == "https://sandbox.example/api/v1/sandbox/sbx-1/auth"
    await client.client.aclose()
