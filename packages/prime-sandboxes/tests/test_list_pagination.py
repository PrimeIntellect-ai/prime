"""Cursor pagination on the sandbox list endpoint."""

import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from prime_sandboxes import APIClient, AsyncSandboxClient, SandboxClient, SandboxListResponse


class DummyAPIClient(APIClient):
    def __init__(
        self,
        response: dict[str, Any],
        captured: dict[str, Any] | None = None,
        *,
        team_id: str | None = None,
    ) -> None:
        self.response = response
        self.captured = captured
        self.config = SimpleNamespace(
            team_id=team_id,
            config_dir=Path(tempfile.gettempdir()) / "prime-list-tests",
        )

    def request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if self.captured is not None and params is not None:
            self.captured["params"] = params
        return self.response


def _list_response(**overrides: Any) -> dict[str, Any]:
    response = {
        "sandboxes": [],
        "total": 0,
        "page": 1,
        "per_page": 50,
        "has_next": False,
    }
    response.update(overrides)
    return response


def test_list_sends_cursor_and_omits_it_without_one():
    captured: dict[str, Any] = {}
    client = SandboxClient(DummyAPIClient(_list_response(), captured))

    client.list(cursor="cursor-1")
    assert captured["params"]["cursor"] == "cursor-1"
    assert "page" not in captured["params"]

    client.list()
    assert "cursor" not in captured["params"]
    assert "page" not in captured["params"]


def test_list_parses_cursor_mode_response():
    response = _list_response(total=None, has_next=True, next_cursor="cursor-2")
    client = SandboxClient(DummyAPIClient(response))

    parsed = client.list(cursor="cursor-1")

    assert isinstance(parsed, SandboxListResponse)
    assert parsed.total is None
    assert parsed.has_next is True
    assert parsed.next_cursor == "cursor-2"


def test_async_list_sends_cursor_param(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PRIME_TEAM_ID", "")
    captured: dict[str, Any] = {}
    client = AsyncSandboxClient(api_key="test-key")

    async def request(
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        captured["method"] = method
        captured["path"] = path
        captured["params"] = params
        return _list_response()

    client.client.request = request

    async def scenario() -> None:
        await client.list(cursor="cursor-1")

    asyncio.run(scenario())

    assert captured["method"] == "GET"
    assert captured["path"] == "/sandbox"
    assert captured["params"]["cursor"] == "cursor-1"
