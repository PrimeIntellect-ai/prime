"""Tests for `prime inference chat`."""

from __future__ import annotations

from typing import Any, Dict, List

from click.testing import CliRunner
from prime_cli.api.inference import InferenceAPIError
from prime_cli.main import app

TEST_ENV: Dict[str, str] = {
    "COLUMNS": "200",
    "LINES": "50",
    "NO_COLOR": "1",
    "PRIME_DISABLE_VERSION_CHECK": "1",
}


class DummyClient:
    """Records the last payload and returns canned responses."""

    last_payload: Dict[str, Any] = {}
    last_stream: bool = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def chat_completion(self, payload: Dict[str, Any], stream: bool = False):
        DummyClient.last_payload = payload
        DummyClient.last_stream = stream
        if stream:
            return iter(
                [
                    {"choices": [{"delta": {"content": "hel"}}]},
                    {"choices": [{"delta": {"content": "lo"}}]},
                ]
            )
        return {
            "id": "chatcmpl-1",
            "model": payload["model"],
            "choices": [
                {"message": {"role": "assistant", "content": "hi there"}, "finish_reason": "stop"}
            ],
        }


def test_chat_prints_assistant_text(monkeypatch):
    monkeypatch.setattr("prime_cli.commands.inference.InferenceClient", DummyClient)

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["inference", "chat", "some-model", "say hi"],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    assert "hi there" in result.output
    assert DummyClient.last_payload["model"] == "some-model"
    assert DummyClient.last_payload["messages"] == [{"role": "user", "content": "say hi"}]
    assert "stream" not in DummyClient.last_payload


def test_chat_includes_system_prompt(monkeypatch):
    monkeypatch.setattr("prime_cli.commands.inference.InferenceClient", DummyClient)

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["inference", "chat", "some-model", "say hi", "--system", "be terse"],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    messages: List[Dict[str, str]] = DummyClient.last_payload["messages"]
    assert messages[0] == {"role": "system", "content": "be terse"}
    assert messages[1] == {"role": "user", "content": "say hi"}


def test_chat_passes_sampling_options(monkeypatch):
    monkeypatch.setattr("prime_cli.commands.inference.InferenceClient", DummyClient)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "inference",
            "chat",
            "some-model",
            "hi",
            "--temperature",
            "0.2",
            "--max-tokens",
            "16",
        ],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    assert DummyClient.last_payload["temperature"] == 0.2
    assert DummyClient.last_payload["max_tokens"] == 16


def test_chat_streams_tokens(monkeypatch):
    monkeypatch.setattr("prime_cli.commands.inference.InferenceClient", DummyClient)

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["inference", "chat", "some-model", "hi", "--stream"],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    assert DummyClient.last_stream is True
    assert DummyClient.last_payload.get("stream") is True
    assert "hello" in result.output


def test_chat_json_output(monkeypatch):
    monkeypatch.setattr("prime_cli.commands.inference.InferenceClient", DummyClient)

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["inference", "chat", "some-model", "hi", "--output", "json"],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    assert '"id": "chatcmpl-1"' in result.output
    assert '"hi there"' in result.output


def test_chat_stream_with_json_errors(monkeypatch):
    monkeypatch.setattr("prime_cli.commands.inference.InferenceClient", DummyClient)

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["inference", "chat", "some-model", "hi", "--stream", "--output", "json"],
        env=TEST_ENV,
    )

    assert result.exit_code == 1
    assert "not supported" in result.output


def test_chat_reports_api_error(monkeypatch):
    class FailingClient(DummyClient):
        def chat_completion(self, payload, stream=False):
            raise InferenceAPIError("boom")

    monkeypatch.setattr("prime_cli.commands.inference.InferenceClient", FailingClient)

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["inference", "chat", "some-model", "hi"],
        env=TEST_ENV,
    )

    assert result.exit_code == 1
    assert "boom" in result.output


def test_chat_reads_message_from_stdin(monkeypatch):
    monkeypatch.setattr("prime_cli.commands.inference.InferenceClient", DummyClient)

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["inference", "chat", "some-model"],
        input="hi from stdin\n",
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    assert DummyClient.last_payload["messages"] == [{"role": "user", "content": "hi from stdin"}]


def test_chat_rejects_empty_stdin(monkeypatch):
    monkeypatch.setattr("prime_cli.commands.inference.InferenceClient", DummyClient)

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["inference", "chat", "some-model"],
        input="",
        env=TEST_ENV,
    )

    assert result.exit_code == 1
    assert "empty message" in result.output
