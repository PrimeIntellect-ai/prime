"""Tests for `prime inference evaluate`."""

from __future__ import annotations

from typing import Any, Dict

import pytest
from prime_cli.api.inference import InferenceAPIError
from prime_cli.commands.inference import _parse_question
from prime_cli.main import app
from typer.testing import CliRunner

TEST_ENV: Dict[str, str] = {
    "COLUMNS": "200",
    "LINES": "50",
    "NO_COLOR": "1",
    "PRIME_DISABLE_VERSION_CHECK": "1",
}


class DummyClient:
    """Records the last payload and returns a canned evaluation response."""

    last_payload: Dict[str, Any] = {}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def evaluation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        DummyClient.last_payload = payload
        return {
            "answers": {
                "ok": {"type": "boolean", "probability": 0.97},
            },
            "rounding": {"probabilityDecimals": 2, "scoreDecimals": 2},
            "usage": {"inputTokens": 42, "outputTokens": 7},
            "warnings": [],
        }


def test_parse_boolean_question() -> None:
    q = _parse_question("ok:boolean:Was a refund issued?")
    assert q == {"type": "boolean", "instructions": "Was a refund issued?"}


def test_parse_boolean_preserves_colons_in_instructions() -> None:
    q = _parse_question("ok:boolean:Was a refund issued: yes or no?")
    assert q == {"type": "boolean", "instructions": "Was a refund issued: yes or no?"}


def test_evaluate_rejects_duplicate_question_ids(monkeypatch) -> None:
    monkeypatch.setattr("prime_cli.commands.inference.InferenceClient", DummyClient)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "inference",
            "evaluate",
            "typesafe-ai/jev",
            "state",
            "-q",
            "ok:boolean:Fine?",
            "-q",
            "ok:boolean:Also fine?",
        ],
        env=TEST_ENV,
    )

    assert result.exit_code == 1
    assert "Duplicate question id" in result.output


def test_parse_choice_question_with_criteria() -> None:
    q = _parse_question("tone:choice:Classify:professional=Formal,casual")
    assert q == {
        "type": "choice",
        "instructions": "Classify",
        "criteria": {"professional": "Formal", "casual": None},
    }


def test_parse_score_question_levels() -> None:
    q = _parse_question("quality:score:Rate it:Bad,Okay,Good")
    assert q == {
        "type": "score",
        "instructions": "Rate it",
        "criteria": ["Bad", "Okay", "Good"],
    }


@pytest.mark.parametrize(
    "bad",
    [
        "only-two-parts",
        "bad:type:instructions",
        "tone:choice:Needs criteria",
        "q:score:Needs levels",
        "q:score:Only one level",
    ],
)
def test_parse_question_rejects_invalid(bad: str) -> None:
    with pytest.raises(ValueError):
        _parse_question(bad)


def test_evaluate_sends_protocol_payload(monkeypatch) -> None:
    monkeypatch.setattr("prime_cli.commands.inference.InferenceClient", DummyClient)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "inference",
            "evaluate",
            "typesafe-ai/jev",
            "The agent issued a refund.",
            "-q",
            "ok:boolean:Was a refund issued?",
        ],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    assert DummyClient.last_payload == {
        "model": "typesafe-ai/jev",
        "state": "The agent issued a refund.",
        "questions": {"ok": {"type": "boolean", "instructions": "Was a refund issued?"}},
    }
    assert "0.97" in result.output


def test_evaluate_rejects_model_without_question(monkeypatch) -> None:
    monkeypatch.setattr("prime_cli.commands.inference.InferenceClient", DummyClient)

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["inference", "evaluate", "typesafe-ai/jev", "some state"],
        env=TEST_ENV,
    )

    assert result.exit_code == 1
    assert "--question" in result.output


def test_evaluate_surfaces_api_errors(monkeypatch) -> None:
    class FailingClient(DummyClient):
        def evaluation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
            raise InferenceAPIError("POST .../evaluations failed: 404 model not found")

    monkeypatch.setattr("prime_cli.commands.inference.InferenceClient", FailingClient)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "inference",
            "evaluate",
            "typesafe-ai/jev",
            "state",
            "-q",
            "ok:boolean:Fine?",
        ],
        env=TEST_ENV,
    )

    assert result.exit_code == 1
    assert "404" in result.output
