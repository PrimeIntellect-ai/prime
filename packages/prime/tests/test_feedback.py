from typing import Any, Dict, Optional

import pytest
from prime_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()

TEST_ENV = {
    "COLUMNS": "200",
    "LINES": "50",
    "PRIME_DISABLE_VERSION_CHECK": "1",
    "PRIME_API_KEY": "test-key",
}


@pytest.fixture
def capture_feedback_post(monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    captured: Dict[str, Any] = {}

    def mock_post(
        self: Any, endpoint: str, json: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        captured["endpoint"] = endpoint
        captured["json"] = json
        return {"message": "Feedback submitted"}

    monkeypatch.setattr("prime_cli.client.APIClient.post", mock_post)
    return captured


def test_feedback_submits_general_without_run_id(
    capture_feedback_post: Dict[str, Any], keys: Any
) -> None:
    # move down to general (bug, feature, general); blank run id; then the message
    keys.send(keys.DOWN + keys.DOWN + keys.ENTER).send(keys.ENTER).text("The CLI is great")
    result = runner.invoke(app, ["feedback"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert "Feedback submitted" in result.output
    assert capture_feedback_post["endpoint"] == "/feedback"
    payload = capture_feedback_post["json"]
    assert payload["category"] == "general"
    assert payload["message"] == "The CLI is great"
    assert payload["run_id"] is None
    assert payload["cli_version"]


def test_feedback_submits_bug_with_run_id(capture_feedback_post: Dict[str, Any], keys: Any) -> None:
    # bug is the first choice, then run id, then message
    keys.send(keys.ENTER).text("run_abc123").text("Training crashed on step 42")
    result = runner.invoke(app, ["feedback"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    payload = capture_feedback_post["json"]
    assert payload["category"] == "bug"
    assert payload["run_id"] == "run_abc123"
    assert payload["message"] == "Training crashed on step 42"


def test_feedback_rejects_empty_message(capture_feedback_post: Dict[str, Any], keys: Any) -> None:
    # general, no run id, one empty message (re-prompted), then the real one
    keys.send(keys.DOWN + keys.DOWN + keys.ENTER).send(keys.ENTER).send(keys.ENTER).text(
        "Actually here is my feedback"
    )
    result = runner.invoke(app, ["feedback"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert capture_feedback_post["json"]["message"] == "Actually here is my feedback"
