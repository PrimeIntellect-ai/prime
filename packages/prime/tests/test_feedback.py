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
    capture_feedback_post: Dict[str, Any],
) -> None:
    # 3 = general, blank run id, message on final prompt
    result = runner.invoke(app, ["feedback"], input="3\n\nThe CLI is great\n", env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert "Feedback submitted" in result.output
    assert capture_feedback_post["endpoint"] == "/feedback"
    payload = capture_feedback_post["json"]
    assert payload["category"] == "general"
    assert payload["message"] == "The CLI is great"
    assert payload["run_id"] is None
    assert payload["cli_version"]


def test_feedback_submits_bug_with_run_id(
    capture_feedback_post: Dict[str, Any],
) -> None:
    result = runner.invoke(
        app,
        ["feedback"],
        input="1\nrun_abc123\nTraining crashed on step 42\n",
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    payload = capture_feedback_post["json"]
    assert payload["category"] == "bug"
    assert payload["run_id"] == "run_abc123"
    assert payload["message"] == "Training crashed on step 42"


def test_feedback_rejects_empty_message(
    capture_feedback_post: Dict[str, Any],
) -> None:
    # category general, no run id, empty message once, then real message
    result = runner.invoke(
        app,
        ["feedback"],
        input="3\n\n\nActually here is my feedback\n",
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    # Two '> ' prompts prove the first empty attempt was re-prompted.
    assert result.output.count("> ") >= 2
    assert capture_feedback_post["json"]["message"] == "Actually here is my feedback"
