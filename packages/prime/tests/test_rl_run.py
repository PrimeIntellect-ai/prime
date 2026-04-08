from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
from prime_cli.api.rl import RLRun
from prime_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


def _write_wandb_config(config_path: Path) -> None:
    config_path.write_text(
        """
model = \"PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT\"
max_steps = 10

[[env]]
id = \"testuser/test-env\"

[wandb]
entity = \"test-team\"
project = \"test-project\"
""".strip()
        + "\n"
    )


def _fake_run() -> RLRun:
    now = datetime.now(timezone.utc)
    return RLRun.model_validate(
        {
            "id": "run-123",
            "name": "test-run",
            "userId": "user-123",
            "teamId": None,
            "rftClusterId": None,
            "status": "PENDING",
            "rolloutsPerExample": 8,
            "seqLen": 2048,
            "maxSteps": 10,
            "maxTokens": 2048,
            "batchSize": 128,
            "baseModel": "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT",
            "environments": [{"id": "testuser/test-env", "slug": "testuser/test-env"}],
            "runConfig": None,
            "evalConfig": None,
            "valConfig": None,
            "bufferConfig": None,
            "learningRate": None,
            "loraAlpha": None,
            "oversamplingFactor": None,
            "maxAsyncLevel": None,
            "wandbEntity": "test-team",
            "wandbProject": "test-project",
            "wandbRunName": None,
            "startedAt": None,
            "completedAt": None,
            "errorMessage": None,
            "createdAt": now.isoformat(),
            "updatedAt": now.isoformat(),
        }
    )


@pytest.fixture
def rl_cli_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_API_KEY", "test-key")
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")


class TestRLRunWandbSecrets:
    def test_rl_run_accepts_wandb_secret_attached_to_hub_env(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        rl_cli_env: None,
    ) -> None:
        config_path = tmp_path / "rl.toml"
        _write_wandb_config(config_path)

        captured: Dict[str, Any] = {}

        def mock_get(
            self: Any, endpoint: str, params: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            if endpoint == "/environmentshub/testuser/test-env/@latest":
                return {"data": {"id": "env-uuid-12345"}}
            if endpoint == "/environmentshub/env-uuid-12345/secrets":
                return {
                    "data": [
                        {
                            "id": "esecret-1",
                            "name": "WANDB_API_KEY",
                            "source": "global-linked",
                        }
                    ]
                }
            raise AssertionError(f"Unexpected GET {endpoint}")

        def mock_create_run(self: Any, **kwargs: Any) -> RLRun:
            captured.update(kwargs)
            return _fake_run()

        monkeypatch.setattr("prime_cli.client.APIClient.get", mock_get)
        monkeypatch.setattr("prime_cli.commands.rl.RLClient.create_run", mock_create_run)
        monkeypatch.setattr(
            "prime_cli.commands.rl.RLClient.get_environment_status",
            lambda self, owner, name: {"action": {"status": "SUCCESS"}},
        )

        result = runner.invoke(
            app,
            ["rl", "run", str(config_path)],
            env={"COLUMNS": "200", "LINES": "50"},
        )

        assert result.exit_code == 0, result.output
        assert captured["secrets"] is None
        assert "API Key: detected on hub environment secret(s): testuser/test-env" in result.output
        assert "Run created successfully" in result.output

    def test_rl_run_explains_hub_secret_linking_when_wandb_key_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        rl_cli_env: None,
    ) -> None:
        config_path = tmp_path / "rl.toml"
        _write_wandb_config(config_path)

        def mock_get(
            self: Any, endpoint: str, params: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            if endpoint == "/environmentshub/testuser/test-env/@latest":
                return {"data": {"id": "env-uuid-12345"}}
            if endpoint == "/environmentshub/env-uuid-12345/secrets":
                return {"data": []}
            raise AssertionError(f"Unexpected GET {endpoint}")

        monkeypatch.setattr("prime_cli.client.APIClient.get", mock_get)

        result = runner.invoke(
            app,
            ["rl", "run", str(config_path)],
            env={"COLUMNS": "200", "LINES": "50"},
        )

        assert result.exit_code == 1, result.output
        assert "WANDB_API_KEY is required when W&B monitoring is configured" in result.output
        assert "prime env secret list testuser/test-env" in result.output
        assert "prime env secret link <secret-id> owner/env" in result.output
        assert "reads your local shell env, not Prime hub secrets" in result.output
