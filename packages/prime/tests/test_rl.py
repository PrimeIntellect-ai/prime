from prime_cli.api.rl import RLClient
from prime_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


class DummyAPIClient:
    def get(self, endpoint, params=None):
        assert endpoint == "/rft/runs"
        return {
            "runs": [
                {
                    "id": "run-123",
                    "name": "monitor-run",
                    "userId": "user-123",
                    "teamId": None,
                    "rftClusterId": None,
                    "status": "COMPLETED",
                    "rolloutsPerExample": 1,
                    "seqLen": 0,
                    "maxSteps": 10,
                    "maxTokens": None,
                    "batchSize": 32,
                    "baseModel": "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT",
                    "environments": [{"id": "reverse-text"}],
                    "runConfig": None,
                    "evalConfig": None,
                    "valConfig": None,
                    "bufferConfig": None,
                    "learningRate": None,
                    "loraAlpha": None,
                    "oversamplingFactor": None,
                    "maxAsyncLevel": None,
                    "wandbEntity": None,
                    "wandbProject": "platform-e2e",
                    "wandbRunName": None,
                    "startedAt": "2026-03-18T22:36:20.251000",
                    "completedAt": "2026-03-18T22:36:59.216000",
                    "errorMessage": None,
                    "createdAt": "2026-03-18T22:36:20.251000",
                    "updatedAt": "2026-03-18T22:37:02.274000",
                }
            ]
        }


class DummyConfig:
    team_id = None


def test_rl_client_list_runs_allows_null_cluster_id():
    runs = RLClient(DummyAPIClient()).list_runs()

    assert len(runs) == 1
    assert runs[0].id == "run-123"
    assert runs[0].cluster_id is None


def test_rl_list_json_handles_null_cluster_id(monkeypatch):
    monkeypatch.setattr("prime_cli.commands.rl.APIClient", DummyAPIClient)
    monkeypatch.setattr("prime_cli.commands.rl.Config", lambda: DummyConfig())

    result = runner.invoke(
        app,
        ["rl", "list", "--output", "json"],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )

    assert result.exit_code == 0, result.output
    assert '"id": "run-123"' in result.output
    assert '"cluster_id": null' in result.output
