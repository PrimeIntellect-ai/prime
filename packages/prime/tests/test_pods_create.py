import json
from typing import Any

import pytest
from prime_cli.api.availability import GPUAvailability
from prime_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()

TEST_ENV = {
    "COLUMNS": "200",
    "LINES": "50",
    "PRIME_DISABLE_VERSION_CHECK": "1",
    "PRIME_TEAM_ID": "",
}


@pytest.fixture
def temp_home(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.fixture
def disable_update_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))


@pytest.fixture
def mock_pod_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    gpu = GPUAvailability.model_validate(
        {
            "cloudId": "cloud-ctx",
            "gpuType": "A100_80GB",
            "socket": "PCIe",
            "provider": "test-provider",
            "dataCenter": "dc-1",
            "country": "US",
            "gpuCount": 1,
            "gpuMemory": 80,
            "disk": {"minCount": 10, "defaultCount": 10, "maxCount": 100, "step": 1},
            "vcpu": {"minCount": 4, "defaultCount": 8, "maxCount": 32, "step": 1},
            "memory": {"minCount": 16, "defaultCount": 16, "maxCount": 128, "step": 1},
            "interconnect": None,
            "stockStatus": "AVAILABLE",
            "security": None,
            "prices": {"onDemand": 1.23, "currency": "USD"},
            "images": ["ubuntu"],
            "isSpot": False,
        }
    )

    monkeypatch.setattr(
        "prime_cli.commands.pods.AvailabilityClient.get",
        lambda self: {"A100_80GB": [gpu]},
    )


class TestPodsCreate:
    def test_create_uses_context_environment_team_id(
        self,
        temp_home: None,
        disable_update_check: None,
        mock_pod_availability: None,
        tmp_path: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config_dir = tmp_path / ".prime"
        config_dir.mkdir()
        (config_dir / "environments").mkdir()
        (config_dir / "config.json").write_text(
            json.dumps(
                {
                    "api_key": "",
                    "team_id": "production-team",
                    "team_name": "Production Team",
                    "team_role": "member",
                    "user_id": "user-prod",
                    "base_url": "https://api.primeintellect.ai",
                    "frontend_url": "https://app.primeintellect.ai",
                    "inference_url": "https://api.pinference.ai/api/v1",
                    "ssh_key_path": str(tmp_path / ".ssh" / "id_rsa"),
                    "current_environment": "production",
                    "share_resources_with_team": False,
                }
            )
        )
        (config_dir / "environments" / "staging.json").write_text(
            json.dumps(
                {
                    "api_key": "",
                    "team_id": "staging-team",
                    "team_name": "Staging Team",
                    "team_role": "admin",
                    "user_id": "user-staging",
                    "base_url": "https://api.primeintellect.ai",
                    "frontend_url": "https://app.primeintellect.ai",
                    "inference_url": "https://api.pinference.ai/api/v1",
                }
            )
        )

        captured: dict[str, Any] = {}

        def mock_create(self: Any, pod_config: dict) -> Any:
            captured["pod_config"] = pod_config
            return type("PodResult", (), {"id": "pod-123"})()

        monkeypatch.setattr("prime_cli.commands.pods.PodsClient.create", mock_create)

        result = runner.invoke(
            app,
            [
                "--context",
                "staging",
                "pods",
                "create",
                "--cloud-id",
                "cloud-ctx",
                "--name",
                "ctx-pod",
                "--disk-size",
                "20",
                "--vcpus",
                "8",
                "--memory",
                "16",
                "--image",
                "ubuntu",
                "--yes",
            ],
            env=TEST_ENV,
        )

        assert result.exit_code == 0, result.output
        assert captured["pod_config"]["team"] == {"teamId": "staging-team"}
        assert captured["pod_config"].get("sharedWithTeam") is None

    def test_create_rejects_conflicting_sharing_flags(
        self,
        temp_home: None,
        disable_update_check: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "prime_cli.commands.pods.AvailabilityClient.get",
            lambda self: pytest.fail("availability should not be fetched for invalid flags"),
        )

        result = runner.invoke(
            app,
            [
                "pods",
                "create",
                "--cloud-id",
                "cloud-ctx",
                "--name",
                "bad-pod",
                "--disk-size",
                "20",
                "--vcpus",
                "8",
                "--memory",
                "16",
                "--image",
                "ubuntu",
                "--team-id",
                "team-123",
                "--share-with-team",
                "--add-members",
                "--yes",
            ],
            env=TEST_ENV,
        )

        assert result.exit_code == 1, result.output
        assert "mutually exclusive" in result.output

    def test_create_add_members_shares_only_with_selected_members(
        self,
        temp_home: None,
        disable_update_check: None,
        mock_pod_availability: None,
        tmp_path: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config_dir = tmp_path / ".prime"
        config_dir.mkdir()
        (config_dir / "environments").mkdir()
        (config_dir / "config.json").write_text(
            json.dumps(
                {
                    "api_key": "",
                    "team_id": None,
                    "team_name": None,
                    "team_role": None,
                    "user_id": "user-1",
                    "base_url": "https://api.primeintellect.ai",
                    "frontend_url": "https://app.primeintellect.ai",
                    "inference_url": "https://api.pinference.ai/api/v1",
                    "ssh_key_path": str(tmp_path / ".ssh" / "id_rsa"),
                    "current_environment": "production",
                    "share_resources_with_team": False,
                }
            )
        )

        monkeypatch.setattr(
            "prime_cli.commands.pods.fetch_team_members",
            lambda _client, _team_id: [
                {
                    "userId": "user-1",
                    "userName": "Current User",
                    "userEmail": "current@example.com",
                    "role": "admin",
                },
                {
                    "userId": "user-2",
                    "userName": "Other User",
                    "userEmail": "other@example.com",
                    "role": "member",
                },
            ],
        )

        captured: dict[str, Any] = {}

        def mock_create(self: Any, pod_config: dict) -> Any:
            captured["pod_config"] = pod_config
            return type("PodResult", (), {"id": "pod-456"})()

        monkeypatch.setattr("prime_cli.commands.pods.PodsClient.create", mock_create)

        result = runner.invoke(
            app,
            [
                "pods",
                "create",
                "--cloud-id",
                "cloud-ctx",
                "--name",
                "member-pod",
                "--disk-size",
                "20",
                "--vcpus",
                "8",
                "--memory",
                "16",
                "--image",
                "ubuntu",
                "--team-id",
                "team-123",
                "--add-members",
                "--yes",
            ],
            input="1\n",
            env=TEST_ENV,
        )

        assert result.exit_code == 0, result.output
        assert captured["pod_config"]["team"] == {"teamId": "team-123"}
        assert captured["pod_config"]["teamMemberIds"] == ["user-2"]
        assert "sharedWithTeam" not in captured["pod_config"]
