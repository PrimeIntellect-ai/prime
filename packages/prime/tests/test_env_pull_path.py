from typing import Any

from prime_cli.commands import env as env_commands
from prime_cli.commands.env import _resolve_pull_environment_path
from prime_cli.leaves.env.pull import Config as EnvPullConfig
from verifiers.utils import install_utils


def test_defaults_to_cwd_when_environments_dir_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    resolved = _resolve_pull_environment_path(target=None, env_name="my-env")

    assert resolved == tmp_path / "my_env"


def test_defaults_to_environments_dir_when_present(tmp_path, monkeypatch):
    (tmp_path / "environments").mkdir()
    monkeypatch.chdir(tmp_path)

    resolved = _resolve_pull_environment_path(target=None, env_name="my-env")

    assert resolved == tmp_path / "environments" / "my_env"


def test_respects_explicit_target_path(tmp_path):
    explicit = tmp_path / "custom-target"

    resolved = _resolve_pull_environment_path(target=str(explicit), env_name="my-env")

    assert resolved == explicit


def test_pull_uses_shared_source_downloader(tmp_path, monkeypatch):
    class FakeAPIClient:
        api_key = "test-token"

        def __init__(self, require_auth: bool = False) -> None:
            assert require_auth is False

        def get(self, path: str) -> dict[str, Any]:
            assert path == "/environmentshub/alice/demo/@latest"
            return {
                "data": {
                    "id": "env-1",
                    "tracked_package_url": "https://example.test/tracked",
                    "package_url": "https://example.test/direct",
                    "semantic_version": "0.1.0",
                    "metadata": {},
                }
            }

    captured = {}

    def fake_download(details, destination, api_key=None):
        captured.update(details=details, destination=destination, api_key=api_key)
        destination.mkdir(parents=True)
        (destination / "README.md").write_text("# Demo\n", encoding="utf-8")
        return destination

    target = tmp_path / "demo"
    monkeypatch.setattr(env_commands, "APIClient", FakeAPIClient)
    monkeypatch.setattr(install_utils, "download_environment_source", fake_download, raising=False)

    env_commands.pull(EnvPullConfig(env_id="alice/demo", target=str(target), version="latest"))

    assert (target / "README.md").read_text(encoding="utf-8") == "# Demo\n"
    assert captured["details"]["tracked_package_url"] == "https://example.test/tracked"
    assert captured["destination"] == target
    assert captured["api_key"] == "test-token"
