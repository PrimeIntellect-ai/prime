import tarfile
from typing import Any

import pytest
from prime_cli.commands import env as env_commands
from prime_cli.commands.env import _resolve_pull_environment_path
from prime_cli.leaves.env.install import Config as EnvInstallConfig
from prime_cli.leaves.env.pull import Config as EnvPullConfig
from prime_cli.utils import env_metadata


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
        base_url = "https://api.test"

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

    def fake_download(details, destination, *, api_key=None, base_url=None):
        captured.update(
            details=details,
            destination=destination,
            api_key=api_key,
            base_url=base_url,
        )
        destination.mkdir(parents=True)
        (destination / "README.md").write_text("# Demo\n", encoding="utf-8")
        return destination

    target = tmp_path / "demo"
    monkeypatch.setattr(env_commands, "APIClient", FakeAPIClient)
    monkeypatch.setattr(env_commands, "download_environment_source", fake_download)

    env_commands.pull(EnvPullConfig(env_id="alice/demo", target=str(target), version="latest"))

    assert (target / "README.md").read_text(encoding="utf-8") == "# Demo\n"
    assert captured["details"]["tracked_package_url"] == "https://example.test/tracked"
    assert captured["destination"] == target
    assert captured["api_key"] == "test-token"
    assert captured["base_url"] == "https://api.test"


def test_prime_owns_hub_reference_parsing():
    assert env_metadata.parse_env_id("alice/my-env@1.2.3") == (
        "alice",
        "my-env",
        "1.2.3",
    )
    with pytest.raises(ValueError, match="Invalid environment ID"):
        env_metadata.parse_env_id("my-env")


def test_hub_index_install_targets_workspace_python(monkeypatch):
    calls = []
    monkeypatch.setattr(
        env_metadata.subprocess, "run", lambda command, check: calls.append(command)
    )

    module = env_metadata.install_environment_from_hub(
        "alice/my-env@1.2.3",
        {
            "simple_index_url": "https://hub.example/simple",
            "url_dependencies": ["dep @ https://example.test/dep.whl"],
        },
        python_executable="/workspace/.venv/bin/python",
        prerelease=True,
    )

    assert module == "my_env"
    assert calls == [
        [
            "uv",
            "pip",
            "install",
            "--python",
            "/workspace/.venv/bin/python",
            "-P",
            "my_env",
            "my_env==1.2.3",
            "dep @ https://example.test/dep.whl",
            "--extra-index-url",
            "https://hub.example/simple",
            "--prerelease=allow",
        ]
    ]


def test_env_install_resolves_hub_details_in_prime(monkeypatch):
    class FakeAPIClient:
        api_key = "test-token"
        base_url = "https://api.test"

        def __init__(self, require_auth: bool = False) -> None:
            assert require_auth is False

        def get(self, path: str) -> dict[str, Any]:
            assert path == "/environmentshub/alice/my-env/@1.2.3"
            return {"data": {"simple_index_url": "https://hub.example/simple"}}

    captured = {}

    def fake_install(env_id, details, **kwargs):
        captured.update(env_id=env_id, details=details, **kwargs)
        return "my_env"

    monkeypatch.setattr(env_commands, "APIClient", FakeAPIClient)
    monkeypatch.setattr(env_commands.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(env_commands, "resolve_workspace_python", lambda: "/workspace/python")
    monkeypatch.setattr(env_commands, "install_environment_from_hub", fake_install)

    env_commands.install(EnvInstallConfig(env_ids=["alice/my-env@1.2.3"]))

    assert captured == {
        "env_id": "alice/my-env@1.2.3",
        "details": {"simple_index_url": "https://hub.example/simple"},
        "api_key": "test-token",
        "base_url": "https://api.test",
        "python_executable": "/workspace/python",
        "prerelease": False,
    }


def test_source_download_flattens_archive_wrapper(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text('[project]\nname = "demo"\n')
    (source / "demo.py").write_text("VALUE = 1\n")
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source, arcname="demo-1.0")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def raise_for_status(self):
            return None

        def iter_bytes(self, chunk_size):
            assert chunk_size == 8192
            yield archive.read_bytes()

    monkeypatch.setattr(env_metadata.httpx, "stream", lambda *args, **kwargs: Response())
    destination = env_metadata.download_environment_source(
        {"package_url": "https://storage.example.test/demo.tar.gz"},
        tmp_path / "destination",
    )

    assert (destination / "pyproject.toml").is_file()
    assert (destination / "demo.py").is_file()
    assert not (destination / "demo-1.0").exists()
