import io
import tarfile
from pathlib import Path
from typing import Any

import pytest
import typer
from prime_cli.commands import env as env_commands
from prime_cli.commands.env import _environment_package_download_url, _resolve_pull_environment_path


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


def test_environment_package_download_url_prefers_tracked_url():
    details = {
        "tracked_package_url": "https://example.test/tracked",
        "package_url": "https://example.test/direct",
    }

    assert _environment_package_download_url(details) == "https://example.test/tracked"


def test_environment_package_download_url_falls_back_to_package_url_when_untracked():
    details = {"package_url": "https://example.test/direct"}

    assert _environment_package_download_url(details) == "https://example.test/direct"


def test_pull_prefers_tracked_url_and_follows_redirects(tmp_path, monkeypatch):
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

    class FakeStream:
        def __enter__(self) -> "FakeStream":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_bytes(self, chunk_size: int) -> list[bytes]:
            return [b"archive"]

    class FakeTar:
        def __enter__(self) -> "FakeTar":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def getmembers(self) -> list[Any]:
            # Empty members keep _safe_tar_extract validation satisfied; extractall
            # still writes the demo file as before.
            return []

        def extractall(self, target: Any) -> None:
            (Path(target) / "README.md").write_text("# Demo\n", encoding="utf-8")

    def fake_stream(
        method: str,
        url: str,
        headers: dict[str, str],
        timeout: float,
        follow_redirects: bool,
    ) -> FakeStream:
        assert method == "GET"
        assert url == "https://example.test/tracked"
        assert headers == {"Authorization": "Bearer test-token"}
        assert timeout == 60.0
        assert follow_redirects is True
        return FakeStream()

    target = tmp_path / "demo"
    monkeypatch.setattr(env_commands, "APIClient", FakeAPIClient)
    monkeypatch.setattr(env_commands.httpx, "stream", fake_stream)
    monkeypatch.setattr(env_commands.tarfile, "open", lambda *_args, **_kwargs: FakeTar())

    env_commands.pull("alice/demo", target=str(target), version="latest")

    assert (target / "README.md").read_text(encoding="utf-8") == "# Demo\n"


def _gzip_tar_bytes(member_name: str, *, symlink: bool = False) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        info = tarfile.TarInfo(name=member_name)
        if symlink:
            info.type = tarfile.SYMTYPE
            info.linkname = "/tmp"
            tar.addfile(info)
        else:
            info.size = 0
            tar.addfile(info)
    return buffer.getvalue()


def _stub_pull_download(
    monkeypatch: pytest.MonkeyPatch,
    *,
    archive_bytes: bytes,
    details: dict[str, Any] | None = None,
) -> None:
    payload = details or {
        "id": "env-1",
        "tracked_package_url": "https://example.test/pkg.tar.gz",
        "semantic_version": "0.1.0",
        "metadata": {},
    }

    class FakeAPIClient:
        api_key = "test-token"

        def __init__(self, require_auth: bool = False) -> None:
            assert require_auth is False

        def get(self, path: str) -> dict[str, Any]:
            return {"data": payload}

    class FakeStream:
        def __enter__(self) -> "FakeStream":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_bytes(self, chunk_size: int) -> list[bytes]:
            return [archive_bytes]

    def fake_stream(*_args: Any, **_kwargs: Any) -> FakeStream:
        return FakeStream()

    monkeypatch.setattr(env_commands, "APIClient", FakeAPIClient)
    monkeypatch.setattr(env_commands.httpx, "stream", fake_stream)


def test_pull_rejects_path_traversal_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_pull_download(
        monkeypatch, archive_bytes=_gzip_tar_bytes("../../../etc/malicious")
    )
    target = tmp_path / "safe-target"
    outside = tmp_path / "etc"
    outside.mkdir()

    with pytest.raises(typer.Exit) as exc:
        env_commands.pull("alice/demo", target=str(target), version="latest")

    assert exc.value.exit_code == 1
    assert not any(outside.iterdir())
    assert not (tmp_path / "malicious").exists()


def test_pull_rejects_symlink_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_pull_download(
        monkeypatch, archive_bytes=_gzip_tar_bytes("evil_link", symlink=True)
    )
    target = tmp_path / "safe-target"

    with pytest.raises(typer.Exit) as exc:
        env_commands.pull("alice/demo", target=str(target), version="latest")

    assert exc.value.exit_code == 1
    assert not (target / "evil_link").exists()


def test_pull_extracts_safe_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        data = b"ok\n"
        info = tarfile.TarInfo(name="README.md")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    _stub_pull_download(monkeypatch, archive_bytes=buffer.getvalue())
    target = tmp_path / "safe-target"

    env_commands.pull("alice/demo", target=str(target), version="latest")

    assert (target / "README.md").read_text(encoding="utf-8") == "ok\n"
