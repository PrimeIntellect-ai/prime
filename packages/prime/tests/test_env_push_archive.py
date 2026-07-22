import os
import tarfile
from pathlib import Path

import pytest
from prime_cli.commands.env import (
    _add_file_to_archive,
    _collect_archive_files,
    _safe_tar_extract,
    compute_content_hash,
)


def test_collect_archive_files_respects_gitignore(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.1.0'\n")
    (tmp_path / "README.md").write_text("demo\n")
    (tmp_path / "app.py").write_text("print('ok')\n")
    (tmp_path / "ignored.py").write_text("print('ignore me')\n")
    (tmp_path / ".gitignore").write_text("ignored.py\nlogs/\n*.tmp\n")

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "keep.txt").write_text("keep\n")
    (data_dir / "drop.tmp").write_text("drop\n")

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "run.txt").write_text("log\n")

    archive_paths = {
        file_path.relative_to(tmp_path).as_posix() for file_path in _collect_archive_files(tmp_path)
    }

    assert archive_paths == {
        "README.md",
        "app.py",
        "data/keep.txt",
        "pyproject.toml",
    }


def test_collect_archive_files_preserves_gitignore_negation(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.1.0'\n")
    (tmp_path / "env.py").write_text("print('ok')\n")
    (tmp_path / ".gitignore").write_text("artifacts/*\n!artifacts/keep.txt\n")

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / "drop.txt").write_text("drop\n")
    (artifacts_dir / "keep.txt").write_text("keep\n")

    archive_paths = {
        file_path.relative_to(tmp_path).as_posix() for file_path in _collect_archive_files(tmp_path)
    }

    assert "artifacts/keep.txt" in archive_paths
    assert "artifacts/drop.txt" not in archive_paths


def test_compute_content_hash_ignores_gitignored_files(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.1.0'\n")
    (tmp_path / "env.py").write_text("print('ok')\n")
    (tmp_path / ".gitignore").write_text("logs/\n")

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    ignored_file = logs_dir / "run.txt"
    ignored_file.write_text("first\n")

    included_dir = tmp_path / "data"
    included_dir.mkdir()
    included_file = included_dir / "payload.txt"
    included_file.write_text("alpha\n")

    first_hash = compute_content_hash(tmp_path)

    ignored_file.write_text("second\n")
    second_hash = compute_content_hash(tmp_path)

    included_file.write_text("beta\n")
    third_hash = compute_content_hash(tmp_path)

    assert second_hash == first_hash
    assert third_hash != second_hash


@pytest.mark.skipif(not hasattr(os, "link"), reason="os.link is unavailable on this platform")
def test_add_file_to_archive_stores_hardlinks_as_regular_files(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.1.0'\n")
    # Nested dir so files are picked up by _collect_archive_files (root *.txt is ignored)
    payload_dir = tmp_path / "payload"
    payload_dir.mkdir()
    original = payload_dir / "data.txt"
    linked = payload_dir / "data-link.txt"
    original.write_text("shared-content\n")
    try:
        os.link(original, linked)
    except OSError as exc:
        pytest.skip(f"hardlinks unavailable: {exc}")

    archive_path = tmp_path / "demo.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        for file_path in _collect_archive_files(tmp_path):
            _add_file_to_archive(tar, file_path, file_path.relative_to(tmp_path).as_posix())

    dest = tmp_path / "extracted"
    dest.mkdir()
    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        assert any(member.name == "payload/data.txt" for member in members)
        assert any(member.name == "payload/data-link.txt" for member in members)
        assert all(not member.islnk() and not member.issym() for member in members)
        _safe_tar_extract(tar, dest)

    assert (dest / "payload" / "data.txt").read_text(encoding="utf-8") == "shared-content\n"
    assert (dest / "payload" / "data-link.txt").read_text(encoding="utf-8") == "shared-content\n"


def test_tar_add_would_record_hardlink_but_helper_does_not(tmp_path: Path) -> None:
    """Document why we cannot use tar.add for push archives."""
    if not hasattr(os, "link"):
        pytest.skip("os.link is unavailable on this platform")

    original = tmp_path / "a.txt"
    linked = tmp_path / "b.txt"
    original.write_text("x\n")
    try:
        os.link(original, linked)
    except OSError as exc:
        pytest.skip(f"hardlinks unavailable: {exc}")

    naive = tmp_path / "naive.tar.gz"
    with tarfile.open(naive, "w:gz") as tar:
        tar.add(original, arcname="a.txt")
        tar.add(linked, arcname="b.txt")
        naive_has_hardlink = any(member.islnk() for member in tar.getmembers())

    if not naive_has_hardlink:
        pytest.skip("this platform/filesystem does not emit hardlinks via tar.add")

    safe = tmp_path / "safe.tar.gz"
    with tarfile.open(safe, "w:gz") as tar:
        _add_file_to_archive(tar, original, "a.txt")
        _add_file_to_archive(tar, linked, "b.txt")
        assert all(not member.islnk() for member in tar.getmembers())
