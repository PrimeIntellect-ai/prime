from prime_cli.commands.env import _collect_archive_files, compute_content_hash


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
