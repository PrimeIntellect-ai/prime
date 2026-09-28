"""Tests for the bundled in-pod staging script
(prime_cli.commands.volumes_stage_script).

Runs the real script phases against tiny local Parquet/JSONL snapshots.
The fake `huggingface_hub` module replaces network download/resolve; the
offline fresh-process verification is real (datasets is installed in the
test environment).
"""

import errno
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
from prime_cli.commands import volumes_stage_script as stage

REPO = "acme/tiny-sft"
SHA = "deadbeefcafe1234"


def _make_parquet_fixture(root: Path, rows: int = 2, split: str = "train") -> Path:
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    import pandas as pd

    pd.DataFrame(
        {"prompt": [f"p{i}" for i in range(rows)], "completion": ["ok"] * rows}
    ).to_parquet(data_dir / f"{split}-00000-of-00001.parquet")
    (root / "README.md").write_text("# tiny fixture\n")
    return root


def _make_jsonl_fixture(root: Path, rows: int = 2) -> Path:
    root.mkdir(parents=True)
    (root / "data.jsonl").write_text(
        "".join(json.dumps({"prompt": f"p{i}", "completion": "ok"}) + "\n" for i in range(rows))
    )
    (root / "README.md").write_text("# tiny jsonl fixture\n")
    return root


class Sibling:
    def __init__(self, size):
        self.size = size


class FakeDatasetInfo:
    sha = SHA
    siblings = [Sibling(1000)]


class FakeHfApi:
    def __init__(self, token=None):
        FakeHfApi.last_token = token

    def dataset_info(self, repo_id, revision=None):
        if getattr(FakeHfApi, "sha", None):
            FakeDatasetInfo.sha = FakeHfApi.sha
        if getattr(FakeHfApi, "fail", None):
            raise FakeHfApi.fail
        return FakeDatasetInfo


def install_fake_hub(monkeypatch, snapshot_from: Path, download_raises: Exception | None = None):
    hub = types.ModuleType("huggingface_hub")

    def snapshot_download(repo_id, repo_type=None, revision=None, local_dir=None, token=None):
        if download_raises:
            raise download_raises
        dst = Path(local_dir)
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(snapshot_from, dst)
        return str(dst)

    hub.HfApi = FakeHfApi
    hub.snapshot_download = snapshot_download
    FakeHfApi.last_token = None
    FakeHfApi.sha = None
    FakeHfApi.fail = None
    FakeDatasetInfo.sha = SHA
    FakeDatasetInfo.siblings = [Sibling(1000)]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return hub


def _run_stage(volume_root: Path, name: str = "tiny-sft", revision: str = "main", token_file=None):
    return stage.main(
        [
            "stage",
            "--source",
            REPO,
            "--revision",
            revision,
            "--dataset-name",
            name,
            "--volume-root",
            str(volume_root),
            "--operation-id",
            "op1",
        ]
        + (["--hf-token-file", str(token_file)] if token_file else [])
    )


def _assert_staged(volume_root: Path, name: str) -> dict:
    final = volume_root / "datasets" / name
    assert final.is_dir()
    manifest = json.loads((final / stage.MANIFEST_NAME).read_text())
    assert manifest["schemaVersion"] == 1
    assert manifest["source"] == REPO
    assert manifest["revision"] == SHA
    assert manifest["requestedRevision"] == "main"
    assert manifest["libraryVersions"]["datasets"]
    data_files = [f["path"] for f in manifest["files"] if f["path"].endswith(stage.DATA_EXTENSIONS)]
    assert data_files, manifest["files"]
    assert any(f["path"] == "README.md" for f in manifest["files"])
    assert (final / "README.md").exists()
    # trainer-readable permissions, no leftovers
    assert (final / "README.md").stat().st_mode & 0o777 == 0o644
    for child in final.rglob("*"):
        if child.is_dir():
            assert child.stat().st_mode & 0o777 == 0o755, child
        else:
            assert child.stat().st_mode & 0o777 == 0o644, child
    assert not (final / ".cache").exists()
    hidden = [p.name for p in (volume_root / "datasets").iterdir() if p.name.startswith(".")]
    assert hidden == [], f"scratch left behind: {hidden}"
    return manifest


def _supports_noreplace(tmp_path: Path) -> bool:
    src = tmp_path / "probe-src"
    dst = tmp_path / "probe-dst"
    src.mkdir()
    try:
        stage._rename_noreplace(src, dst)
    except stage.StageError:
        shutil.rmtree(dst, ignore_errors=True)
        return False
    except OSError:
        shutil.rmtree(dst, ignore_errors=True)
        return False
    shutil.rmtree(dst)
    return True


@pytest.fixture()
def checked_rename(monkeypatch, tmp_path):
    """On filesystems without renameat2(RENAME_NOREPLACE) (macOS APFS),
    fall back to a check-then-rename for single-process publication tests.
    The concurrency test skips there instead of using this shim."""
    if _supports_noreplace(tmp_path):
        yield
        return

    def checked(src, dst):
        if dst.exists():
            raise FileExistsError(errno.EEXIST, "exists", str(src), None, str(dst))
        os.rename(src, dst)

    monkeypatch.setattr(stage, "_rename_noreplace", checked)
    yield


@pytest.fixture()
def captured_emit(monkeypatch):
    lines: list[dict] = []
    real_emit = stage._emit

    def emit(payload):
        lines.append(payload)
        real_emit(payload)

    monkeypatch.setattr(stage, "_emit", emit)
    return lines


# ---------------------------------------------------------------------------
# Layout checks
# ---------------------------------------------------------------------------


def test_layout_accepts_parquet_snapshot(tmp_path) -> None:
    root = _make_parquet_fixture(tmp_path / "candidate")
    assert stage.check_layout(root) == []


def test_layout_accepts_jsonl_snapshot(tmp_path) -> None:
    root = _make_jsonl_fixture(tmp_path / "candidate")
    assert stage.check_layout(root) == []


def test_layout_rejects_missing_data_files(tmp_path) -> None:
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "README.md").write_text("# no data\n")
    problems = stage.check_layout(root)
    assert any("no supported data files" in p for p in problems)


def test_layout_rejects_loading_script(tmp_path) -> None:
    root = _make_parquet_fixture(tmp_path / "candidate")
    (root / "load.py").write_text("import datasets\n")
    problems = stage.check_layout(root)
    assert any("loading script" in p for p in problems)


def test_layout_rejects_save_to_disk_layout(tmp_path) -> None:
    root = _make_parquet_fixture(tmp_path / "candidate")
    (root / "state.json").write_text("{}\n")
    problems = stage.check_layout(root)
    assert any("save_to_disk" in p for p in problems)


def test_layout_rejects_symlink(tmp_path) -> None:
    root = _make_parquet_fixture(tmp_path / "candidate")
    target = tmp_path / "elsewhere.parquet"
    import pandas as pd

    pd.DataFrame({"a": [1]}).to_parquet(target)
    (root / "data" / "linked.parquet").symlink_to(target)
    problems = stage.check_layout(root)
    assert any("symlink" in p for p in problems)


def test_rename_noreplace_conflict_fails_or_skips(tmp_path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    try:
        stage._rename_noreplace(src, dst)
    except stage.StageError:
        pytest.skip("filesystem lacks RENAME_NOREPLACE support")
    except FileExistsError:
        assert not (dst / "src").exists()
    else:
        raise AssertionError("rename onto existing dir must not succeed")


# ---------------------------------------------------------------------------
# verify: fresh process, offline, empty cache
# ---------------------------------------------------------------------------


def test_verify_records_config_splits_and_columns(tmp_path) -> None:
    root = _make_parquet_fixture(tmp_path / "candidate", rows=3)
    payload = stage.verify_dataset(root)
    assert payload["status"] == "verified"
    splits = payload["configs"]["default"]["splits"]
    assert splits["train"]["rows"] == 3
    assert splits["train"]["columns"] == ["prompt", "completion"]


def test_verify_fresh_process_offline_with_empty_cache(tmp_path) -> None:
    """Run the verify subcommand exactly as the pod does: a new
    interpreter, empty cache, HF_HUB_OFFLINE=1, HF_DATASETS_OFFLINE=1, no
    token."""
    root = _make_parquet_fixture(tmp_path / "candidate", rows=2)
    cache = tmp_path / "cache"
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGING_FACE_API_TOKEN")
    }
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HOME": str(cache / "hf"),
            "HF_DATASETS_CACHE": str(cache / "datasets"),
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            stage._self_source(),
            "verify",
            "--dataset",
            str(root),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr
    result_line = [
        line for line in completed.stdout.splitlines() if line.startswith(stage.RESULT_MARKER)
    ]
    assert result_line, completed.stdout
    payload = json.loads(result_line[0][len(stage.RESULT_MARKER) :])
    assert payload["status"] == "verified"
    assert payload["configs"]["default"]["splits"]["train"]["rows"] == 2


def test_verify_corrupt_shard_fails_closed(tmp_path) -> None:
    root = _make_parquet_fixture(tmp_path / "candidate")
    parquet = next((root / "data").glob("*.parquet"))
    parquet.write_bytes(b"not a parquet file")
    rc = stage.main(["verify", "--dataset", str(root)])
    assert rc == 1


def test_verify_rejects_external_http_reference(tmp_path) -> None:
    """A README pointing a config at a remote URL cannot load offline."""
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "README.md").write_text(
        "---\n"
        "configs:\n"
        "- config_name: default\n"
        "  data_files:\n"
        "  - split: train\n"
        "    path: https://example.com/train.parquet\n"
        "---\n"
        "# external reference\n"
    )
    (root / "data" / "ignored.parquet").parent.mkdir(exist_ok=True)
    import pandas as pd

    pd.DataFrame({"a": [1]}).to_parquet(root / "data" / "ignored.parquet")
    with pytest.raises(stage.StageError) as excinfo:
        stage._run_verifier(root, tmp_path / "cache")
    assert "offline verification failed" in str(excinfo.value)


def test_verify_all_empty_splits_fails(tmp_path, monkeypatch) -> None:
    """A dataset whose splits load successfully but hold zero rows must be
    rejected ('at least one nonempty split'). Uses a stub datasets module
    because real zero-row shards are unreadable by the loader anyway."""
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "README.md").write_text("# empty rows\n")

    empty_datasets = types.ModuleType("datasets")

    class EmptyDS:
        column_names = []

        def __len__(self):
            return 0

    class EmptyDict(dict):
        def items(self):
            return [("train", EmptyDS())]

    empty_datasets.get_dataset_config_names = lambda path: ["default"]
    empty_datasets.load_dataset = lambda path, name=None: EmptyDict()
    monkeypatch.setitem(sys.modules, "datasets", empty_datasets)

    with pytest.raises(stage.StageError) as excinfo:
        stage.verify_dataset(root)
    assert "empty" in str(excinfo.value)


def test_verify_zero_row_parquet_fails_closed(tmp_path) -> None:
    """Zero-row parquet shards are unreadable by the loader; the verifier
    must fail closed (nonzero), never pass an unusable dataset."""
    root = _make_parquet_fixture(tmp_path / "candidate", rows=0)
    rc = stage.main(["verify", "--dataset", str(root)])
    assert rc == 1


# ---------------------------------------------------------------------------
# stage: download -> verify -> publish
# ---------------------------------------------------------------------------


def test_stage_publishes_dataset(tmp_path, monkeypatch, checked_rename, captured_emit) -> None:
    fixture = _make_parquet_fixture(tmp_path / "fixture", rows=4)
    install_fake_hub(monkeypatch, fixture)
    volume_root = tmp_path / "volume"

    rc = _run_stage(volume_root)
    assert rc == 0
    _assert_staged(volume_root, "tiny-sft")
    (payload,) = captured_emit
    assert payload["status"] == "staged"
    assert payload["source"] == REPO
    assert payload["revision"] == SHA
    assert payload["datasetName"] == "tiny-sft"
    assert payload["configs"]["default"]["splits"]["train"]["rows"] == 4


def test_stage_jsonl_dataset(tmp_path, monkeypatch, checked_rename, captured_emit) -> None:
    fixture = _make_jsonl_fixture(tmp_path / "fixture", rows=5)
    install_fake_hub(monkeypatch, fixture)
    volume_root = tmp_path / "volume"

    rc = _run_stage(volume_root, name="tiny-jsonl")
    assert rc == 0
    _assert_staged(volume_root, "tiny-jsonl")
    (payload,) = captured_emit
    assert payload["configs"]["default"]["splits"]["train"]["rows"] == 5


def test_stage_same_sha_is_idempotent(tmp_path, monkeypatch, checked_rename, captured_emit) -> None:
    fixture = _make_parquet_fixture(tmp_path / "fixture", rows=2)
    hub = install_fake_hub(monkeypatch, fixture)
    volume_root = tmp_path / "volume"

    assert _run_stage(volume_root) == 0
    assert captured_emit[-1]["status"] == "staged"

    # a second stage of the same sha must NOT download again
    def boom(*args, **kwargs):
        raise AssertionError("snapshot_download must not run on re-stage")

    hub.snapshot_download = boom
    assert _run_stage(volume_root) == 0
    assert captured_emit[-1]["status"] == "already_staged"
    _assert_staged(volume_root, "tiny-sft")


def test_stage_changed_sha_is_a_conflict(
    tmp_path, monkeypatch, checked_rename, captured_emit
) -> None:
    fixture = _make_parquet_fixture(tmp_path / "fixture", rows=2)
    install_fake_hub(monkeypatch, fixture)
    volume_root = tmp_path / "volume"

    assert _run_stage(volume_root) == 0
    staged_manifest = (volume_root / "datasets" / "tiny-sft" / stage.MANIFEST_NAME).read_text()
    data_file = next(
        (volume_root / "datasets" / "tiny-sft" / "data").glob("*.parquet")
    ).read_bytes()

    # upstream moved: same destination, different SHA
    FakeHfApi.sha = "0123456789abcdef"
    rc = _run_stage(volume_root)
    assert rc == 1
    # the last emitted result is still the original staged one; nothing new
    assert captured_emit[-1]["status"] == "staged"
    # existing dataset untouched
    assert (
        volume_root / "datasets" / "tiny-sft" / stage.MANIFEST_NAME
    ).read_text() == staged_manifest
    assert (
        next((volume_root / "datasets" / "tiny-sft" / "data").glob("*.parquet")).read_bytes()
        == data_file
    )
    assert not any(p.name.startswith(".prime-stage-") for p in (volume_root / "datasets").iterdir())


def test_preexisting_unmanaged_path_is_a_conflict(tmp_path, monkeypatch, checked_rename) -> None:
    fixture = _make_parquet_fixture(tmp_path / "fixture", rows=2)
    install_fake_hub(monkeypatch, fixture)
    volume_root = tmp_path / "volume"
    manual = volume_root / "datasets" / "tiny-sft"
    _make_parquet_fixture(manual, rows=9)  # no manifest: a manual stage

    rc = _run_stage(volume_root)
    assert rc == 1
    # untouched: same files, no manifest added
    assert not (manual / stage.MANIFEST_NAME).exists()


def test_stage_download_failure_publishes_nothing(tmp_path, monkeypatch, checked_rename) -> None:
    fixture = _make_parquet_fixture(tmp_path / "fixture", rows=2)
    install_fake_hub(monkeypatch, fixture, download_raises=OSError("connection reset"))
    volume_root = tmp_path / "volume"

    rc = _run_stage(volume_root)
    assert rc == 1
    assert not (volume_root / "datasets" / "tiny-sft").exists()
    assert not any(p.name.startswith(".prime-stage-") for p in (volume_root / "datasets").iterdir())


def test_stage_unsupported_layout_publishes_nothing(tmp_path, monkeypatch, checked_rename) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "README.md").write_text("# no data files\n")
    install_fake_hub(monkeypatch, fixture)
    volume_root = tmp_path / "volume"

    rc = _run_stage(volume_root)
    assert rc == 1
    assert not (volume_root / "datasets" / "tiny-sft").exists()


def test_stage_enospc_hint_resizes_volume(tmp_path, monkeypatch, checked_rename) -> None:
    fixture = _make_parquet_fixture(tmp_path / "fixture", rows=2)
    install_fake_hub(monkeypatch, fixture)
    FakeDatasetInfo.siblings = [Sibling(10**18)]
    try:
        volume_root = tmp_path / "volume"

        rc = _run_stage(volume_root)
        assert rc == 1
        assert not (volume_root / "datasets" / "tiny-sft").exists()
    finally:
        FakeDatasetInfo.siblings = [Sibling(1000)]


def test_stage_download_failure_enospc_suggests_resize(
    tmp_path, monkeypatch, checked_rename, capsys
) -> None:
    fixture = _make_parquet_fixture(tmp_path / "fixture", rows=2)
    install_fake_hub(
        monkeypatch, fixture, download_raises=OSError(errno.ENOSPC, "No space left on device")
    )
    volume_root = tmp_path / "volume"

    rc = _run_stage(volume_root)
    assert rc == 1
    err = capsys.readouterr().err
    assert "No space left" in err
    assert "prime volumes resize" in err
    assert not (volume_root / "datasets" / "tiny-sft").exists()


def test_stage_private_uses_token_file_only(
    tmp_path, monkeypatch, checked_rename, captured_emit
) -> None:
    fixture = _make_parquet_fixture(tmp_path / "fixture", rows=2)
    install_fake_hub(monkeypatch, fixture)
    volume_root = tmp_path / "volume"
    token_file = tmp_path / "token"
    token_file.write_text("hf_token_value_123\n")

    rc = _run_stage(volume_root, token_file=token_file)
    assert rc == 0
    assert FakeHfApi.last_token == "hf_token_value_123"
    # the emitted result carries no credential material
    payload_text = json.dumps(captured_emit)
    assert "hf_token_value_123" not in payload_text


def test_stage_rejects_no_space_before_download(
    tmp_path, monkeypatch, checked_rename, capsys
) -> None:
    fixture = _make_parquet_fixture(tmp_path / "fixture", rows=2)
    install_fake_hub(monkeypatch, fixture)
    FakeDatasetInfo.siblings = [Sibling(2**60)]
    try:
        monkeypatch.setattr(stage.shutil, "disk_usage", lambda p: types.SimpleNamespace(free=1024))
        volume_root = tmp_path / "volume"

        rc = _run_stage(volume_root)
        assert rc == 1
        err = capsys.readouterr().err
        assert "not enough space" in err
        assert "prime volumes resize" in err
    finally:
        FakeDatasetInfo.siblings = [Sibling(1000)]


# ---------------------------------------------------------------------------
# Publication race and no-replace semantics
# ---------------------------------------------------------------------------


def test_publish_conflict_same_sha_returns_already_staged(
    tmp_path, monkeypatch, checked_rename, captured_emit
) -> None:
    """EEXIST during publish with a matching winner manifest ->
    already_staged (concurrent identical publishers are idempotent)."""
    fixture = _make_parquet_fixture(tmp_path / "fixture", rows=2)
    install_fake_hub(monkeypatch, fixture)
    volume_root = tmp_path / "volume"

    def rename_conflict(src, dst):
        raise FileExistsError(errno.EEXIST, "exists", str(src), None, str(dst))

    monkeypatch.setattr(stage, "_rename_noreplace", rename_conflict)

    # pre-create the winner with a matching manifest
    winner = volume_root / "datasets" / "tiny-sft"
    _make_parquet_fixture(winner, rows=2)
    from prime_cli.commands import volumes_stage_script as mod

    manifest = {
        "schemaVersion": mod.MANIFEST_SCHEMA_VERSION,
        "source": REPO,
        "revision": SHA,
        "files": [{"path": "data/train-00000-of-00001.parquet", "bytes": 1}],
    }
    (winner / mod.MANIFEST_NAME).write_text(json.dumps(manifest))

    rc = _run_stage(volume_root)
    assert rc == 0
    assert captured_emit[-1]["status"] == "already_staged"
    # the loser's candidate scratch is gone
    assert not any(p.name.startswith(".prime-stage-") for p in (volume_root / "datasets").iterdir())


def test_publish_conflict_different_source_fails(tmp_path, monkeypatch, checked_rename) -> None:
    fixture = _make_parquet_fixture(tmp_path / "fixture", rows=2)
    install_fake_hub(monkeypatch, fixture)
    volume_root = tmp_path / "volume"

    def rename_conflict(src, dst):
        raise FileExistsError(errno.EEXIST, "exists", str(src), None, str(dst))

    monkeypatch.setattr(stage, "_rename_noreplace", rename_conflict)

    winner = volume_root / "datasets" / "tiny-sft"
    _make_parquet_fixture(winner, rows=2)
    from prime_cli.commands import volumes_stage_script as mod

    manifest = {
        "schemaVersion": mod.MANIFEST_SCHEMA_VERSION,
        "source": "someone/else",
        "revision": "othersha",
        "files": [],
    }
    (winner / mod.MANIFEST_NAME).write_text(json.dumps(manifest))
    winner_data = next((winner / "data").glob("*.parquet")).read_bytes()

    rc = _run_stage(volume_root)
    assert rc == 1
    # winner untouched
    assert next((winner / "data").glob("*.parquet")).read_bytes() == winner_data


def test_concurrent_publication_no_overwrite(tmp_path, monkeypatch, captured_emit) -> None:
    """Two simultaneous publishers of the same repo+sha: exactly one wins,
    the other returns already_staged; the published data is the winner's."""
    if not _supports_noreplace(tmp_path):
        pytest.skip("filesystem lacks renameat2(RENAME_NOREPLACE)")

    import threading

    fixture = _make_parquet_fixture(tmp_path / "fixture", rows=2)
    install_fake_hub(monkeypatch, fixture)
    volume_root = tmp_path / "volume"
    ops = [f"op{i}" for i in range(2)]
    results: list[int] = []
    results_lock = threading.Lock()

    def run(op: str) -> None:
        rc = stage.main(
            [
                "stage",
                "--source",
                REPO,
                "--revision",
                "main",
                "--dataset-name",
                "tiny-sft",
                "--volume-root",
                str(volume_root),
                "--operation-id",
                op,
            ]
        )
        with results_lock:
            results.append(rc)

    threads = [threading.Thread(target=run, args=(op,)) for op in ops]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    assert all(rc == 0 for rc in results), results
    statuses = sorted(payload["status"] for payload in captured_emit)
    assert statuses == ["already_staged", "staged"]
    _assert_staged(volume_root, "tiny-sft")


def test_unsupported_rename_fails_safely(tmp_path, monkeypatch) -> None:
    fixture = _make_parquet_fixture(tmp_path / "fixture", rows=2)
    install_fake_hub(monkeypatch, fixture)
    volume_root = tmp_path / "volume"

    def unsupported(src, dst):
        raise stage.UnsupportedRenameError("no renameat2 here")

    monkeypatch.setattr(stage, "_rename_noreplace", unsupported)
    rc = _run_stage(volume_root)
    assert rc == 1
    # nothing published, nothing left behind
    assert not (volume_root / "datasets" / "tiny-sft").exists()
    assert not any(p.name.startswith(".prime-stage-") for p in (volume_root / "datasets").iterdir())


def test_stage_failure_never_touches_runs_directory(tmp_path, monkeypatch, checked_rename) -> None:
    fixture = _make_parquet_fixture(tmp_path / "fixture", rows=2)
    install_fake_hub(monkeypatch, fixture)
    volume_root = tmp_path / "volume"
    runs = volume_root / "runs" / "run-1"
    runs.mkdir(parents=True)
    (runs / "outputs.txt").write_text("previous run outputs\n")

    FakeHfApi.fail = OSError("hub is down")
    try:
        rc = _run_stage(volume_root)
        assert rc == 1
    finally:
        FakeHfApi.fail = None
    assert (runs / "outputs.txt").read_text() == "previous run outputs\n"


def test_main_verify_subcommand_emits_result(tmp_path, capsys) -> None:
    root = _make_parquet_fixture(tmp_path / "candidate", rows=2)
    rc = stage.main(["verify", "--dataset", str(root)])
    assert rc == 0
    out = capsys.readouterr().out
    line = [line for line in out.splitlines() if line.startswith(stage.RESULT_MARKER)][0]
    assert json.loads(line[len(stage.RESULT_MARKER) :])["status"] == "verified"
