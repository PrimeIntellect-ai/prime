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
import time
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
    hidden = [
        p.name
        for p in (volume_root / "datasets").iterdir()
        if p.name.startswith(".") and p.name != ".prime-stage-publish.lock"
    ]
    assert hidden == [], f"scratch left behind: {hidden}"
    return manifest


def _leftover_scratch(datasets_root: Path) -> list[str]:
    """Operation scratch that must be cleaned up; the publish lock is
    bookkeeping, not scratch."""
    return [
        p.name
        for p in datasets_root.iterdir()
        if p.name.startswith(".prime-stage-") and p.name != ".prime-stage-publish.lock"
    ]


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


def test_stage_publishes_dataset(tmp_path, monkeypatch, captured_emit) -> None:
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


def test_stage_jsonl_dataset(tmp_path, monkeypatch, captured_emit) -> None:
    fixture = _make_jsonl_fixture(tmp_path / "fixture", rows=5)
    install_fake_hub(monkeypatch, fixture)
    volume_root = tmp_path / "volume"

    rc = _run_stage(volume_root, name="tiny-jsonl")
    assert rc == 0
    _assert_staged(volume_root, "tiny-jsonl")
    (payload,) = captured_emit
    assert payload["configs"]["default"]["splits"]["train"]["rows"] == 5


def test_stage_same_sha_is_idempotent(tmp_path, monkeypatch, captured_emit) -> None:
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


def test_stage_changed_sha_is_a_conflict(tmp_path, monkeypatch, captured_emit) -> None:
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
    assert _leftover_scratch(volume_root / "datasets") == []


def test_preexisting_unmanaged_path_is_a_conflict(tmp_path, monkeypatch) -> None:
    fixture = _make_parquet_fixture(tmp_path / "fixture", rows=2)
    install_fake_hub(monkeypatch, fixture)
    volume_root = tmp_path / "volume"
    manual = volume_root / "datasets" / "tiny-sft"
    _make_parquet_fixture(manual, rows=9)  # no manifest: a manual stage

    rc = _run_stage(volume_root)
    assert rc == 1
    # untouched: same files, no manifest added
    assert not (manual / stage.MANIFEST_NAME).exists()


def test_stage_download_failure_publishes_nothing(tmp_path, monkeypatch) -> None:
    fixture = _make_parquet_fixture(tmp_path / "fixture", rows=2)
    install_fake_hub(monkeypatch, fixture, download_raises=OSError("connection reset"))
    volume_root = tmp_path / "volume"

    rc = _run_stage(volume_root)
    assert rc == 1
    assert not (volume_root / "datasets" / "tiny-sft").exists()
    assert _leftover_scratch(volume_root / "datasets") == []


def test_stage_unsupported_layout_publishes_nothing(tmp_path, monkeypatch) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "README.md").write_text("# no data files\n")
    install_fake_hub(monkeypatch, fixture)
    volume_root = tmp_path / "volume"

    rc = _run_stage(volume_root)
    assert rc == 1
    assert not (volume_root / "datasets" / "tiny-sft").exists()


def test_stage_enospc_hint_resizes_volume(tmp_path, monkeypatch) -> None:
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


def test_stage_download_failure_enospc_suggests_resize(tmp_path, monkeypatch, capsys) -> None:
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


def test_stage_private_uses_token_file_only(tmp_path, monkeypatch, captured_emit) -> None:
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


def test_stage_rejects_no_space_before_download(tmp_path, monkeypatch, capsys) -> None:
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
    tmp_path, monkeypatch, captured_emit
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
        "files": mod._file_inventory(winner),
    }
    (winner / mod.MANIFEST_NAME).write_text(json.dumps(manifest))

    rc = _run_stage(volume_root)
    assert rc == 0
    assert captured_emit[-1]["status"] == "already_staged"
    # the loser's candidate scratch is gone
    assert _leftover_scratch(volume_root / "datasets") == []


def test_publish_conflict_different_source_fails(tmp_path, monkeypatch) -> None:
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
    the other returns already_staged (verifying the winner); the published
    data is the winner's. Works through renameat2 where supported and the
    locked fallback where not (e.g. macOS APFS, and production CephFS)."""
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


def test_unsupported_rename_falls_back_to_locked_rename(tmp_path, monkeypatch) -> None:
    """Filesystems without renameat2 flags (production CephFS returns
    EINVAL) publish via lock + absence check + rename, and still never
    overwrite."""
    fixture = _make_parquet_fixture(tmp_path / "fixture", rows=2)
    install_fake_hub(monkeypatch, fixture)
    volume_root = tmp_path / "volume"

    def unsupported(src, dst):
        raise stage.UnsupportedRenameError("filesystem lacks renameat2 flags")

    monkeypatch.setattr(stage, "_rename_noreplace", unsupported)
    assert _run_stage(volume_root) == 0
    _assert_staged(volume_root, "tiny-sft")
    # the publisher lock is hidden bookkeeping in datasets/
    assert (volume_root / "datasets" / ".prime-stage-publish.lock").exists()


def test_locked_rename_never_overwrites_existing(tmp_path, monkeypatch) -> None:
    fixture = _make_parquet_fixture(tmp_path / "fixture", rows=2)
    install_fake_hub(monkeypatch, fixture)
    volume_root = tmp_path / "volume"
    manual = volume_root / "datasets" / "tiny-sft"
    _make_parquet_fixture(manual, rows=7)  # unmanaged existing destination

    def unsupported(src, dst):
        raise stage.UnsupportedRenameError("filesystem lacks renameat2 flags")

    monkeypatch.setattr(stage, "_rename_noreplace", unsupported)
    assert _run_stage(volume_root) == 1
    # existing data untouched, no scratch left
    assert not (manual / stage.MANIFEST_NAME).exists()
    leftovers = [
        p.name
        for p in (volume_root / "datasets").iterdir()
        if p.name.startswith(".prime-stage-") and "publish" not in p.name
    ]
    assert leftovers == []


def test_stage_failure_never_touches_runs_directory(tmp_path, monkeypatch) -> None:
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


# the genuine shutil.rmtree before any test monkeypatches it
_REAL_RMTREE = shutil.rmtree


def unsupported_rename(src, dst):
    raise stage.UnsupportedRenameError("filesystem lacks renameat2 flags")


def pair(tmp_path):
    """A candidate/publish pair for direct publish_no_replace tests."""
    src, dst = tmp_path / "candidate", tmp_path / "published"
    src.mkdir()
    (src / "data.jsonl").write_text('{"x": 1}')
    return src, dst


# ---------------------------------------------------------------------------
# Roast-driven regressions (t033 probes, converted)
# ---------------------------------------------------------------------------


def test_lock_failure_aborts_publication(tmp_path, monkeypatch) -> None:
    """A filesystem without working advisory locks must not get a silently
    weakened check-then-rename: publication aborts, nothing is published."""
    import fcntl

    src, dst = pair(tmp_path)
    monkeypatch.setattr(stage, "_rename_noreplace", unsupported_rename)

    def fail_lock(fd, operation):
        raise OSError(errno.ENOLCK, "no locks available")

    monkeypatch.setattr(fcntl, "flock", fail_lock)
    with pytest.raises((OSError, stage.StageError)):
        stage.publish_no_replace(src, dst)
    assert not dst.exists()


def test_raced_unmanaged_empty_directory_is_not_replaced(tmp_path, monkeypatch) -> None:
    """An unmanaged writer that does not take the CLI's lock creates an
    empty destination between the check and the rename: the rename must
    fail closed, never occupy the other writer's directory."""
    src, dst = pair(tmp_path)
    original_rename = os.rename

    monkeypatch.setattr(stage, "_rename_noreplace", unsupported_rename)

    def racing_rename(source, destination):
        destination.mkdir()  # a manual writer races in with an empty dir
        original_rename(source, destination)

    monkeypatch.setattr(stage.os, "rename", racing_rename)
    with pytest.raises(FileExistsError):
        stage.publish_no_replace(src, dst)
    assert not (dst / "data.jsonl").exists()
    assert src.is_dir()  # the candidate survives untouched


def test_empty_reservation_marker_is_cleaned_on_failure(tmp_path, monkeypatch) -> None:
    """If publication fails after the empty marker was reserved, only the
    (still empty, operation-owned) marker is removed - os.rmdir refuses
    non-empty directories, so raced-in content survives."""
    src, dst = pair(tmp_path)
    monkeypatch.setattr(stage, "_rename_noreplace", unsupported_rename)

    def foreign_content(source, destination):
        (destination / "foreign.txt").write_text("not ours")
        raise OSError(errno.ENOTEMPTY, "not empty")

    monkeypatch.setattr(stage.os, "rename", foreign_content)
    with pytest.raises(FileExistsError):
        stage.publish_no_replace(src, dst)
    # the marker now holds foreign content and was left alone
    assert (dst / "foreign.txt").exists()


def test_eperm_rename_is_not_evaded_by_fallback(tmp_path, monkeypatch) -> None:
    """A denied operation (EPERM) must fail the stage, not silently select
    the weaker publication path."""
    fixture = _make_jsonl_fixture(tmp_path / "fixture")
    install_fake_hub(monkeypatch, fixture)
    volume_root = tmp_path / "volume"

    def denied(src, dst):
        raise OSError(errno.EPERM, "operation not permitted")

    monkeypatch.setattr(stage, "_rename_noreplace", denied)
    assert _run_stage(volume_root) == 1
    assert not (volume_root / "datasets" / "tiny-sft").exists()


def test_datasets_root_symlink_rejected(tmp_path, monkeypatch) -> None:
    """datasets/ -> runs/ would redirect scratch and published data into
    run outputs; staging must refuse before any write."""
    fixture = _make_jsonl_fixture(tmp_path / "fixture")
    install_fake_hub(monkeypatch, fixture)
    volume_root = tmp_path / "volume"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (volume_root / "datasets").mkdir(parents=True)
    (volume_root / "datasets").rmdir()
    (volume_root / "datasets").symlink_to(elsewhere, target_is_directory=True)

    assert _run_stage(volume_root) == 1
    assert list(elsewhere.iterdir()) == []


def test_final_symlink_destination_rejected(tmp_path, monkeypatch) -> None:
    """datasets/<name> pointing at another directory must never be
    followed, replaced, or adopted as already_staged."""
    fixture = _make_jsonl_fixture(tmp_path / "fixture")
    install_fake_hub(monkeypatch, fixture)
    outside = _make_jsonl_fixture(tmp_path / "outside")
    (outside / stage.MANIFEST_NAME).write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "source": REPO,
                "revision": SHA,
                "files": [{"path": "data.jsonl", "bytes": (outside / "data.jsonl").stat().st_size}],
            }
        )
    )
    volume_root = tmp_path / "volume"
    (volume_root / "datasets").mkdir(parents=True)
    (volume_root / "datasets" / "tiny-sft").symlink_to(outside, target_is_directory=True)

    assert _run_stage(volume_root) == 1
    # the symlink itself and its target are untouched
    assert (volume_root / "datasets" / "tiny-sft").is_symlink()
    assert (outside / "data.jsonl").exists()


def test_missing_manifest_shard_fails_restage(tmp_path, monkeypatch) -> None:
    """A staged dataset with a deleted shard must not be re-certified as
    already_staged (inventory check, real fresh-process verifier)."""
    fixture = _make_jsonl_fixture(tmp_path / "fixture")
    install_fake_hub(monkeypatch, fixture)
    volume_root = tmp_path / "volume"
    final = _make_jsonl_fixture(volume_root / "datasets" / "tiny-sft")
    (final / stage.MANIFEST_NAME).write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "source": REPO,
                "revision": SHA,
                "files": [
                    {"path": "data.jsonl", "bytes": (final / "data.jsonl").stat().st_size},
                    {"path": "missing.jsonl", "bytes": 99},
                ],
            }
        )
    )
    assert _run_stage(volume_root) == 1
    # and a fully consistent manifest re-verifies idempotently
    (final / stage.MANIFEST_NAME).write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "source": REPO,
                "revision": SHA,
                "files": [
                    entry
                    for entry in stage._file_inventory(final)
                    if entry["path"] != stage.MANIFEST_NAME
                ],
            }
        )
    )
    assert _run_stage(volume_root) == 0


def test_restage_detects_unlisted_and_resized_files(tmp_path, monkeypatch) -> None:
    fixture = _make_jsonl_fixture(tmp_path / "fixture")
    install_fake_hub(monkeypatch, fixture)
    volume_root = tmp_path / "volume"

    assert _run_stage(volume_root) == 0
    final = volume_root / "datasets" / "tiny-sft"
    (final / "extra.jsonl").write_text('{"x": 1}')
    assert _run_stage(volume_root) == 1  # unlisted file
    (final / "extra.jsonl").unlink()
    (final / "data.jsonl").write_text("{}")  # resized, still listed
    assert _run_stage(volume_root) == 1


def test_absolute_local_external_reference_rejected(tmp_path) -> None:
    """HF offline mode blocks the network, not other local paths: a README
    data_files entry naming an absolute local file must fail the verifier
    (real fresh-process verification, no mocks)."""
    outside = tmp_path / "outside.jsonl"
    outside.write_text('{"x": "external"}')
    candidate = _make_jsonl_fixture(tmp_path / "candidate")
    readme = "---" + chr(10)
    readme += "configs:" + chr(10)
    readme += "- config_name: default" + chr(10)
    readme += "  data_files:" + chr(10)
    readme += "  - split: train" + chr(10)
    readme += "    path: " + str(outside) + chr(10)
    readme += "---" + chr(10)
    readme += "# references a file outside the snapshot" + chr(10)
    (candidate / "README.md").write_text(readme)
    with pytest.raises(stage.StageError):
        stage._run_verifier(candidate, tmp_path / "cache")


def test_scratch_cleanup_error_fails_after_publication(tmp_path, monkeypatch, capsys) -> None:
    """A post-publication scratch cleanup failure must not be reported as
    success: no staged result line, nonzero exit, explicit residual path."""
    fixture = _make_jsonl_fixture(tmp_path / "fixture")
    install_fake_hub(monkeypatch, fixture)
    volume_root = tmp_path / "volume"
    real_rmtree = _REAL_RMTREE

    def rmtree(path, *args, **kwargs):
        if str(path).endswith("-cache") and Path(path).exists():
            if kwargs.get("ignore_errors"):
                return
            raise PermissionError("simulated cache cleanup error")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(stage.shutil, "rmtree", rmtree)
    rc = _run_stage(volume_root)
    assert rc == 1
    err = capsys.readouterr().err
    assert "scratch cleanup incomplete" in err
    assert "published" in err  # the rename happened; the user is told
    out = capsys.readouterr().out
    assert stage.RESULT_MARKER not in out  # no success result line
    # the committed dataset stays (never delete committed data)
    assert (volume_root / "datasets" / "tiny-sft" / "data.jsonl").exists()


def test_sigterm_mid_download_cleans_scratch(tmp_path) -> None:
    """SIGTERM (how kubectl deletes the pod) must unwind through the
    cleanup path: owned scratch removed, nonzero exit."""
    import signal
    import subprocess
    import sys as _sys

    volume_root = tmp_path / "volume"
    import prime_cli

    src_root = Path(prime_cli.__file__).parent
    driver = tmp_path / "sigterm_driver.py"
    driver_lines = [
        "import sys, time, types",
        "sys.path.insert(0, " + repr(str(src_root)) + ")",
        "from prime_cli.commands import volumes_stage_script as stage",
        "class Sib:",
        "    def __init__(self, size):",
        "        self.size = size",
        "class Info:",
        "    sha = " + repr(SHA),
        "    siblings = [Sib(1000)]",
        "class Api:",
        "    def __init__(self, token=None):",
        "        pass",
        "    def dataset_info(self, repo_id, revision=None):",
        "        return Info()",
        "def slow_snapshot_download(**kwargs):",
        "    from pathlib import Path",
        "    dst = Path(kwargs['local_dir'])",
        "    (dst / 'data.jsonl').write_text('{x: 1}')",
        "    (dst / 'download.incomplete').write_text('halfway')",
        "    time.sleep(60)",
        "hub = types.ModuleType('huggingface_hub')",
        "hub.HfApi = Api",
        "hub.snapshot_download = slow_snapshot_download",
        "sys.modules['huggingface_hub'] = hub",
        "rc = stage.main([",
        "    'stage', '--source', " + repr(REPO) + ", '--revision', 'main',",
        "    '--dataset-name', 'tiny-sft', '--volume-root', " + repr(str(volume_root)) + ",",
        "    '--operation-id', 'sigterm1',",
        "])",
        "sys.exit(rc)",
    ]
    driver.write_text(chr(10).join(driver_lines) + chr(10))
    proc = subprocess.Popen([_sys.executable, str(driver)], cwd=str(tmp_path))
    scratch = volume_root / "datasets" / ".prime-stage-sigterm1"
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not (scratch / "download.incomplete").exists():
        if proc.poll() is not None:
            raise AssertionError(f"driver exited early with {proc.returncode}")
        time.sleep(0.05)
    assert (scratch / "download.incomplete").exists(), "download never started"
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=30)
    assert proc.returncode != 0, proc.returncode
    # owned scratch is gone; nothing was published
    assert not scratch.exists()
    assert not (volume_root / "datasets" / "tiny-sft").exists()
