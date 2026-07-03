import json

import prime_cli.commands.images_bulk as images_bulk
import prime_cli.commands.images_transfer_bulk as images_transfer_bulk
import pytest
from prime_cli.commands.images_transfer_bulk import derive_transfer_destination
from prime_cli.main import app
from prime_sandboxes import APIError
from typer.testing import CliRunner

runner = CliRunner()

TEST_ENV = {
    "COLUMNS": "200",
    "LINES": "50",
    "PRIME_DISABLE_VERSION_CHECK": "1",
    "PRIME_TEAM_ID": "",
    "HF_TOKEN": "",
}


def _write_manifest(path, entries):
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")


def _make_harbor_task(root, dirname, *, docker_image=None, with_dockerfile=False):
    task = root / dirname
    env = task / "environment"
    env.mkdir(parents=True)
    if with_dockerfile:
        (env / "Dockerfile").write_text("FROM busybox\n")
    toml_lines = ['version = "1.0"', "", "[task]", f'name = "{dirname}"']
    if docker_image:
        toml_lines += ["", "[environment]", f'docker_image = "{docker_image}"']
    (task / "task.toml").write_text("\n".join(toml_lines) + "\n")
    return task


class FakeTransferAPI:
    """Scriptable stand-in for APIClient, shared across one test invocation."""

    def __init__(self):
        self.calls = []
        self.payloads = []
        self.build_counter = 0
        # build_id -> list of statuses returned by successive GET polls;
        # the last status repeats once the list is exhausted.
        self.poll_scripts = {}
        self.default_poll = ["COMPLETED"]
        # Exceptions raised (FIFO) by POST /images/build before any transfer is queued.
        self.build_error_queue = []
        # Respond with the per-source results shape instead of a top-level build_id.
        self.respond_bulk_shape = False
        self.bulk_entry_error = None
        self.bulk_results_count = 1

    def request(self, method, path, json=None, params=None):
        self.calls.append((method, path))
        if method == "POST" and path == "/images/build":
            if self.build_error_queue:
                raise self.build_error_queue.pop(0)
            assert "source_image" in json
            if self.respond_bulk_shape and self.bulk_entry_error is not None:
                entry = {
                    "sourceImage": json["source_image"],
                    "success": False,
                    "error": self.bulk_entry_error,
                    "retryable": False,
                }
                return {"results": [entry], "failed": [entry]}
            self.payloads.append(json)
            self.build_counter += 1
            build_id = f"build-{self.build_counter}"
            if self.respond_bulk_shape:
                entry = {
                    "sourceImage": json["source_image"],
                    "success": True,
                    "buildId": build_id,
                    "fullImagePath": f"user/{json.get('image_name') or 'derived'}",
                }
                return {"results": [entry] * self.bulk_results_count, "failed": []}
            return {
                "build_id": build_id,
                "fullImagePath": f"user/{json.get('image_name') or 'derived'}",
            }
        if method == "GET" and path.startswith("/images/build/"):
            build_id = path.rsplit("/", 1)[1]
            script = self.poll_scripts.setdefault(build_id, list(self.default_poll))
            return {"status": script.pop(0) if len(script) > 1 else script[0]}
        raise AssertionError(f"Unexpected request: {method} {path}")

    def post_build_count(self):
        return len([c for c in self.calls if c == ("POST", "/images/build")])


@pytest.fixture
def fake_api(monkeypatch):
    api = FakeTransferAPI()
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    monkeypatch.setattr(images_transfer_bulk, "APIClient", lambda: api)
    monkeypatch.setattr(images_bulk, "POLL_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(images_transfer_bulk, "TRANSFER_RATE_LIMIT_PAUSE_SECONDS", 0.0)
    return api


def _fake_hf(monkeypatch, *, info, pages, total, rate_limit_first_rows=False):
    """Serve datasets-server /info and /rows from canned data.

    ``pages`` maps a row offset to the list of row values (the image column)
    served at that offset; None values become rows missing the column. With
    ``rate_limit_first_rows`` the first /rows request gets a 429 response.
    """
    calls = []
    state = {"rate_limited": rate_limit_first_rows}

    class FakeResponse:
        def __init__(self, payload, status_code=200, headers=None):
            self.status_code = status_code
            self._payload = payload
            self.text = json.dumps(payload)
            self.headers = headers or {}

        def json(self):
            return self._payload

    def fake_get(url, params=None, headers=None, timeout=None):
        params = dict(params or {})
        calls.append((url, params))
        if url.endswith("/info"):
            return FakeResponse(info)
        if url.endswith("/rows"):
            if state["rate_limited"]:
                state["rate_limited"] = False
                return FakeResponse({}, status_code=429, headers={"retry-after": "0"})
            offset = params["offset"]
            values = pages.get(offset, [])
            rows = [
                {
                    "row_idx": offset + i,
                    "row": {} if value is None else {"docker_image": value},
                }
                for i, value in enumerate(values)
            ]
            return FakeResponse({"rows": rows, "num_rows_total": total})
        raise AssertionError(f"Unexpected HF request: {url}")

    monkeypatch.setattr(images_transfer_bulk.httpx, "get", fake_get)
    return calls


HF_INFO_ONE_CONFIG = {
    "dataset_info": {
        "default": {
            "features": {
                "docker_image": {"dtype": "string", "_type": "Value"},
                "prompt": {"dtype": "string", "_type": "Value"},
                "picture": {"_type": "Image"},
            },
            "splits": {"train": {"name": "train"}},
        }
    },
    "partial": False,
}


# ---------------------------------------------------------------------------
# Destination derivation
# ---------------------------------------------------------------------------


def test_derive_destination_matches_server_rules():
    assert derive_transfer_destination("ubuntu") == ("ubuntu", "latest")
    assert derive_transfer_destination("docker.io/library/ubuntu:22.04") == ("ubuntu", "22.04")
    assert derive_transfer_destination("ghcr.io/Org/My-App:v1") == ("my-app", "v1")
    assert derive_transfer_destination("quay.io/org/app@sha256:" + "a" * 64) == (
        "app",
        "sha256-" + "a" * 16,
    )
    with pytest.raises(ValueError):
        derive_transfer_destination("")
    with pytest.raises(ValueError):
        derive_transfer_destination("app@sha512:abc")
    # Comma-separated refs (the push --source-image ad-hoc form) are one-per-entry here.
    with pytest.raises(ValueError, match="commas"):
        derive_transfer_destination("a/app:v1,b/app:v2")


# ---------------------------------------------------------------------------
# Manifest mode
# ---------------------------------------------------------------------------


def test_manifest_happy_path(tmp_path, fake_api):
    manifest = tmp_path / "transfers.jsonl"
    _write_manifest(
        manifest,
        [
            {"source": "docker.io/org/app:v1"},
            {"source": "ghcr.io/org/other:v2", "image": "renamed:v9"},
        ],
    )
    result = runner.invoke(
        app, ["images", "transfer-bulk", "--manifest", str(manifest)], env=TEST_ENV
    )
    assert result.exit_code == 0, result.output
    assert "All images transferred successfully" in result.output
    assert fake_api.payloads[0]["source_image"] == "docker.io/org/app:v1"
    assert "image_name" not in fake_api.payloads[0]
    assert fake_api.payloads[1]["image_name"] == "renamed"
    assert fake_api.payloads[1]["image_tag"] == "v9"
    assert all(p["platform"] == "linux/amd64" for p in fake_api.payloads)


def test_manifest_validation_fails_before_any_submission(tmp_path, fake_api):
    manifest = tmp_path / "transfers.jsonl"
    manifest.write_text(
        "\n".join(
            [
                "not json",
                json.dumps({"source": "org/app:v1", "context": "./x"}),
                json.dumps({"image": "no-source:v1"}),
                # Same derived destination app:v2 from two different namespaces.
                json.dumps({"source": "a/app:v2"}),
                json.dumps({"source": "b/app:v2"}),
            ]
        )
        + "\n"
    )
    result = runner.invoke(
        app, ["images", "transfer-bulk", "--manifest", str(manifest)], env=TEST_ENV
    )
    assert result.exit_code == 1
    assert "invalid JSON" in result.output
    assert "unknown key(s) context" in result.output
    assert "'source' is required" in result.output
    assert "duplicate image reference 'app:v2'" in result.output
    assert fake_api.post_build_count() == 0


def test_manifest_destination_override_resolves_duplicates(tmp_path, fake_api):
    manifest = tmp_path / "transfers.jsonl"
    _write_manifest(
        manifest,
        [
            {"source": "a/app:v2"},
            {"source": "b/app:v2", "image": "app-b:v2"},
        ],
    )
    result = runner.invoke(
        app, ["images", "transfer-bulk", "--manifest", str(manifest)], env=TEST_ENV
    )
    assert result.exit_code == 0, result.output
    assert fake_api.post_build_count() == 2


def test_dry_run_prints_derived_destinations_without_api_calls(tmp_path, fake_api):
    manifest = tmp_path / "transfers.jsonl"
    _write_manifest(manifest, [{"source": "ghcr.io/org/My-App:v1"}])
    result = runner.invoke(
        app,
        ["images", "transfer-bulk", "--manifest", str(manifest), "--dry-run"],
        env=TEST_ENV,
    )
    assert result.exit_code == 0, result.output
    assert "my-app:v1" in result.output
    assert "Dry run only" in result.output
    assert fake_api.calls == []


# ---------------------------------------------------------------------------
# Harbor mode
# ---------------------------------------------------------------------------


def test_harbor_transfers_prebuilt_images_and_skips_buildable_tasks(tmp_path, fake_api):
    root = tmp_path / "tasks"
    _make_harbor_task(root, "task-a", docker_image="ghcr.io/org/img-a:v1")
    _make_harbor_task(root, "task-b", with_dockerfile=True)
    _make_harbor_task(root, "task-c", docker_image="ghcr.io/org/img-a:v1")
    result = runner.invoke(app, ["images", "transfer-bulk", "--harbor", str(root)], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    assert "Skipping task-b" in result.output
    assert "Skipping task-c: same image as task-a" in result.output
    assert fake_api.post_build_count() == 1
    assert fake_api.payloads[0]["source_image"] == "ghcr.io/org/img-a:v1"


def test_harbor_with_no_transferable_tasks_fails(tmp_path, fake_api):
    root = tmp_path / "tasks"
    _make_harbor_task(root, "task-a", with_dockerfile=True)
    result = runner.invoke(app, ["images", "transfer-bulk", "--harbor", str(root)], env=TEST_ENV)
    assert result.exit_code == 1
    assert "no tasks with a prebuilt docker_image" in result.output
    assert fake_api.post_build_count() == 0


# ---------------------------------------------------------------------------
# Hugging Face mode
# ---------------------------------------------------------------------------


def test_hf_pages_rows_and_dedupes(tmp_path, fake_api, monkeypatch):
    hf_calls = _fake_hf(
        monkeypatch,
        info=HF_INFO_ONE_CONFIG,
        pages={
            0: ["org/img-a:v1", "org/img-b:v1", "org/img-a:v1"],
            3: ["org/img-c:v1", None],
        },
        total=5,
    )
    result = runner.invoke(
        app,
        ["images", "transfer-bulk", "--hf", "Org/DataSet", "--column", "docker_image"],
        env=TEST_ENV,
    )
    assert result.exit_code == 0, result.output
    assert "Collapsed 1 duplicate image reference(s)" in result.output
    assert "Skipped 1 row(s) with an empty 'docker_image' value" in result.output
    assert fake_api.post_build_count() == 3
    sources = [p["source_image"] for p in fake_api.payloads]
    assert sources == ["org/img-a:v1", "org/img-b:v1", "org/img-c:v1"]
    row_requests = [params for url, params in hf_calls if url.endswith("/rows")]
    assert [p["offset"] for p in row_requests] == [0, 3]
    assert all(p["dataset"] == "Org/DataSet" for p in row_requests)


def test_hf_rate_limited_request_retries(tmp_path, fake_api, monkeypatch):
    monkeypatch.setattr(images_transfer_bulk, "HF_REQUEST_BACKOFF_SECONDS", 0.0)
    hf_calls = _fake_hf(
        monkeypatch,
        info=HF_INFO_ONE_CONFIG,
        pages={0: ["org/img-a:v1"]},
        total=1,
        rate_limit_first_rows=True,
    )
    result = runner.invoke(
        app,
        ["images", "transfer-bulk", "--hf", "org/ds", "--column", "docker_image"],
        env=TEST_ENV,
    )
    assert result.exit_code == 0, result.output
    assert fake_api.post_build_count() == 1
    # The rate-limited /rows request is retried: info + 429 + 200.
    assert len(hf_calls) == 3


def test_hf_accepts_dataset_url(tmp_path, fake_api, monkeypatch):
    hf_calls = _fake_hf(monkeypatch, info=HF_INFO_ONE_CONFIG, pages={0: ["org/img-a:v1"]}, total=1)
    result = runner.invoke(
        app,
        [
            "images",
            "transfer-bulk",
            "--hf",
            "https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset/viewer/default/train",
            "--column",
            "docker_image",
        ],
        env=TEST_ENV,
    )
    assert result.exit_code == 0, result.output
    assert all(params["dataset"] == "R2E-Gym/R2E-Gym-Subset" for _, params in hf_calls)


def test_hf_multiple_configs_requires_flag(tmp_path, fake_api, monkeypatch):
    info = {
        "dataset_info": {
            "cfg-a": HF_INFO_ONE_CONFIG["dataset_info"]["default"],
            "cfg-b": HF_INFO_ONE_CONFIG["dataset_info"]["default"],
        }
    }
    _fake_hf(monkeypatch, info=info, pages={0: ["org/img-a:v1"]}, total=1)
    result = runner.invoke(app, ["images", "transfer-bulk", "--hf", "org/ds"], env=TEST_ENV)
    assert result.exit_code == 1
    assert "multiple configs" in result.output

    result = runner.invoke(
        app,
        [
            "images",
            "transfer-bulk",
            "--hf",
            "org/ds",
            "--hf-config",
            "cfg-a",
            "--column",
            "docker_image",
        ],
        env=TEST_ENV,
    )
    assert result.exit_code == 0, result.output


def test_hf_unknown_column_and_split_fail(tmp_path, fake_api, monkeypatch):
    _fake_hf(monkeypatch, info=HF_INFO_ONE_CONFIG, pages={0: []}, total=0)
    result = runner.invoke(
        app,
        ["images", "transfer-bulk", "--hf", "org/ds", "--column", "nope"],
        env=TEST_ENV,
    )
    assert result.exit_code == 1
    assert "column 'nope' not found" in result.output

    result = runner.invoke(
        app,
        [
            "images",
            "transfer-bulk",
            "--hf",
            "org/ds",
            "--column",
            "docker_image",
            "--hf-split",
            "test",
        ],
        env=TEST_ENV,
    )
    assert result.exit_code == 1
    assert "split 'test' not found" in result.output


def test_hf_column_is_required(tmp_path, fake_api, monkeypatch):
    _fake_hf(monkeypatch, info=HF_INFO_ONE_CONFIG, pages={0: []}, total=0)
    result = runner.invoke(app, ["images", "transfer-bulk", "--hf", "org/ds"], env=TEST_ENV)
    assert result.exit_code == 1
    assert "--column is required" in result.output
    # The error lists the dataset's columns so the user can pick one.
    assert "docker_image" in result.output
    assert fake_api.post_build_count() == 0


def test_hf_large_dataset_prints_size_note(tmp_path, fake_api, monkeypatch):
    info = json.loads(json.dumps(HF_INFO_ONE_CONFIG))
    info["dataset_info"]["default"]["dataset_size"] = 3_700_000_000
    _fake_hf(monkeypatch, info=info, pages={0: ["org/img-a:v1"]}, total=1)
    result = runner.invoke(
        app,
        ["images", "transfer-bulk", "--hf", "org/ds", "--column", "docker_image"],
        env=TEST_ENV,
    )
    assert result.exit_code == 0, result.output
    assert "may take a few minutes" in result.output


def test_hf_non_string_column_rejected(tmp_path, fake_api, monkeypatch):
    _fake_hf(monkeypatch, info=HF_INFO_ONE_CONFIG, pages={0: []}, total=0)
    result = runner.invoke(
        app,
        ["images", "transfer-bulk", "--hf", "org/ds", "--column", "picture"],
        env=TEST_ENV,
    )
    assert result.exit_code == 1
    assert "not a string column" in result.output
    assert fake_api.post_build_count() == 0


# ---------------------------------------------------------------------------
# Submission behavior
# ---------------------------------------------------------------------------


def test_rate_limited_submit_defers_and_retries(tmp_path, fake_api):
    manifest = tmp_path / "transfers.jsonl"
    _write_manifest(manifest, [{"source": "a/app-a:v1"}, {"source": "b/app-b:v1"}])
    fake_api.build_error_queue.append(
        APIError("HTTP 429: Image transfer rate limit exceeded: 10/10 images")
    )
    result = runner.invoke(
        app, ["images", "transfer-bulk", "--manifest", str(manifest)], env=TEST_ENV
    )
    assert result.exit_code == 0, result.output
    assert "pacing submissions" in result.output
    assert "2/2 transfers completed" in result.output
    # First POST is rejected, the spec is requeued, then both succeed.
    assert fake_api.post_build_count() == 3


def test_bulk_shape_response_is_unwrapped(tmp_path, fake_api):
    fake_api.respond_bulk_shape = True
    manifest = tmp_path / "transfers.jsonl"
    _write_manifest(manifest, [{"source": "a/app-a:v1"}, {"source": "b/app-b:v1"}])
    result = runner.invoke(
        app, ["images", "transfer-bulk", "--manifest", str(manifest)], env=TEST_ENV
    )
    assert result.exit_code == 0, result.output
    assert "All images transferred successfully" in result.output
    # The build ids from the unwrapped entries are what gets polled.
    assert ("GET", "/images/build/build-1") in fake_api.calls
    assert ("GET", "/images/build/build-2") in fake_api.calls


def test_comma_separated_sources_rejected_in_manifest_and_hf(tmp_path, fake_api, monkeypatch):
    manifest = tmp_path / "transfers.jsonl"
    _write_manifest(manifest, [{"source": "a/app:v1,b/app:v2"}])
    result = runner.invoke(
        app, ["images", "transfer-bulk", "--manifest", str(manifest)], env=TEST_ENV
    )
    assert result.exit_code == 1
    assert "commas are not allowed" in result.output
    assert fake_api.post_build_count() == 0

    _fake_hf(
        monkeypatch,
        info=HF_INFO_ONE_CONFIG,
        pages={0: ["org/img-a:v1,org/img-b:v1"]},
        total=1,
    )
    result = runner.invoke(
        app,
        ["images", "transfer-bulk", "--hf", "org/ds", "--column", "docker_image"],
        env=TEST_ENV,
    )
    assert result.exit_code == 1
    assert "row 0" in result.output
    assert "commas are not allowed" in result.output
    assert fake_api.post_build_count() == 0


def test_bulk_shape_multi_entry_response_fails_loudly(tmp_path, fake_api):
    fake_api.respond_bulk_shape = True
    fake_api.bulk_results_count = 2
    manifest = tmp_path / "transfers.jsonl"
    _write_manifest(manifest, [{"source": "a/app-a:v1"}])
    result = runner.invoke(
        app, ["images", "transfer-bulk", "--manifest", str(manifest)], env=TEST_ENV
    )
    assert result.exit_code == 1
    assert "expected one transfer result, got 2" in result.output


def test_bulk_shape_failed_entry_records_submit_failed(tmp_path, fake_api):
    fake_api.respond_bulk_shape = True
    fake_api.bulk_entry_error = "Invalid transfer source: nope"
    manifest = tmp_path / "transfers.jsonl"
    _write_manifest(manifest, [{"source": "a/app-a:v1"}])
    result = runner.invoke(
        app, ["images", "transfer-bulk", "--manifest", str(manifest)], env=TEST_ENV
    )
    assert result.exit_code == 1
    assert "0/1 transfers completed" in result.output
    assert "Invalid transfer source: nope" in result.output


def test_persistent_rate_limit_gives_up_instead_of_pacing_forever(tmp_path, fake_api, monkeypatch):
    monkeypatch.setattr(images_bulk, "MAX_CONSECUTIVE_SUBMIT_DEFERRALS", 2)
    manifest = tmp_path / "transfers.jsonl"
    _write_manifest(manifest, [{"source": "a/app-a:v1"}, {"source": "b/app-b:v1"}])
    fake_api.build_error_queue.extend(
        APIError("HTTP 429: Image transfer rate limit exceeded: 10/10 images") for _ in range(10)
    )
    failures_out = tmp_path / "failures.jsonl"
    result = runner.invoke(
        app,
        [
            "images",
            "transfer-bulk",
            "--manifest",
            str(manifest),
            "--failures-out",
            str(failures_out),
        ],
        env=TEST_ENV,
    )
    assert result.exit_code == 1
    assert "0/2 transfers completed" in result.output
    assert "kept rate-limiting" in result.output
    # Rate-limit failures must not trigger the quota guidance.
    assert "Image quota reached" not in result.output
    # cap+1 attempts for the first spec, then the run stops without touching the second.
    assert fake_api.post_build_count() == 3
    lines = [json.loads(line) for line in failures_out.read_text().splitlines()]
    assert [entry["source"] for entry in lines] == ["a/app-a:v1", "b/app-b:v1"]


def test_quota_429_aborts_and_skips_remaining(tmp_path, fake_api):
    manifest = tmp_path / "transfers.jsonl"
    _write_manifest(
        manifest,
        [{"source": "a/app-a:v1"}, {"source": "b/app-b:v1"}, {"source": "c/app-c:v1"}],
    )
    fake_api.build_error_queue.append(
        APIError("HTTP 429: Image limit exceeded. Personal has 100 images")
    )
    failures_out = tmp_path / "failures.jsonl"
    result = runner.invoke(
        app,
        [
            "images",
            "transfer-bulk",
            "--manifest",
            str(manifest),
            "--failures-out",
            str(failures_out),
        ],
        env=TEST_ENV,
    )
    assert result.exit_code == 1
    assert "0/3 transfers completed" in result.output
    assert "Image quota reached" in result.output
    lines = [json.loads(line) for line in failures_out.read_text().splitlines()]
    assert [entry["source"] for entry in lines] == ["a/app-a:v1", "b/app-b:v1", "c/app-c:v1"]
    assert fake_api.post_build_count() == 1


def test_failed_transfer_writes_rerunnable_failures_manifest(tmp_path, fake_api):
    manifest = tmp_path / "transfers.jsonl"
    _write_manifest(
        manifest,
        [{"source": "a/app-a:v1"}, {"source": "b/app-b:v1", "image": "renamed:v1"}],
    )
    fake_api.poll_scripts["build-2"] = ["PENDING", "FAILED"]
    failures_out = tmp_path / "failures.jsonl"
    result = runner.invoke(
        app,
        [
            "images",
            "transfer-bulk",
            "--manifest",
            str(manifest),
            "--failures-out",
            str(failures_out),
        ],
        env=TEST_ENV,
    )
    assert result.exit_code == 1
    assert "1/2 transfers completed" in result.output
    lines = [json.loads(line) for line in failures_out.read_text().splitlines()]
    assert lines == [{"source": "b/app-b:v1", "platform": "linux/amd64", "image": "renamed:v1"}]
    assert f"transfer-bulk --manifest {failures_out}" in result.output


# ---------------------------------------------------------------------------
# Mode validation
# ---------------------------------------------------------------------------


def test_requires_exactly_one_mode(tmp_path, fake_api):
    result = runner.invoke(app, ["images", "transfer-bulk"], env=TEST_ENV)
    assert result.exit_code == 1
    assert "exactly one of --manifest, --harbor or --hf" in result.output

    manifest = tmp_path / "transfers.jsonl"
    _write_manifest(manifest, [{"source": "a/app:v1"}])
    result = runner.invoke(
        app,
        ["images", "transfer-bulk", "--manifest", str(manifest), "--hf", "org/ds"],
        env=TEST_ENV,
    )
    assert result.exit_code == 1
    assert "exactly one of --manifest, --harbor or --hf" in result.output


def test_hf_flags_rejected_outside_hf_mode(tmp_path, fake_api):
    manifest = tmp_path / "transfers.jsonl"
    _write_manifest(manifest, [{"source": "a/app:v1"}])
    result = runner.invoke(
        app,
        ["images", "transfer-bulk", "--manifest", str(manifest), "--column", "docker_image"],
        env=TEST_ENV,
    )
    assert result.exit_code == 1
    assert "only apply to --hf mode" in result.output
