import json

import prime_cli.commands.images_bulk as images_bulk
import pytest
from prime_cli.main import app
from prime_sandboxes import APIError
from typer.testing import CliRunner

runner = CliRunner()

TEST_ENV = {
    "COLUMNS": "200",
    "LINES": "50",
    "PRIME_DISABLE_VERSION_CHECK": "1",
    "PRIME_TEAM_ID": "",
}


def _make_context(base, name):
    ctx = base / name
    ctx.mkdir(parents=True)
    (ctx / "Dockerfile").write_text("FROM busybox\n")
    return ctx


def _write_manifest(path, entries):
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")


def _make_harbor_task(root, dirname, *, name=None, docker_image=None, with_dockerfile=True):
    task = root / dirname
    env = task / "environment"
    env.mkdir(parents=True)
    if with_dockerfile:
        (env / "Dockerfile").write_text("FROM busybox\n")
    toml_lines = ['version = "1.0"', "", "[task]", f'name = "{name or dirname}"']
    if docker_image:
        toml_lines += ["", "[environment]", f'docker_image = "{docker_image}"']
    (task / "task.toml").write_text("\n".join(toml_lines) + "\n")
    return task


class FakeAPI:
    """Scriptable stand-in for APIClient, shared across one test invocation."""

    def __init__(self):
        self.calls = []
        self.build_counter = 0
        # build_id -> list of statuses returned by successive GET polls;
        # the last status repeats once the list is exhausted.
        self.poll_scripts = {}
        self.default_poll = ["COMPLETED"]
        # Exceptions raised (FIFO) by POST /images/build before any build is created.
        self.build_error_queue = []

    def request(self, method, path, json=None, params=None):
        self.calls.append((method, path))
        if method == "POST" and path == "/images/build":
            if self.build_error_queue:
                raise self.build_error_queue.pop(0)
            self.build_counter += 1
            build_id = f"build-{self.build_counter}"
            return {
                "build_id": build_id,
                "upload_url": f"https://example.test/upload/{build_id}",
                "fullImagePath": f"user/{json['image_name']}:{json['image_tag']}",
            }
        if method == "POST" and path.endswith("/start"):
            return {}
        if method == "GET" and path.startswith("/images/build/"):
            build_id = path.rsplit("/", 1)[1]
            script = self.poll_scripts.setdefault(build_id, list(self.default_poll))
            return {"status": script.pop(0) if len(script) > 1 else script[0]}
        raise AssertionError(f"Unexpected request: {method} {path}")

    def post_build_indices(self):
        return [i for i, c in enumerate(self.calls) if c == ("POST", "/images/build")]


@pytest.fixture
def fake_api(monkeypatch):
    api = FakeAPI()

    class DummyUploadResponse:
        def raise_for_status(self):
            return None

    def fake_put(url, content, headers=None, timeout=None):
        api.calls.append(("PUT", url))
        content.read()
        return DummyUploadResponse()

    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    monkeypatch.setattr(images_bulk, "APIClient", lambda: api)
    monkeypatch.setattr(images_bulk.httpx, "put", fake_put)
    monkeypatch.setattr(images_bulk, "POLL_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(images_bulk, "RATE_LIMIT_BACKOFF_INITIAL_SECONDS", 0.0)
    return api


def test_manifest_happy_path(tmp_path, fake_api, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_context(tmp_path, "a")
    _make_context(tmp_path, "b")
    manifest = tmp_path / "builds.jsonl"
    _write_manifest(
        manifest,
        [
            {"image": "app-a:v1", "context": "a"},
            {"image": "app-b", "context": "b"},  # tag defaults to latest
        ],
    )

    result = runner.invoke(app, ["images", "push-bulk", "--manifest", str(manifest)], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert len(fake_api.post_build_indices()) == 2
    uploads = [c for c in fake_api.calls if c[0] == "PUT"]
    assert len(uploads) == 2
    starts = [c for c in fake_api.calls if c[0] == "POST" and c[1].endswith("/start")]
    assert len(starts) == 2
    assert "2/2 builds completed" in result.output
    assert not (tmp_path / "push-bulk-failures.jsonl").exists()


def test_manifest_relative_paths_resolve_from_manifest_dir(tmp_path, fake_api, monkeypatch):
    # cwd differs from the manifest's directory; ../a only exists relative to
    # the manifest, so cwd-relative resolution would fail validation.
    monkeypatch.chdir(tmp_path)
    _make_context(tmp_path, "a")
    sub = tmp_path / "sub"
    sub.mkdir()
    manifest = sub / "builds.jsonl"
    _write_manifest(manifest, [{"image": "app-a:v1", "context": "../a"}])

    result = runner.invoke(app, ["images", "push-bulk", "--manifest", str(manifest)], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert len(fake_api.post_build_indices()) == 1


def test_manifest_validation_fails_before_any_submission(tmp_path, fake_api, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_context(tmp_path, "a")
    manifest = tmp_path / "builds.jsonl"
    _write_manifest(
        manifest,
        [
            {"image": "app-a:v1", "context": "missing"},
            {"image": "app-b:v1", "context": "a", "dokerfile": "x"},  # typo key
            {"image": "dup:v1", "context": "a"},
            {"image": "dup:v1", "context": "a"},
            {"image": "good:v1", "context": "a"},
            {"image": "bad:bad/tag", "context": "a"},
            {"image": "app-c:v1", "context": "a", "dockerfile": 123},  # non-string type
        ],
    )

    result = runner.invoke(app, ["images", "push-bulk", "--manifest", str(manifest)], env=TEST_ENV)

    assert result.exit_code == 1
    # The valid good:v1 entry must not be submitted while other lines are invalid.
    assert fake_api.calls == []
    assert "builds.jsonl:1" in result.output
    assert "dokerfile" in result.output
    assert "duplicate image reference 'dup:v1'" in result.output
    assert "invalid image tag 'bad/tag'" in result.output
    assert "'dockerfile' must be a string" in result.output


def test_sliding_window_refills_as_builds_finish(tmp_path, fake_api, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("a", "b", "c"):
        _make_context(tmp_path, name)
    manifest = tmp_path / "builds.jsonl"
    _write_manifest(manifest, [{"image": f"app-{n}:v1", "context": n} for n in ("a", "b", "c")])
    fake_api.default_poll = ["BUILDING", "COMPLETED"]

    result = runner.invoke(
        app,
        ["images", "push-bulk", "--manifest", str(manifest), "--concurrency", "1"],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    posts = fake_api.post_build_indices()
    assert len(posts) == 3
    # With a window of 1, the second build may only be submitted after the
    # first reaches a terminal poll status.
    build_1_polls = [
        i for i, c in enumerate(fake_api.calls) if c == ("GET", "/images/build/build-1")
    ]
    assert len(build_1_polls) == 2  # BUILDING, then COMPLETED
    assert posts[1] > build_1_polls[-1]


def test_failed_build_writes_failures_manifest(tmp_path, fake_api, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_context(tmp_path, "a")
    _make_context(tmp_path, "b")
    manifest = tmp_path / "builds.jsonl"
    _write_manifest(
        manifest,
        [
            {"image": "app-a:v1", "context": "a"},
            {"image": "app-b:v1", "context": "b"},
        ],
    )
    fake_api.poll_scripts["build-2"] = ["FAILED"]

    result = runner.invoke(app, ["images", "push-bulk", "--manifest", str(manifest)], env=TEST_ENV)

    assert result.exit_code == 1
    assert "1/2 builds completed" in result.output
    failures_file = tmp_path / "push-bulk-failures.jsonl"
    assert failures_file.exists()
    lines = [json.loads(line) for line in failures_file.read_text().splitlines()]
    assert len(lines) == 1
    assert lines[0]["image"] == "app-b:v1"
    # Paths in the failures manifest are absolute so it re-runs from any cwd.
    assert lines[0]["context"] == str(tmp_path / "b")


def test_rate_limited_submit_retries(tmp_path, fake_api, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_context(tmp_path, "a")
    manifest = tmp_path / "builds.jsonl"
    _write_manifest(manifest, [{"image": "app-a:v1", "context": "a"}])
    fake_api.build_error_queue = [APIError("HTTP 429: Too Many Requests (per-token limit)")]

    result = runner.invoke(app, ["images", "push-bulk", "--manifest", str(manifest)], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert len(fake_api.post_build_indices()) == 2  # first 429s, retry succeeds


def test_quota_429_aborts_and_skips_remaining(tmp_path, fake_api, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("a", "b", "c"):
        _make_context(tmp_path, name)
    manifest = tmp_path / "builds.jsonl"
    _write_manifest(manifest, [{"image": f"app-{n}:v1", "context": n} for n in ("a", "b", "c")])
    fake_api.build_error_queue = [
        APIError("HTTP 429: Image limit exceeded. Personal has 100 images. Maximum allowed is 100.")
    ]

    result = runner.invoke(app, ["images", "push-bulk", "--manifest", str(manifest)], env=TEST_ENV)

    assert result.exit_code == 1
    # No retry and no further submissions after the quota error.
    assert len(fake_api.post_build_indices()) == 1
    assert "quota" in result.output.lower()
    failures_file = tmp_path / "push-bulk-failures.jsonl"
    lines = [json.loads(line) for line in failures_file.read_text().splitlines()]
    assert {line["image"] for line in lines} == {"app-a:v1", "app-b:v1", "app-c:v1"}


def test_requires_exactly_one_mode(tmp_path, fake_api, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["images", "push-bulk"], env=TEST_ENV)
    assert result.exit_code == 1
    assert "exactly one of --manifest or --harbor" in result.output

    manifest = tmp_path / "builds.jsonl"
    manifest.write_text("")
    result = runner.invoke(
        app,
        ["images", "push-bulk", "--manifest", str(manifest), "--harbor", str(tmp_path)],
        env=TEST_ENV,
    )
    assert result.exit_code == 1
    assert "exactly one of --manifest or --harbor" in result.output


def test_tag_flag_rejected_in_manifest_mode(tmp_path, fake_api, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_context(tmp_path, "a")
    manifest = tmp_path / "builds.jsonl"
    _write_manifest(manifest, [{"image": "app-a:v1", "context": "a"}])

    result = runner.invoke(
        app,
        ["images", "push-bulk", "--manifest", str(manifest), "--tag", "v2"],
        env=TEST_ENV,
    )
    assert result.exit_code == 1
    assert "only apply to --harbor" in result.output
    assert fake_api.calls == []


def test_harbor_dry_run_discovery_and_skips(tmp_path, fake_api, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "tasks"
    root.mkdir()
    _make_harbor_task(root, "Hello_World", name="harbor/hello-world")
    _make_harbor_task(root, "prebuilt-task", docker_image="python:3.11")
    _make_harbor_task(root, "compose-only", with_dockerfile=False)
    (root / "not-a-task").mkdir()

    result = runner.invoke(
        app,
        ["images", "push-bulk", "--harbor", str(root), "--tag", "v1", "--dry-run"],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    assert "hello_world:v1" in result.output
    assert "Skipping prebuilt-task" in result.output
    assert "Skipping compose-only" in result.output
    assert "not-a-task" not in result.output
    assert fake_api.calls == []


def test_harbor_name_template_override(tmp_path, fake_api, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "tasks"
    root.mkdir()
    _make_harbor_task(root, "Hello_World", name="harbor/hello-world")

    result = runner.invoke(
        app,
        [
            "images",
            "push-bulk",
            "--harbor",
            str(root),
            "--name-template",
            "swe-{name}",
            "--dry-run",
        ],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    # task.toml name "harbor/hello-world" sanitized: "/" becomes "-"
    assert "swe-harbor-hello-world:latest" in result.output


def test_harbor_bad_name_template(tmp_path, fake_api, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "tasks"
    root.mkdir()
    _make_harbor_task(root, "task-a")

    result = runner.invoke(
        app,
        ["images", "push-bulk", "--harbor", str(root), "--name-template", "{nope}"],
        env=TEST_ENV,
    )
    assert result.exit_code == 1
    assert "invalid --name-template" in result.output


def test_harbor_name_collision(tmp_path, fake_api, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "tasks"
    root.mkdir()
    _make_harbor_task(root, "task-a")
    _make_harbor_task(root, "task a")  # the space sanitizes to "-": same image name

    result = runner.invoke(app, ["images", "push-bulk", "--harbor", str(root)], env=TEST_ENV)

    assert result.exit_code == 1
    assert "duplicate image reference 'task-a:latest'" in result.output
    assert fake_api.calls == []


def test_harbor_root_is_single_task(tmp_path, fake_api, monkeypatch):
    monkeypatch.chdir(tmp_path)
    task = _make_harbor_task(tmp_path, "solo-task")

    result = runner.invoke(app, ["images", "push-bulk", "--harbor", str(task)], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert len(fake_api.post_build_indices()) == 1
    assert "1/1 builds completed" in result.output
