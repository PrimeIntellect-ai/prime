import io
import json
import tarfile
from typing import Any

from prime_cli.commands.images import BATCH_BUILD_ENDPOINT, PACKAGED_DOCKERFILE_PATH
from prime_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()

TEST_ENV = {
    "COLUMNS": "200",
    "LINES": "50",
    "PRIME_DISABLE_VERSION_CHECK": "1",
}


def test_push_image_defaults_dockerfile_to_context(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    context_path = tmp_path / "context"
    context_path.mkdir()
    (context_path / "Dockerfile").write_text("FROM busybox\nCOPY run-cowsay.sh /run-cowsay.sh\n")
    (context_path / "run-cowsay.sh").write_text("#!/bin/sh\necho hi\n")

    captured = {}

    class DummyAPIClient:
        def request(self, method, path, json=None, params=None):
            if method == "POST" and path == "/images/build":
                captured["build_payload"] = json
                return {
                    "build_id": "build-123",
                    "upload_url": "https://example.test/upload",
                    "fullImagePath": "rehl:latest",
                }

            if method == "POST" and path == "/images/build/build-123/start":
                return {}

            raise AssertionError(f"Unexpected request: {method} {path}")

    class DummyUploadResponse:
        def raise_for_status(self):
            return None

    def fake_put(url, content, headers, timeout):
        captured["tar_bytes"] = content.read()
        return DummyUploadResponse()

    monkeypatch.setattr("prime_cli.commands.images.APIClient", DummyAPIClient)
    monkeypatch.setattr("prime_cli.commands.images.httpx.put", fake_put)

    result = runner.invoke(
        app,
        ["images", "push", "rehl:latest", "--context", "context"],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    assert captured["build_payload"]["dockerfile_path"] == PACKAGED_DOCKERFILE_PATH
    assert "visibility" not in captured["build_payload"]
    assert "PRIVATE for new images" in result.output
    assert "existing tags keep their current visibility" in result.output

    with tarfile.open(fileobj=io.BytesIO(captured["tar_bytes"]), mode="r:gz") as tar:
        names = set(tar.getnames())
        assert "./run-cowsay.sh" in names
        assert PACKAGED_DOCKERFILE_PATH in names
        dockerfile_member = tar.extractfile(PACKAGED_DOCKERFILE_PATH)
        assert dockerfile_member is not None
        assert dockerfile_member.read().decode() == (context_path / "Dockerfile").read_text()


def test_push_image_public_sends_visibility(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    context_path = tmp_path / "context"
    context_path.mkdir()
    (context_path / "Dockerfile").write_text("FROM busybox\n")

    captured = {}

    class DummyAPIClient:
        def request(self, method, path, json=None, params=None):
            if method == "POST" and path == "/images/build":
                captured["build_payload"] = json
                return {
                    "build_id": "build-123",
                    "upload_url": "https://example.test/upload",
                    "fullImagePath": "rehl:latest",
                }

            if method == "POST" and path == "/images/build/build-123/start":
                return {}

            raise AssertionError(f"Unexpected request: {method} {path}")

    class DummyUploadResponse:
        def raise_for_status(self):
            return None

    def fake_put(url, content, headers, timeout):
        return DummyUploadResponse()

    monkeypatch.setattr("prime_cli.commands.images.APIClient", DummyAPIClient)
    monkeypatch.setattr("prime_cli.commands.images.httpx.put", fake_put)

    result = runner.invoke(
        app,
        ["images", "push", "rehl:latest", "--context", "context", "--public"],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    assert captured["build_payload"]["visibility"] == "PUBLIC"
    assert "Visibility:" in result.output


def test_push_batch_dockerfile_jsonl_sends_future_batch_payload(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    rows_path = tmp_path / "rows.jsonl"
    rows = [
        {"id": "task-a", "dockerfile": "FROM busybox\nRUN echo a\n"},
        {"task_id": "task-b", "dockerfile": "FROM busybox\nRUN echo b\n"},
    ]
    rows_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    captured = {"requests": []}

    class DummyAPIClient:
        def request(self, method, path, json=None, params=None):
            captured["requests"].append((method, path, json, params))
            if method == "POST" and path == BATCH_BUILD_ENDPOINT:
                return {"batch_id": "batch-123"}
            if method == "POST" and path == f"{BATCH_BUILD_ENDPOINT}/batch-123/start":
                return {}
            raise AssertionError(f"Unexpected request: {method} {path}")

    def fake_put(*args, **kwargs):
        raise AssertionError("raw Dockerfile batch mode should not upload contexts")

    monkeypatch.setattr("prime_cli.commands.images.APIClient", DummyAPIClient)
    monkeypatch.setattr("prime_cli.commands.images.httpx.put", fake_put)

    result = runner.invoke(
        app,
        ["images", "push-batch", "rows.jsonl", "--image-name", "cligym", "--public"],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    create_request = captured["requests"][0]
    assert create_request[0] == "POST"
    assert create_request[1] == BATCH_BUILD_ENDPOINT
    payload = create_request[2]
    assert payload["image_name"] == "cligym"
    assert payload["mode"] == "dockerfile"
    assert payload["visibility"] == "PUBLIC"
    assert payload["items"][0]["id"] == "task-a"
    assert payload["items"][0]["dockerfile"] == rows[0]["dockerfile"]
    assert payload["items"][0]["image_tag"].startswith("0001-")
    assert payload["items"][1]["id"] == "task-b"
    assert payload["items"][1]["image_tag"].startswith("0002-")
    assert "context_archive" not in payload["items"][0]
    assert captured["requests"][1] == (
        "POST",
        f"{BATCH_BUILD_ENDPOINT}/batch-123/start",
        {"contexts_uploaded": False},
        None,
    )
    assert "task-a -> cligym:" in result.output
    assert "Image build batch initiated successfully" in result.output


def test_push_batch_dry_run_writes_manifest_without_api_call(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text(json.dumps({"id": "task-a", "dockerfile": "FROM busybox\n"}) + "\n")
    manifest_path = tmp_path / "manifest.jsonl"

    class DummyAPIClient:
        def request(self, method, path, json=None, params=None):
            raise AssertionError(f"Unexpected request: {method} {path}")

    monkeypatch.setattr("prime_cli.commands.images.APIClient", DummyAPIClient)

    result = runner.invoke(
        app,
        [
            "images",
            "push-batch",
            "rows.jsonl",
            "--image-name",
            "cligym",
            "--manifest-output",
            str(manifest_path),
            "--dry-run",
        ],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    manifest_rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    assert manifest_rows[0]["id"] == "task-a"
    assert manifest_rows[0]["image_name"] == "cligym"
    assert manifest_rows[0]["source_type"] == "dockerfile"
    assert "dockerfile_sha256" in manifest_rows[0]
    assert "Dry run: no backend request was sent" in result.output


def test_push_batch_harbor_packages_environment_context_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    task_dir = tmp_path / "tasks" / "task-a"
    environment_dir = task_dir / "environment"
    environment_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text("name = 'task-a'\n")
    (task_dir / "instruction.md").write_text("Do the task.\n")
    (environment_dir / "Dockerfile").write_text("FROM busybox\nCOPY run.sh /run.sh\n")
    (environment_dir / "run.sh").write_text("#!/bin/sh\necho hi\n")
    (environment_dir / "tests").mkdir()
    (environment_dir / "tests" / "test_hidden.py").write_text("assert False\n")
    (environment_dir / "solution").mkdir()
    (environment_dir / "solution" / "answer.py").write_text("print('secret')\n")

    captured_requests: list[tuple[str, str, Any, Any]] = []
    uploaded_urls: list[str] = []
    uploaded_tar_bytes: list[bytes] = []

    class DummyAPIClient:
        def request(self, method, path, json=None, params=None):
            captured_requests.append((method, path, json, params))
            if method == "POST" and path == BATCH_BUILD_ENDPOINT:
                return {
                    "batch_id": "batch-123",
                    "items": [{"id": "task-a", "upload_url": "https://example.test/task-a"}],
                }
            if method == "POST" and path == f"{BATCH_BUILD_ENDPOINT}/batch-123/start":
                return {}
            raise AssertionError(f"Unexpected request: {method} {path}")

    class DummyUploadResponse:
        def raise_for_status(self):
            return None

    def fake_put(url, content, headers, timeout):
        uploaded_urls.append(url)
        uploaded_tar_bytes.append(content.read())
        return DummyUploadResponse()

    monkeypatch.setattr("prime_cli.commands.images.APIClient", DummyAPIClient)
    monkeypatch.setattr("prime_cli.commands.images.httpx.put", fake_put)

    result = runner.invoke(
        app,
        ["images", "push-batch", "tasks", "--mode", "harbor", "--image-name", "cligym"],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    payload = captured_requests[0][2]
    assert payload is not None
    assert payload["mode"] == "harbor"
    assert payload["items"][0]["id"] == "task-a"
    assert payload["items"][0]["dockerfile_path"] == PACKAGED_DOCKERFILE_PATH
    assert payload["items"][0]["context_archive"]["size_bytes"] > 0
    assert uploaded_urls == ["https://example.test/task-a"]

    with tarfile.open(fileobj=io.BytesIO(uploaded_tar_bytes[0]), mode="r:gz") as tar:
        names = set(tar.getnames())
        assert "./run.sh" in names
        assert "./tests/test_hidden.py" not in names
        assert "./solution/answer.py" not in names
        assert PACKAGED_DOCKERFILE_PATH in names


def test_push_batch_harbor_rejects_compose_tasks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    task_dir = tmp_path / "tasks" / "task-a"
    environment_dir = task_dir / "environment"
    environment_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text("name = 'task-a'\n")
    (task_dir / "instruction.md").write_text("Do the task.\n")
    (environment_dir / "Dockerfile").write_text("FROM busybox\n")
    (environment_dir / "docker-compose.yaml").write_text("services: {}\n")

    class DummyAPIClient:
        def request(self, method, path, json=None, params=None):
            raise AssertionError(f"Unexpected request: {method} {path}")

    monkeypatch.setattr("prime_cli.commands.images.APIClient", DummyAPIClient)

    result = runner.invoke(
        app,
        ["images", "push-batch", "tasks", "--mode", "harbor", "--image-name", "cligym"],
        env=TEST_ENV,
    )

    assert result.exit_code == 1
    assert "docker-compose tasks are not supported yet" in result.output
    assert "environment/docker-compose.yaml" in result.output


def test_publish_image_calls_visibility_endpoint(monkeypatch):
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    captured = {}

    class DummyAPIClient:
        def request(self, method, path, json=None, params=None):
            captured["method"] = method
            captured["path"] = path
            captured["json"] = json
            return {"success": True, "message": "ok", "visibility": "PUBLIC", "images": []}

    monkeypatch.setattr("prime_cli.commands.images.APIClient", DummyAPIClient)

    result = runner.invoke(
        app,
        ["images", "publish", "team-abc123/rehl:latest"],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    assert captured["method"] == "PATCH"
    assert captured["path"] == "/images/rehl/latest/visibility"
    assert captured["json"] == {"visibility": "PUBLIC", "teamId": "abc123"}


def test_publish_image_accepts_owner_prefixed_personal_ref(monkeypatch):
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    captured = {}

    class DummyAPIClient:
        def request(self, method, path, json=None, params=None):
            captured["method"] = method
            captured["path"] = path
            captured["json"] = json
            return {"success": True, "message": "ok", "visibility": "PUBLIC", "images": []}

    monkeypatch.setattr("prime_cli.commands.images.APIClient", DummyAPIClient)

    result = runner.invoke(
        app,
        ["images", "publish", "cmk123/rehl:latest"],
        env={**TEST_ENV, "PRIME_USER_ID": "cmk123"},
    )

    assert result.exit_code == 0, result.output
    assert captured["method"] == "PATCH"
    assert captured["path"] == "/images/rehl/latest/visibility"
    assert captured["json"] == {"visibility": "PUBLIC"}


def test_publish_image_rejects_other_user_prefixed_personal_ref(monkeypatch):
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    class DummyAPIClient:
        def request(self, method, path, json=None, params=None):
            raise AssertionError(f"Unexpected request: {method} {path}")

    monkeypatch.setattr("prime_cli.commands.images.APIClient", DummyAPIClient)

    result = runner.invoke(
        app,
        ["images", "publish", "other-user/rehl:latest"],
        env={**TEST_ENV, "PRIME_USER_ID": "cmk123"},
    )

    assert result.exit_code == 1
    assert "Unrecognized image namespace 'other-user'" in result.output


def test_delete_image_accepts_owner_prefixed_personal_ref(monkeypatch):
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    captured = {}

    class DummyAPIClient:
        def request(self, method, path, json=None, params=None):
            captured["method"] = method
            captured["path"] = path
            captured["params"] = params
            return {"success": True, "message": "ok"}

    monkeypatch.setattr("prime_cli.commands.images.APIClient", DummyAPIClient)

    result = runner.invoke(
        app,
        ["images", "delete", "cmk123/rehl:latest", "--yes"],
        env={**TEST_ENV, "PRIME_USER_ID": "cmk123"},
    )

    assert result.exit_code == 0, result.output
    assert captured["method"] == "DELETE"
    assert captured["path"] == "/images/rehl/latest"
    assert captured["params"] is None


def test_push_image_accepts_dockerfile_outside_context(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    context_path = tmp_path / "context"
    context_path.mkdir()
    (context_path / "run-cowsay.sh").write_text("#!/bin/sh\necho hi\n")

    dockerfile_path = tmp_path / "dockerfiles" / "Customfile"
    dockerfile_path.parent.mkdir()
    dockerfile_path.write_text("FROM busybox\nCOPY run-cowsay.sh /run-cowsay.sh\n")

    captured = {}

    class DummyAPIClient:
        def request(self, method, path, json=None, params=None):
            if method == "POST" and path == "/images/build":
                captured["build_payload"] = json
                return {
                    "build_id": "build-123",
                    "upload_url": "https://example.test/upload",
                    "fullImagePath": "rehl:latest",
                }

            if method == "POST" and path == "/images/build/build-123/start":
                captured["start_payload"] = json
                return {}

            raise AssertionError(f"Unexpected request: {method} {path}")

    class DummyUploadResponse:
        def raise_for_status(self):
            return None

    def fake_put(url, content, headers, timeout):
        captured["upload_url"] = url
        captured["tar_bytes"] = content.read()
        captured["upload_headers"] = headers
        captured["upload_timeout"] = timeout
        return DummyUploadResponse()

    monkeypatch.setattr("prime_cli.commands.images.APIClient", DummyAPIClient)
    monkeypatch.setattr("prime_cli.commands.images.httpx.put", fake_put)

    result = runner.invoke(
        app,
        [
            "images",
            "push",
            "rehl:latest",
            "--context",
            "context",
            "--dockerfile",
            "dockerfiles/Customfile",
        ],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    assert captured["build_payload"]["dockerfile_path"] == PACKAGED_DOCKERFILE_PATH
    assert captured["start_payload"] == {"context_uploaded": True}

    with tarfile.open(fileobj=io.BytesIO(captured["tar_bytes"]), mode="r:gz") as tar:
        names = set(tar.getnames())
        assert "./run-cowsay.sh" in names
        assert PACKAGED_DOCKERFILE_PATH in names
        dockerfile_member = tar.extractfile(PACKAGED_DOCKERFILE_PATH)
        assert dockerfile_member is not None
        assert dockerfile_member.read().decode() == dockerfile_path.read_text()
