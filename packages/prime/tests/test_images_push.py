import io
import tarfile

from prime_cli.commands.images import PACKAGED_DOCKERFILE_PATH
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
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    assert captured["method"] == "PATCH"
    assert captured["path"] == "/images/cmk123/rehl/latest/visibility"
    assert captured["json"] == {"visibility": "PUBLIC"}


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
