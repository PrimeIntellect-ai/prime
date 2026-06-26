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


def test_push_platform_image_forces_public_owner_scope(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)

    context_path = tmp_path / "context"
    context_path.mkdir()
    (context_path / "Dockerfile").write_text("FROM ubuntu:22.04\n")

    captured = {}

    class DummyAPIClient:
        def request(self, method, path, json=None, params=None):
            if method == "POST" and path == "/images/build":
                captured["build_payload"] = json
                return {
                    "build_id": "build-123",
                    "upload_url": "https://example.test/upload",
                    "fullImagePath": "ubuntu:22.04",
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
        ["images", "push", "ubuntu:22.04", "--context", "context", "--platform-image"],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    assert captured["build_payload"] == {
        "image_name": "ubuntu",
        "image_tag": "22.04",
        "dockerfile_path": PACKAGED_DOCKERFILE_PATH,
        "platform": "linux/amd64",
        "owner_scope": "platform",
        "visibility": "PUBLIC",
    }
    assert "Building and pushing platform image" in result.output
    assert "Owner:" in result.output
    assert "Platform" in result.output


def test_push_platform_image_source_image_queues_platform_transfer(monkeypatch):
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)
    captured = {}

    class DummyAPIClient:
        def request(self, method, path, json=None, params=None):
            captured["method"] = method
            captured["path"] = path
            captured["json"] = json
            return {
                "build_id": "build-123",
                "buildIds": ["build-123"],
                "upload_url": None,
                "fullImagePath": "ubuntu:22.04",
                "visibility": "PUBLIC",
            }

    def fake_put(*args, **kwargs):
        raise AssertionError("transfer should not upload a build context")

    monkeypatch.setattr("prime_cli.commands.images.APIClient", DummyAPIClient)
    monkeypatch.setattr("prime_cli.commands.images.httpx.put", fake_put)

    result = runner.invoke(
        app,
        ["images", "push", "--source-image", "ubuntu:22.04", "--platform-image"],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    assert captured["method"] == "POST"
    assert captured["path"] == "/images/build"
    assert captured["json"] == {
        "dockerfile_path": "Dockerfile",
        "source_image": "ubuntu:22.04",
        "platform": "linux/amd64",
        "visibility": "PUBLIC",
        "owner_scope": "platform",
    }
    assert "Transferring platform image" in result.output
    assert "Owner:" in result.output
    assert "Platform" in result.output
    assert "Visibility:" in result.output
    assert "PUBLIC" in result.output


def test_push_platform_image_rejects_private(monkeypatch):
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    class DummyAPIClient:
        def request(self, method, path, json=None, params=None):
            raise AssertionError(f"Unexpected request: {method} {path}")

    monkeypatch.setattr("prime_cli.commands.images.APIClient", DummyAPIClient)

    result = runner.invoke(
        app,
        ["images", "push", "ubuntu:22.04", "--platform-image", "--private"],
        env=TEST_ENV,
    )

    assert result.exit_code == 1
    assert "Platform images must be public" in result.output


def test_push_platform_image_rejects_team_context(monkeypatch):
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    class DummyAPIClient:
        def request(self, method, path, json=None, params=None):
            raise AssertionError(f"Unexpected request: {method} {path}")

    monkeypatch.setattr("prime_cli.commands.images.APIClient", DummyAPIClient)

    result = runner.invoke(
        app,
        ["images", "push", "ubuntu:22.04", "--platform-image"],
        env={**TEST_ENV, "PRIME_TEAM_ID": "team-123"},
    )

    assert result.exit_code == 1
    assert "Platform images cannot be pushed in a team context" in result.output


def test_push_image_source_image_queues_transfer_without_upload(monkeypatch):
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)
    captured = {}

    class DummyAPIClient:
        def request(self, method, path, json=None, params=None):
            captured["method"] = method
            captured["path"] = path
            captured["json"] = json
            return {
                "build_id": "build-123",
                "buildIds": ["build-123"],
                "upload_url": None,
                "fullImagePath": "cmk123/ubuntu:22.04",
                "visibility": "PRIVATE",
            }

    def fake_put(*args, **kwargs):
        raise AssertionError("transfer should not upload a build context")

    monkeypatch.setattr("prime_cli.commands.images.APIClient", DummyAPIClient)
    monkeypatch.setattr("prime_cli.commands.images.httpx.put", fake_put)

    result = runner.invoke(
        app,
        ["images", "push", "--source-image", "ubuntu:22.04"],
        env={**TEST_ENV, "PRIME_USER_ID": "cmk123"},
    )

    assert result.exit_code == 0, result.output
    assert captured["method"] == "POST"
    assert captured["path"] == "/images/build"
    assert captured["json"] == {
        "dockerfile_path": "Dockerfile",
        "source_image": "ubuntu:22.04",
        "platform": "linux/amd64",
    }
    assert "Transfer queued" in result.output
    assert "build-123" in result.output


def test_push_image_source_image_with_destination_override(monkeypatch):
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)
    captured = {}

    class DummyAPIClient:
        def request(self, method, path, json=None, params=None):
            captured["json"] = json
            return {
                "build_id": "build-123",
                "buildIds": ["build-123"],
                "fullImagePath": "cmk123/myubuntu:22.04",
                "visibility": "PUBLIC",
            }

    monkeypatch.setattr("prime_cli.commands.images.APIClient", DummyAPIClient)

    result = runner.invoke(
        app,
        ["images", "push", "myubuntu:22.04", "--source-image", "ubuntu:22.04", "--public"],
        env={**TEST_ENV, "PRIME_USER_ID": "cmk123"},
    )

    assert result.exit_code == 0, result.output
    assert captured["json"] == {
        "image_name": "myubuntu",
        "image_tag": "22.04",
        "dockerfile_path": "Dockerfile",
        "source_image": "ubuntu:22.04",
        "platform": "linux/amd64",
        "visibility": "PUBLIC",
    }


def test_push_image_source_image_multi_rejects_destination(monkeypatch):
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    class DummyAPIClient:
        def request(self, method, path, json=None, params=None):
            raise AssertionError(f"Unexpected request: {method} {path}")

    monkeypatch.setattr("prime_cli.commands.images.APIClient", DummyAPIClient)

    result = runner.invoke(
        app,
        [
            "images",
            "push",
            "myubuntu:22.04",
            "--source-image",
            "ubuntu:22.04,ghcr.io/org/app:v1",
        ],
        env=TEST_ENV,
    )

    assert result.exit_code == 1
    assert "single-image transfers" in result.output


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


def test_push_image_source_image_result_shape_uses_full_image_path(monkeypatch):
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)

    class DummyAPIClient:
        def request(self, method, path, json=None, params=None):
            return {
                "results": [
                    {
                        "sourceImage": "ubuntu:jammy",
                        "success": True,
                        "buildId": "buildabc",
                        "fullImagePath": "cmkabc/ubuntu:jammy",
                    }
                ],
                "failed": [],
            }

    monkeypatch.setattr("prime_cli.commands.images.APIClient", DummyAPIClient)

    result = runner.invoke(
        app,
        ["images", "push", "--source-image", "ubuntu:jammy"],
        env={**TEST_ENV, "PRIME_USER_ID": "cmkabc"},
    )

    assert result.exit_code == 0, result.output
    assert "buildabc" in result.output
    assert "cmkabc/ubuntu:jammy" in result.output


def test_push_image_source_image_result_shape_reports_all_failures(monkeypatch):
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)

    class DummyAPIClient:
        def request(self, method, path, json=None, params=None):
            return {
                "results": [
                    {
                        "sourceImage": "missing:notfound",
                        "success": False,
                        "error": "source image not found",
                        "retryable": False,
                    }
                ],
                "failed": [
                    {
                        "sourceImage": "missing:notfound",
                        "success": False,
                        "error": "source image not found",
                        "retryable": False,
                    }
                ],
            }

    monkeypatch.setattr("prime_cli.commands.images.APIClient", DummyAPIClient)

    result = runner.invoke(
        app,
        ["images", "push", "--source-image", "missing:notfound"],
        env={**TEST_ENV, "PRIME_USER_ID": "cmkabc"},
    )

    assert result.exit_code == 1
    assert "Failed to initiate image transfer" in result.output
    assert "missing:notfound: source image not found" in result.output
    assert "Your image transfer is running" not in result.output


def test_push_image_source_image_result_shape_reports_partial_failures(monkeypatch):
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)

    failed = {
        "sourceImage": "missing:notfound",
        "success": False,
        "error": "source image not found",
        "retryable": False,
    }

    class DummyAPIClient:
        def request(self, method, path, json=None, params=None):
            return {
                "results": [
                    {
                        "sourceImage": "ubuntu:jammy",
                        "success": True,
                        "buildId": "buildabc",
                        "fullImagePath": "cmkabc/ubuntu:jammy",
                    },
                    failed,
                ],
                "failed": [failed],
            }

    monkeypatch.setattr("prime_cli.commands.images.APIClient", DummyAPIClient)

    result = runner.invoke(
        app,
        ["images", "push", "--source-image", "ubuntu:jammy,missing:notfound"],
        env={**TEST_ENV, "PRIME_USER_ID": "cmkabc"},
    )

    assert result.exit_code == 1
    assert "buildabc" in result.output
    assert "image transfer" in result.output
    assert "failed" in result.output
    assert "missing:notfound: source image not found" in result.output
