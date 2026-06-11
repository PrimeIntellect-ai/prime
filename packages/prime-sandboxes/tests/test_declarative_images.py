import io
import tarfile
from types import SimpleNamespace
from typing import Any

from prime_sandboxes import (
    CreateSandboxRequest,
    Image,
    ImageBuildResult,
    SandboxClient,
)
from prime_sandboxes.images import PACKAGED_DOCKERFILE_PATH, ImageClient


def test_declarative_image_renders_dockerfile() -> None:
    image = (
        Image.debian_slim("3.12")
        .apt_install(["git", "curl"])
        .pip_install(["requests", "pytest"])
        .env({"PRIME_ENV": "test value"})
        .workdir("/home/daytona")
        .cmd(["python", "-m", "pytest"])
    )

    assert image.dockerfile() == (
        "FROM python:3.12-slim\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends "
        "git curl && rm -rf /var/lib/apt/lists/*\n"
        "RUN python -m pip install --no-cache-dir requests pytest\n"
        'ENV PRIME_ENV="test value"\n'
        "WORKDIR /home/daytona\n"
        'CMD ["python", "-m", "pytest"]\n'
    )
    assert image.default_tag().startswith("sha-")


def test_image_client_builds_uploads_and_waits(monkeypatch: Any) -> None:
    image = Image.debian_slim("3.12").pip_install(["requests"]).workdir("/workspace")
    image_tag = image.default_tag()
    captured: dict[str, Any] = {}

    class DummyAPIClient:
        config = SimpleNamespace(team_id=None)

        def __init__(self) -> None:
            self.started = False

        def request(self, method: str, path: str, json: Any = None, params: Any = None) -> dict:
            captured.setdefault("requests", []).append((method, path, json, params))
            if method == "GET" and path == "/images":
                if self.started:
                    return {
                        "data": [
                            {
                                "imageName": "declarative-sandbox",
                                "imageTag": image_tag,
                                "artifactType": "CONTAINER_IMAGE",
                                "status": "COMPLETED",
                                "fullImagePath": f"user/declarative-sandbox:{image_tag}",
                                "pushedAt": "2026-05-24T00:00:00Z",
                            }
                        ]
                    }
                return {"data": []}
            if method == "POST" and path == "/images/build":
                captured["build_payload"] = json
                return {
                    "build_id": "build-123",
                    "upload_url": "https://example.test/upload",
                    "fullImagePath": f"user/declarative-sandbox:{image_tag}",
                }
            if method == "POST" and path == "/images/build/build-123/start":
                self.started = True
                captured["start_payload"] = json
                return {}
            raise AssertionError(f"unexpected request: {method} {path}")

    class DummyUploadResponse:
        def raise_for_status(self) -> None:
            return None

    def fake_put(url: str, content: bytes, headers: dict[str, str], timeout: float) -> Any:
        captured["upload_url"] = url
        captured["archive"] = content
        captured["upload_headers"] = headers
        captured["upload_timeout"] = timeout
        return DummyUploadResponse()

    monkeypatch.setattr("prime_sandboxes.images.httpx.put", fake_put)

    logs: list[str] = []
    result = ImageClient(DummyAPIClient()).build(
        image,
        poll_interval_seconds=0,
        on_build_log=logs.append,
    )

    assert result.image_reference == f"user/declarative-sandbox:{image_tag}"
    assert result.status == "COMPLETED"
    assert result.build_id == "build-123"
    assert captured["build_payload"] == {
        "image_name": "declarative-sandbox",
        "image_tag": image_tag,
        "dockerfile_path": PACKAGED_DOCKERFILE_PATH,
        "platform": "linux/amd64",
    }
    assert captured["start_payload"] == {"context_uploaded": True}
    assert captured["upload_url"] == "https://example.test/upload"
    assert captured["upload_headers"] == {"Content-Type": "application/octet-stream"}
    assert captured["upload_timeout"] == 600.0
    assert "Image ready" in logs[-1]

    with tarfile.open(fileobj=io.BytesIO(captured["archive"]), mode="r:gz") as tar:
        assert PACKAGED_DOCKERFILE_PATH in tar.getnames()
        dockerfile_member = tar.extractfile(PACKAGED_DOCKERFILE_PATH)
        assert dockerfile_member is not None
        assert dockerfile_member.read().decode() == image.dockerfile()


def test_image_client_reuses_existing_completed_image() -> None:
    image = Image.debian_slim("3.12")
    image_tag = image.default_tag()

    class DummyAPIClient:
        config = SimpleNamespace(team_id="team-123")

        def request(self, method: str, path: str, json: Any = None, params: Any = None) -> dict:
            assert method == "GET"
            assert path == "/images"
            assert params == {"limit": "250", "offset": "0", "teamId": "team-123"}
            return {
                "data": [
                    {
                        "imageName": "declarative-sandbox",
                        "imageTag": image_tag,
                        "teamId": "team-123",
                        "artifactType": "CONTAINER_IMAGE",
                        "status": "COMPLETED",
                        "fullImagePath": f"team-team-123/declarative-sandbox:{image_tag}",
                        "pushedAt": "2026-05-24T00:00:00Z",
                    }
                ]
            }

    result = ImageClient(DummyAPIClient()).build(image)

    assert result.reused is True
    assert result.build_id is None
    assert result.image_reference == f"team-team-123/declarative-sandbox:{image_tag}"


def test_sandbox_client_build_image_supports_batch_fanout(
    monkeypatch: Any, tmp_path: Any
) -> None:
    captured: dict[str, Any] = {}

    class DummyImageClient:
        def __init__(self, api_client: Any) -> None:
            captured["image_client_api_client"] = api_client

        def build(self, image: Image, **kwargs: Any) -> ImageBuildResult:
            captured["image"] = image
            captured["build_kwargs"] = kwargs
            return ImageBuildResult(
                image_reference="user/batch-runtime:sha-123",
                image_name="batch-runtime",
                image_tag="sha-123",
                status="COMPLETED",
                build_id="build-123",
            )

    class DummyAPIClient:
        config = SimpleNamespace(team_id=None, config_dir=tmp_path)

    monkeypatch.setattr("prime_sandboxes.sandbox.ImageClient", DummyImageClient)

    image = Image.debian_slim("3.12").pip_install("requests")
    build = SandboxClient(DummyAPIClient()).build_image(
        image,
        image_name="batch-runtime",
        timeout_seconds=0,
        poll_interval_seconds=0,
    )
    requests = [
        CreateSandboxRequest(name=f"batch-{i}", docker_image=build.image_reference)
        for i in range(3)
    ]

    assert build.image_reference == "user/batch-runtime:sha-123"
    assert captured["image"] is image
    assert captured["build_kwargs"]["image_name"] == "batch-runtime"
    assert captured["build_kwargs"]["timeout_seconds"] == 0
    assert [request.docker_image for request in requests] == [
        "user/batch-runtime:sha-123",
        "user/batch-runtime:sha-123",
        "user/batch-runtime:sha-123",
    ]
