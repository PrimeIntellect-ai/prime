from typing import Any

from prime_sandboxes import (
    APIClient,
    BuildImageResponse,
    BulkImageTransferResponse,
    ImageClient,
    ImageVisibility,
)


class DummyAPIClient(APIClient):
    def __init__(self, response: dict[str, Any], captured: dict[str, Any] | None = None) -> None:
        self.response = response
        self.captured = captured

    def request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.captured is not None:
            self.captured["method"] = method
            self.captured["path"] = path
            self.captured["json"] = json
        return self.response


def test_image_client_transfer_image_payload_and_response():
    captured: dict[str, Any] = {}
    client = ImageClient(
        DummyAPIClient(
            {
                "build_id": "build-123",
                "buildIds": ["build-123"],
                "upload_url": None,
                "fullImagePath": "team-team1/ubuntu:22.04",
                "visibility": "PUBLIC",
            },
            captured,
        )
    )
    response = client.transfer_image(
        "ubuntu:22.04",
        image_name="ubuntu",
        image_tag="22.04",
        team_id="team1",
        visibility=ImageVisibility.PUBLIC,
    )

    assert captured["method"] == "POST"
    assert captured["path"] == "/images/build"
    assert captured["json"] == {
        "image_name": "ubuntu",
        "image_tag": "22.04",
        "dockerfile_path": "Dockerfile",
        "source_image": "ubuntu:22.04",
        "platform": "linux/amd64",
        "team_id": "team1",
        "visibility": ImageVisibility.PUBLIC,
    }
    assert isinstance(response, BuildImageResponse)
    assert response.build_id == "build-123"
    assert response.build_ids == ["build-123"]
    assert response.upload_url is None
    assert response.full_image_path == "team-team1/ubuntu:22.04"


def test_build_image_response_allows_multi_transfer_without_full_image_path():
    response = BuildImageResponse.model_validate(
        {
            "build_id": "build-123",
            "buildIds": ["build-123", "build-456"],
            "upload_url": None,
            "visibility": "PRIVATE",
        }
    )

    assert response.build_id == "build-123"
    assert response.build_ids == ["build-123", "build-456"]
    assert response.full_image_path is None


def test_image_client_transfer_image_accepts_bulk_transfer_response():
    response = ImageClient(
        DummyAPIClient(
            {
                "results": [
                    {
                        "sourceImage": "ubuntu:22.04",
                        "success": True,
                        "buildId": "build-123",
                        "fullImagePath": "team-team1/ubuntu:22.04",
                        "visibility": "PRIVATE",
                    },
                    {
                        "sourceImage": "missing:notfound",
                        "success": False,
                        "error": "source image not found",
                        "retryable": False,
                    },
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
        )
    ).transfer_image("ubuntu:22.04,missing:notfound")

    assert isinstance(response, BulkImageTransferResponse)
    assert response.results[0].source_image == "ubuntu:22.04"
    assert response.results[0].build_id == "build-123"
    assert response.results[0].full_image_path == "team-team1/ubuntu:22.04"
    assert response.failed[0].source_image == "missing:notfound"
    assert response.failed[0].error == "source image not found"
