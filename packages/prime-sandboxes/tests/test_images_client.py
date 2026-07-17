from typing import Any

from prime_sandboxes import (
    APIClient,
    AsyncImageClient,
    BuildImageRequest,
    BuildImageResponse,
    BulkImageTransferResponse,
    ImageClient,
    ImageUpdateItem,
    ImageUpdatePatch,
    ImageUpdateSource,
    ImageVisibility,
    PersonalImageOwner,
    TeamImageOwner,
    UpdateImagesRequest,
    UpdateImagesResponse,
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
                "fullImagePath": "prime/research/ubuntu:22.04",
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
    assert response.full_image_path == "prime/research/ubuntu:22.04"


def test_build_image_response_allows_multi_transfer_without_full_image_path():
    response = BuildImageResponse.model_validate(
        {
            "build_id": "build-123",
            "buildIds": ["build-123", "build-456"],
            "upload_url": None,
            "fullImagePath": "prime/research/ubuntu:22.04",
            "visibility": "PRIVATE",
        }
    )

    assert response.build_id == "build-123"
    assert response.build_ids == ["build-123", "build-456"]
    assert response.full_image_path == "prime/research/ubuntu:22.04"


def test_image_client_initiate_build_accepts_platform_owner_scope():
    captured: dict[str, Any] = {}
    client = ImageClient(
        DummyAPIClient(
            {
                "build_id": "build-123",
                "buildIds": ["build-123"],
                "upload_url": "https://example.test/upload",
                "fullImagePath": "ubuntu:22.04",
                "visibility": "PUBLIC",
            },
            captured,
        )
    )

    response = client.initiate_build(
        BuildImageRequest(
            image_name="ubuntu",
            image_tag="22.04",
            visibility=ImageVisibility.PUBLIC,
            owner_scope="platform",
        )
    )

    assert captured["method"] == "POST"
    assert captured["path"] == "/images/build"
    assert captured["json"] == {
        "image_name": "ubuntu",
        "image_tag": "22.04",
        "dockerfile_path": "Dockerfile",
        "platform": "linux/amd64",
        "visibility": ImageVisibility.PUBLIC,
        "owner_scope": "platform",
    }
    assert isinstance(response, BuildImageResponse)
    assert response.build_id == "build-123"


def test_image_client_transfer_image_accepts_platform_owner_scope():
    captured: dict[str, Any] = {}
    client = ImageClient(
        DummyAPIClient(
            {
                "build_id": "build-123",
                "buildIds": ["build-123"],
                "upload_url": None,
                "fullImagePath": "ubuntu:22.04",
                "visibility": "PUBLIC",
            },
            captured,
        )
    )

    response = client.transfer_image(
        "ubuntu:22.04",
        visibility=ImageVisibility.PUBLIC,
        owner_scope="platform",
    )

    assert captured["method"] == "POST"
    assert captured["path"] == "/images/build"
    assert captured["json"] == {
        "dockerfile_path": "Dockerfile",
        "source_image": "ubuntu:22.04",
        "platform": "linux/amd64",
        "visibility": ImageVisibility.PUBLIC,
        "owner_scope": "platform",
    }
    assert isinstance(response, BuildImageResponse)
    assert response.full_image_path == "ubuntu:22.04"


def test_image_client_transfer_image_accepts_bulk_transfer_response():
    response = ImageClient(
        DummyAPIClient(
            {
                "results": [
                    {
                        "sourceImage": "ubuntu:22.04",
                        "success": True,
                        "buildId": "build-123",
                        "fullImagePath": "prime/research/ubuntu:22.04",
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
    assert response.results[0].full_image_path == "prime/research/ubuntu:22.04"
    assert response.failed[0].source_image == "missing:notfound"
    assert response.failed[0].error == "source image not found"


class DummyAsyncAPIClient:
    def __init__(self, response: dict[str, Any], captured: dict[str, Any] | None = None) -> None:
        self.response = response
        self.captured = captured

    async def request(
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


def _update_images_response(reference: str) -> dict[str, Any]:
    return {
        "success": True,
        "dryRun": False,
        "results": [
            {
                "source": {"reference": reference},
                "success": True,
                "before": {
                    "owner": {"type": "personal"},
                    "name": "app",
                    "tag": "v1",
                    "visibility": "PRIVATE",
                },
                "after": {
                    "owner": {"type": "team", "teamId": "team1"},
                    "name": "app",
                    "tag": "v1",
                    "visibility": "PRIVATE",
                },
            }
        ],
    }


def test_image_client_update_images_explicit_payload_and_response():
    captured: dict[str, Any] = {}
    client = ImageClient(DummyAPIClient(_update_images_response("prime/alice/app:v1"), captured))

    response = client.update_images(
        UpdateImagesRequest(
            updates=[
                ImageUpdateItem(
                    source=ImageUpdateSource(reference="prime/alice/app:v1"),
                    set=ImageUpdatePatch(owner=TeamImageOwner(team_id="team1")),
                )
            ]
        )
    )

    assert captured["method"] == "PATCH"
    assert captured["path"] == "/images"
    assert captured["json"] == {
        "mode": "explicit",
        "dryRun": False,
        "updates": [
            {
                "source": {"reference": "prime/alice/app:v1"},
                "set": {"owner": {"type": "team", "teamId": "team1"}},
            }
        ],
    }
    assert isinstance(response, UpdateImagesResponse)
    assert response.success
    result = response.results[0]
    assert result.success
    assert result.before is not None and result.before.visibility == ImageVisibility.PRIVATE
    assert isinstance(result.after.owner, TeamImageOwner)
    assert result.after.owner.team_id == "team1"


def test_image_client_update_images_partial_failure():
    client = ImageClient(
        DummyAPIClient(
            {
                "success": False,
                "dryRun": False,
                "results": [
                    {
                        "source": {
                            "owner": {"type": "personal"},
                            "name": "app",
                            "tag": "v1",
                        },
                        "success": True,
                    },
                    {
                        "source": {
                            "owner": {"type": "personal"},
                            "name": "missing",
                            "tag": "latest",
                        },
                        "success": False,
                        "error": {
                            "code": "image_not_found",
                            "message": "Image missing:latest not found",
                        },
                    },
                ],
            }
        )
    )

    response = client.update_images(
        UpdateImagesRequest(
            updates=[
                ImageUpdateItem(
                    source=ImageUpdateSource(owner=PersonalImageOwner(), name="app", tag="v1"),
                    set=ImageUpdatePatch(visibility=ImageVisibility.PUBLIC),
                ),
                ImageUpdateItem(
                    source=ImageUpdateSource(
                        owner=PersonalImageOwner(), name="missing", tag="latest"
                    ),
                    set=ImageUpdatePatch(visibility=ImageVisibility.PUBLIC),
                ),
            ]
        )
    )

    assert not response.success
    assert response.results[0].success
    failure = response.results[1]
    assert not failure.success
    assert failure.error is not None
    assert failure.error.code == "image_not_found"
    assert "not found" in failure.error.message


def test_async_image_client_update_images():
    import asyncio

    captured: dict[str, Any] = {}
    client = AsyncImageClient(
        DummyAsyncAPIClient(_update_images_response("app:v1"), captured)  # type: ignore[arg-type]
    )

    response = asyncio.run(
        client.update_images(
            UpdateImagesRequest(
                updates=[
                    ImageUpdateItem(
                        source=ImageUpdateSource(owner=PersonalImageOwner(), name="app", tag="v1"),
                        set=ImageUpdatePatch(visibility=ImageVisibility.PRIVATE),
                    )
                ]
            )
        )
    )

    assert captured["method"] == "PATCH"
    assert captured["path"] == "/images"
    assert captured["json"]["updates"] == [
        {
            "source": {"owner": {"type": "personal"}, "name": "app", "tag": "v1"},
            "set": {"visibility": ImageVisibility.PRIVATE},
        }
    ]
    assert isinstance(response, UpdateImagesResponse)
    assert response.results[0].success
