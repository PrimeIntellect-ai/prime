import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from prime_sandboxes import (
    APIClient,
    AsyncImageClient,
    BuildImageRequest,
    BuildImageResponse,
    BulkImageTransferResponse,
    ImageArtifactType,
    ImageBuildStatus,
    ImageClient,
    ImageListItem,
    ImageListResponse,
    ImageOwnerType,
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
    def __init__(
        self,
        response: dict[str, Any],
        captured: dict[str, Any] | None = None,
        *,
        team_id: str | None = None,
    ) -> None:
        self.response = response
        self.captured = captured
        self.config = SimpleNamespace(team_id=team_id)

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
            if params is not None:
                self.captured["params"] = params
        return self.response


def _image_list_response() -> dict[str, Any]:
    return {
        "data": [
            {
                "id": "image-container",
                "artifactType": "CONTAINER_IMAGE",
                "imageName": "ubuntu",
                "imageTag": "22.04",
                "status": "COMPLETED",
                "fullImagePath": "registry.test/ubuntu:22.04",
                "sizeBytes": 1024,
                "visibility": "PUBLIC",
                "createdAt": "2026-01-01T00:00:00Z",
                "pushedAt": "2026-01-01T00:01:00Z",
                "ownerType": "platform",
                "displayRef": "ubuntu:22.04",
            },
            {
                "id": "image-vm",
                "artifactType": "VM_SANDBOX",
                "imageName": "ubuntu",
                "imageTag": "22.04",
                "status": "COMPLETED",
                "fullImagePath": "vm/ubuntu:22.04",
                "errorMessage": None,
                "sizeBytes": 2048,
                "visibility": "PUBLIC",
                "createdAt": "2026-01-01T00:00:00Z",
                "startedAt": None,
                "completedAt": None,
                "pushedAt": "2026-01-01T00:02:00Z",
                "teamId": None,
                "ownerType": "platform",
                "displayRef": "ubuntu:22.04",
            },
        ],
        "totalCount": 1,
        "offset": 25,
        "limit": 1,
        "status": "ok",
    }


def test_image_client_list_forwards_query_and_parses_artifact_rows():
    captured: dict[str, Any] = {}
    client = ImageClient(
        DummyAPIClient(_image_list_response(), captured, team_id="team-configured")
    )

    response = client.list(team_id="team-explicit", search="ubuntu", offset=25, limit=1)

    assert captured == {
        "method": "GET",
        "path": "/images",
        "json": None,
        "params": {
            "teamId": "team-explicit",
            "search": "ubuntu",
            "offset": 25,
            "limit": 1,
        },
    }
    assert isinstance(response, ImageListResponse)
    assert response.total_count == 1
    assert response.status == "ok"
    assert len(response.data) == 2
    assert isinstance(response.data[0], ImageListItem)
    assert response.data[0].artifact_type == ImageArtifactType.CONTAINER_IMAGE
    assert response.data[1].artifact_type == ImageArtifactType.VM_SANDBOX
    assert response.data[1].status == ImageBuildStatus.COMPLETED
    assert response.data[1].owner_type == ImageOwnerType.PLATFORM
    assert response.data[1].display_ref == "ubuntu:22.04"
    assert response.model_dump(by_alias=True)["totalCount"] == 1
    assert response.data[1].model_dump(by_alias=True)["artifactType"] == "VM_SANDBOX"


def test_image_client_list_uses_configured_team_by_default():
    captured: dict[str, Any] = {}
    response = {"data": [], "total_count": 0, "offset": 0, "limit": 100}
    client = ImageClient(DummyAPIClient(response, captured, team_id="team-configured"))

    result = client.list()

    assert result.total_count == 0
    assert captured["params"] == {"offset": 0, "limit": 100, "teamId": "team-configured"}


def test_image_client_list_allows_missing_total_count():
    response = ImageClient(DummyAPIClient({"data": [], "offset": 0, "limit": 100})).list()

    assert response.total_count is None
    assert "totalCount" not in response.model_dump(by_alias=True, exclude_unset=True)


def test_image_client_list_platform_ignores_configured_team():
    captured: dict[str, Any] = {}
    response = {"data": [], "totalCount": 0, "offset": 0, "limit": 250}
    client = ImageClient(DummyAPIClient(response, captured, team_id="team-configured"))

    client.list(platform=True, limit=250)

    assert captured["params"] == {"offset": 0, "limit": 250, "ownerScope": "platform"}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"offset": -1}, "offset"),
        ({"limit": 0}, "limit"),
        ({"limit": 251}, "limit"),
        ({"platform": True, "team_id": "team-explicit"}, "team_id"),
        ({"platform": True, "team_id": ""}, "team_id"),
    ],
)
def test_image_client_list_validates_query(kwargs: dict[str, Any], message: str):
    captured: dict[str, Any] = {}
    client = ImageClient(
        DummyAPIClient({"data": [], "totalCount": 0, "offset": 0, "limit": 100}, captured)
    )

    with pytest.raises(ValueError, match=message):
        client.list(**kwargs)

    assert captured == {}


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


def test_image_client_build_vm_image_accepts_platform_owner_scope():
    captured: dict[str, Any] = {}
    client = ImageClient(DummyAPIClient({"buildId": "build-123"}, captured))

    response = client.build_vm_image(
        "org/ubuntu",
        "22.04",
        owner_scope="platform",
    )

    assert captured == {
        "method": "POST",
        "path": "/images/org/ubuntu/22.04/vm-build",
        "json": {"ownerScope": "platform"},
    }
    assert response == {"buildId": "build-123"}


class DummyAsyncAPIClient:
    def __init__(
        self,
        response: dict[str, Any],
        captured: dict[str, Any] | None = None,
        *,
        team_id: str | None = None,
    ) -> None:
        self.response = response
        self.captured = captured
        self.config = SimpleNamespace(team_id=team_id)

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
            if params is not None:
                self.captured["params"] = params
        return self.response


def test_async_image_client_list_forwards_query_and_parses_response():
    captured: dict[str, Any] = {}
    client = AsyncImageClient(  # type: ignore[arg-type]
        DummyAsyncAPIClient(_image_list_response(), captured, team_id="team-configured")
    )

    response = asyncio.run(
        client.list(team_id="team-explicit", search="ubuntu", offset=25, limit=1)
    )

    assert captured["method"] == "GET"
    assert captured["path"] == "/images"
    assert captured["params"] == {
        "offset": 25,
        "limit": 1,
        "teamId": "team-explicit",
        "search": "ubuntu",
    }
    assert response.total_count == 1
    assert len(response.data) == 2
    assert response.data[1].artifact_type == ImageArtifactType.VM_SANDBOX
    assert response.data[1].status == ImageBuildStatus.COMPLETED


def test_async_image_client_list_uses_configured_team_by_default():
    captured: dict[str, Any] = {}
    response = {"data": [], "totalCount": 0, "offset": 0, "limit": 100}
    client = AsyncImageClient(  # type: ignore[arg-type]
        DummyAsyncAPIClient(response, captured, team_id="team-configured")
    )

    result = asyncio.run(client.list())

    assert result.data == []
    assert captured["params"] == {"offset": 0, "limit": 100, "teamId": "team-configured"}


def test_async_image_client_list_allows_missing_total_count():
    client = AsyncImageClient(  # type: ignore[arg-type]
        DummyAsyncAPIClient({"data": [], "offset": 0, "limit": 100})
    )

    response = asyncio.run(client.list())

    assert response.total_count is None


def test_async_image_client_list_platform_ignores_configured_team():
    captured: dict[str, Any] = {}
    response = {"data": [], "totalCount": 0, "offset": 0, "limit": 1}
    client = AsyncImageClient(  # type: ignore[arg-type]
        DummyAsyncAPIClient(response, captured, team_id="team-configured")
    )

    asyncio.run(client.list(platform=True, limit=1))

    assert captured["params"] == {"offset": 0, "limit": 1, "ownerScope": "platform"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"offset": -1},
        {"limit": 0},
        {"limit": 251},
        {"platform": True, "team_id": "team-explicit"},
    ],
)
def test_async_image_client_list_validates_query(kwargs: dict[str, Any]):
    captured: dict[str, Any] = {}
    client = AsyncImageClient(  # type: ignore[arg-type]
        DummyAsyncAPIClient({"data": [], "totalCount": 0, "offset": 0, "limit": 100}, captured)
    )

    with pytest.raises(ValueError):
        asyncio.run(client.list(**kwargs))

    assert captured == {}


def test_async_image_client_build_vm_image_accepts_platform_owner_scope():
    import asyncio

    captured: dict[str, Any] = {}
    client = AsyncImageClient(DummyAsyncAPIClient({"buildId": "build-123"}, captured))  # type: ignore[arg-type]

    response = asyncio.run(
        client.build_vm_image(
            "org/ubuntu",
            "22.04",
            owner_scope="platform",
        )
    )

    assert captured == {
        "method": "POST",
        "path": "/images/org/ubuntu/22.04/vm-build",
        "json": {"ownerScope": "platform"},
    }
    assert response == {"buildId": "build-123"}


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
