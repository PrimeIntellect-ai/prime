"""Image build SDK client."""

from typing import Literal, Optional

from .core import APIClient, AsyncAPIClient
from .models import (
    BuildImageRequest,
    BuildImageResponse,
    BulkImageTransferResponse,
    ImageListResponse,
    ImageVisibility,
    UpdateImagesRequest,
    UpdateImagesResponse,
)


def _list_params(
    *,
    configured_team_id: Optional[str],
    team_id: Optional[str],
    search: Optional[str],
    platform: bool,
    offset: int,
    limit: int,
) -> dict[str, object]:
    if offset < 0:
        raise ValueError("offset must be greater than or equal to 0")
    if limit < 1 or limit > 250:
        raise ValueError("limit must be between 1 and 250")
    if platform and team_id is not None:
        raise ValueError("team_id cannot be set when platform=True")

    params: dict[str, object] = {"offset": offset, "limit": limit}
    if platform:
        params["ownerScope"] = "platform"
    else:
        resolved_team_id = team_id if team_id is not None else configured_team_id
        if resolved_team_id:
            params["teamId"] = resolved_team_id
    if search is not None:
        params["search"] = search
    return params


class ImageClient:
    """Client for Prime image build APIs."""

    def __init__(self, api_client: Optional[APIClient] = None):
        self.client = api_client or APIClient()

    def list(
        self,
        *,
        team_id: Optional[str] = None,
        search: Optional[str] = None,
        platform: bool = False,
        offset: int = 0,
        limit: int = 100,
    ) -> ImageListResponse:
        """List image artifact rows in a logical-image page."""
        params = _list_params(
            configured_team_id=self.client.config.team_id,
            team_id=team_id,
            search=search,
            platform=platform,
            offset=offset,
            limit=limit,
        )
        response = self.client.request("GET", "/images", params=params)
        return ImageListResponse.model_validate(response)

    def initiate_build(
        self, request: BuildImageRequest
    ) -> BuildImageResponse | BulkImageTransferResponse:
        """Queue a linux/amd64 Dockerfile build or public-registry VM build.

        Docker Hub source requests are public, org-less platform builds. They
        cannot set a custom destination, team, or private visibility.
        """
        payload = request.model_dump(by_alias=False, exclude_none=True)
        response = self.client.request("POST", "/images/build", json=payload)
        if "results" in response:
            return BulkImageTransferResponse.model_validate(response)
        return BuildImageResponse.model_validate(
            response,
            context={"requires_upload": request.source_image is None},
        )

    def transfer_image(
        self,
        source_image: str,
        *,
        image_name: Optional[str] = None,
        image_tag: Optional[str] = None,
        platform: str = "linux/amd64",
        team_id: Optional[str] = None,
        visibility: Optional[ImageVisibility] = None,
        owner_scope: Optional[Literal["platform"]] = None,
    ) -> BuildImageResponse | BulkImageTransferResponse:
        """Build VM images directly from allowed public registry references.

        Only ``linux/amd64`` is supported. Docker Hub sources always build as
        public, org-less platform images. They do not accept a custom
        destination, team, or private visibility.
        """
        request = BuildImageRequest(
            image_name=image_name,
            image_tag=image_tag,
            source_image=source_image,
            platform=platform,
            team_id=team_id,
            visibility=visibility,
            owner_scope=owner_scope,
        )
        return self.initiate_build(request)

    def start_build(self, build_id: str) -> dict:
        return self.client.request(
            "POST",
            f"/images/build/{build_id}/start",
            json={"context_uploaded": True},
        )

    def build_vm_image(
        self,
        image_name: str,
        image_tag: str,
        *,
        team_id: Optional[str] = None,
        owner_scope: Optional[Literal["platform"]] = None,
    ) -> dict:
        """Build a VM image from an existing container image."""
        payload = {"teamId": team_id} if team_id else {}
        if owner_scope:
            payload["ownerScope"] = owner_scope
        return self.client.request(
            "POST",
            f"/images/{image_name}/{image_tag}/vm-build",
            json=payload,
        )

    def get_build_status(self, build_id: str) -> dict:
        """Fetch the status of a build group."""
        return self.client.request("GET", f"/images/build/{build_id}")

    def update_images(self, request: UpdateImagesRequest) -> UpdateImagesResponse:
        """Update one or many logical images (visibility, name/tag, owner).

        Issues ``PATCH /images``. A valid request with item-specific failures
        still returns a response; inspect ``results[*].error``.
        """
        payload = request.model_dump(by_alias=True, exclude_none=True)
        response = self.client.request("PATCH", "/images", json=payload)
        return UpdateImagesResponse.model_validate(response)


class AsyncImageClient:
    """Async client for Prime image build APIs."""

    def __init__(self, api_client: Optional[AsyncAPIClient] = None):
        self.client = api_client or AsyncAPIClient()

    async def list(
        self,
        *,
        team_id: Optional[str] = None,
        search: Optional[str] = None,
        platform: bool = False,
        offset: int = 0,
        limit: int = 100,
    ) -> ImageListResponse:
        """List image artifact rows in a logical-image page."""
        params = _list_params(
            configured_team_id=self.client.config.team_id,
            team_id=team_id,
            search=search,
            platform=platform,
            offset=offset,
            limit=limit,
        )
        response = await self.client.request("GET", "/images", params=params)
        return ImageListResponse.model_validate(response)

    async def initiate_build(
        self, request: BuildImageRequest
    ) -> BuildImageResponse | BulkImageTransferResponse:
        """Queue a linux/amd64 Dockerfile build or public-registry VM build.

        Docker Hub source requests are public, org-less platform builds. They
        cannot set a custom destination, team, or private visibility.
        """
        payload = request.model_dump(by_alias=False, exclude_none=True)
        response = await self.client.request("POST", "/images/build", json=payload)
        if "results" in response:
            return BulkImageTransferResponse.model_validate(response)
        return BuildImageResponse.model_validate(
            response,
            context={"requires_upload": request.source_image is None},
        )

    async def transfer_image(
        self,
        source_image: str,
        *,
        image_name: Optional[str] = None,
        image_tag: Optional[str] = None,
        platform: str = "linux/amd64",
        team_id: Optional[str] = None,
        visibility: Optional[ImageVisibility] = None,
        owner_scope: Optional[Literal["platform"]] = None,
    ) -> BuildImageResponse | BulkImageTransferResponse:
        """Build VM images directly from allowed public registry references.

        Only ``linux/amd64`` is supported. Docker Hub sources always build as
        public, org-less platform images. They do not accept a custom
        destination, team, or private visibility.
        """
        request = BuildImageRequest(
            image_name=image_name,
            image_tag=image_tag,
            source_image=source_image,
            platform=platform,
            team_id=team_id,
            visibility=visibility,
            owner_scope=owner_scope,
        )
        return await self.initiate_build(request)

    async def start_build(self, build_id: str) -> dict:
        return await self.client.request(
            "POST",
            f"/images/build/{build_id}/start",
            json={"context_uploaded": True},
        )

    async def build_vm_image(
        self,
        image_name: str,
        image_tag: str,
        *,
        team_id: Optional[str] = None,
        owner_scope: Optional[Literal["platform"]] = None,
    ) -> dict:
        """Build a VM image from an existing container image."""
        payload = {"teamId": team_id} if team_id else {}
        if owner_scope:
            payload["ownerScope"] = owner_scope
        return await self.client.request(
            "POST",
            f"/images/{image_name}/{image_tag}/vm-build",
            json=payload,
        )

    async def get_build_status(self, build_id: str) -> dict:
        """Fetch the status of a build group."""
        return await self.client.request("GET", f"/images/build/{build_id}")

    async def update_images(self, request: UpdateImagesRequest) -> UpdateImagesResponse:
        """Update one or many logical images (visibility, name/tag, owner).

        Issues ``PATCH /images``. A valid request with item-specific failures
        still returns a response; inspect ``results[*].error``.
        """
        payload = request.model_dump(by_alias=True, exclude_none=True)
        response = await self.client.request("PATCH", "/images", json=payload)
        return UpdateImagesResponse.model_validate(response)

    async def aclose(self) -> None:
        await self.client.aclose()

    async def __aenter__(self) -> "AsyncImageClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.aclose()
