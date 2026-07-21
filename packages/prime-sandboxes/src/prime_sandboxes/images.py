"""Image build and transfer SDK client."""

from typing import Literal, Optional

from .core import APIClient, AsyncAPIClient
from .models import (
    BuildImageRequest,
    BuildImageResponse,
    BulkImageTransferResponse,
    ImageVisibility,
    UpdateImagesRequest,
    UpdateImagesResponse,
)


class ImageClient:
    """Client for Prime image build and transfer APIs."""

    def __init__(self, api_client: Optional[APIClient] = None):
        self.client = api_client or APIClient()

    def initiate_build(
        self, request: BuildImageRequest
    ) -> BuildImageResponse | BulkImageTransferResponse:
        payload = request.model_dump(by_alias=False, exclude_none=True)
        response = self.client.request("POST", "/images/build", json=payload)
        if "results" in response:
            return BulkImageTransferResponse.model_validate(response)
        return BuildImageResponse.model_validate(response)

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
    """Async client for Prime image build and transfer APIs."""

    def __init__(self, api_client: Optional[AsyncAPIClient] = None):
        self.client = api_client or AsyncAPIClient()

    async def initiate_build(
        self, request: BuildImageRequest
    ) -> BuildImageResponse | BulkImageTransferResponse:
        payload = request.model_dump(by_alias=False, exclude_none=True)
        response = await self.client.request("POST", "/images/build", json=payload)
        if "results" in response:
            return BulkImageTransferResponse.model_validate(response)
        return BuildImageResponse.model_validate(response)

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
