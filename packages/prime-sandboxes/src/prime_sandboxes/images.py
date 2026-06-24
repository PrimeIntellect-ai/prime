"""Image build and transfer SDK client."""

from typing import Optional

from .core import APIClient, AsyncAPIClient
from .models import BuildImageRequest, BuildImageResponse, ImageVisibility


class ImageClient:
    """Client for Prime image build and transfer APIs."""

    def __init__(self, api_client: Optional[APIClient] = None):
        self.client = api_client or APIClient()

    def initiate_build(self, request: BuildImageRequest) -> BuildImageResponse:
        payload = request.model_dump(by_alias=False, exclude_none=True)
        response = self.client.request("POST", "/images/build", json=payload)
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
    ) -> BuildImageResponse:
        request = BuildImageRequest(
            image_name=image_name,
            image_tag=image_tag,
            source_image=source_image,
            platform=platform,
            team_id=team_id,
            visibility=visibility,
        )
        return self.initiate_build(request)

    def start_build(self, build_id: str) -> dict:
        return self.client.request(
            "POST",
            f"/images/build/{build_id}/start",
            json={"context_uploaded": True},
        )


class AsyncImageClient:
    """Async client for Prime image build and transfer APIs."""

    def __init__(self, api_client: Optional[AsyncAPIClient] = None):
        self.client = api_client or AsyncAPIClient()

    async def initiate_build(self, request: BuildImageRequest) -> BuildImageResponse:
        payload = request.model_dump(by_alias=False, exclude_none=True)
        response = await self.client.request("POST", "/images/build", json=payload)
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
    ) -> BuildImageResponse:
        request = BuildImageRequest(
            image_name=image_name,
            image_tag=image_tag,
            source_image=source_image,
            platform=platform,
            team_id=team_id,
            visibility=visibility,
        )
        return await self.initiate_build(request)

    async def start_build(self, build_id: str) -> dict:
        return await self.client.request(
            "POST",
            f"/images/build/{build_id}/start",
            json={"context_uploaded": True},
        )

    async def aclose(self) -> None:
        await self.client.aclose()

    async def __aenter__(self) -> "AsyncImageClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.aclose()
