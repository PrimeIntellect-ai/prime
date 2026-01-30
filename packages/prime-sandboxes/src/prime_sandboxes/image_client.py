"""Image client implementations for building and managing container images."""

import asyncio
import io
import os
import tarfile
import time
from typing import Any, Dict, Optional

import httpx
import pathspec

from .core import APIClient, APIError, AsyncAPIClient
from .models import (
    Image,
    ImageBuildResponse,
    ImageBuildStatus,
    ImageListResponse,
)


def _load_dockerignore(context_path: str) -> Optional[pathspec.PathSpec]:
    """Load .dockerignore patterns if the file exists."""
    dockerignore_path = os.path.join(context_path, ".dockerignore")
    if not os.path.exists(dockerignore_path):
        return None

    with open(dockerignore_path, "r") as f:
        patterns = f.read().splitlines()

    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def _create_context_tarball(context_path: str) -> bytes:
    """Create a tar.gz archive of the build context directory."""
    buffer = io.BytesIO()
    context_path = os.path.abspath(context_path)
    spec = _load_dockerignore(context_path)

    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:

        def tar_filter(tarinfo: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
            # Get path relative to context (strip leading ./)
            rel_path = tarinfo.name.removeprefix("./")
            if not rel_path or rel_path == ".":
                return tarinfo  # Keep root directory
            if spec and spec.match_file(rel_path):
                return None  # Exclude ignored files
            return tarinfo

        tar.add(context_path, arcname=".", filter=tar_filter)

    buffer.seek(0)
    return buffer.read()


class ImageClient:
    """Client for building and managing container images."""

    def __init__(self, api_client: Optional[APIClient] = None):
        self.client = api_client or APIClient()

    def build(
        self,
        name: str,
        dockerfile_path: str = "Dockerfile",
        tag: str = "latest",
        context_path: Optional[str] = None,
        ephemeral: bool = False,
        team_id: Optional[str] = None,
        wait: bool = True,
        poll_interval: float = 2.0,
        timeout: int = 1800,
    ) -> Image:
        """Build a container image from a Dockerfile.

        Args:
            name: Image name
            dockerfile_path: Path to Dockerfile (default: "Dockerfile")
            tag: Image tag (default: "latest")
            context_path: Build context directory path. If not provided, only
                the Dockerfile is sent to the server (fast path, no COPY/ADD support).
            ephemeral: Auto-delete after 24 hours (default: False)
            team_id: Team ID for team images (optional)
            wait: Wait for build to complete (default: True)
            poll_interval: Seconds between status polls (default: 2.0)
            timeout: Maximum seconds to wait for build (default: 1800)

        Returns:
            Image object with build result

        Raises:
            FileNotFoundError: If Dockerfile not found
            APIError: If build fails
        """
        # Auto-populate team_id from config if not specified
        if team_id is None:
            team_id = self.client.config.team_id

        # Build request
        build_request: Dict[str, Any] = {
            "image_name": name,
            "image_tag": tag,
            "dockerfile_path": dockerfile_path,
            "is_ephemeral": ephemeral,
        }
        if team_id:
            build_request["team_id"] = team_id

        if context_path and not os.path.isdir(context_path):
            raise FileNotFoundError(f"Build context directory not found: {context_path}")

        # No context_path means read Dockerfile and send content directly
        if not context_path:
            if not os.path.exists(dockerfile_path):
                raise FileNotFoundError(f"Dockerfile not found: {dockerfile_path}")

            with open(dockerfile_path, "r") as f:
                dockerfile_content = f.read()

            build_request["dockerfile_content"] = dockerfile_content

        response = self.client.request("POST", "/images/build", json=build_request)
        build_response = ImageBuildResponse.model_validate(response)

        # Check if cached image was found, construct Image directly without extra API call
        if build_response.cached:
            return Image(
                id=build_response.image_id,
                image_ref=build_response.image_ref,
                name=build_response.image_name,
                tag=build_response.image_tag,
                status=ImageBuildStatus.COMPLETED,
                created_at=build_response.created_at,
            )

        # Upload build context and start build
        if build_response.upload_url and context_path:
            context_bytes = _create_context_tarball(context_path)
            self._upload_context(build_response.upload_url, context_bytes)

            self.client.request(
                "POST",
                f"/images/build/{build_response.build_id}/start",
                json={"context_uploaded": True},
            )

        if not wait:
            return self._get_image_from_build_id(build_response.build_id)

        # Poll for completion
        return self._wait_for_build(
            build_response.build_id,
            poll_interval=poll_interval,
            timeout=timeout,
        )

    def _upload_context(self, upload_url: str, context_bytes: bytes) -> None:
        """Upload build context to presigned URL."""
        with httpx.Client(timeout=600) as http_client:
            response = http_client.put(
                upload_url,
                content=context_bytes,
                headers={"Content-Type": "application/gzip"},
            )
            response.raise_for_status()

    def _get_image_from_build_id(self, build_id: str) -> Image:
        """Get Image object from build status."""
        response = self.client.request("GET", f"/images/build/{build_id}")
        return Image(
            id=response["id"],
            imageRef=response.get("fullImagePath", ""),
            imageName=response["imageName"],
            imageTag=response["imageTag"],
            status=ImageBuildStatus(response["status"]),
            sizeBytes=response.get("sizeBytes"),
            isEphemeral=response.get("isEphemeral", False),
            dockerfileHash=response.get("dockerfileHash"),
            createdAt=response["createdAt"],
            errorMessage=response.get("errorMessage"),
            teamId=response.get("teamId"),
        )

    def _wait_for_build(
        self,
        build_id: str,
        poll_interval: float = 2.0,
        timeout: int = 1800,
    ) -> Image:
        """Wait for build to complete and return Image."""
        start_time = time.time()

        while True:
            if time.time() - start_time > timeout:
                raise APIError(f"Build {build_id} timed out after {timeout} seconds")

            image = self._get_image_from_build_id(build_id)

            if image.status == ImageBuildStatus.COMPLETED:
                return image
            elif image.status == ImageBuildStatus.FAILED:
                raise APIError(f"Build {build_id} failed: {image.error_message or 'Unknown error'}")
            elif image.status == ImageBuildStatus.CANCELLED:
                raise APIError(f"Build {build_id} was cancelled")

            time.sleep(poll_interval)

    def get_build_status(self, build_id: str) -> Image:
        """Get the current status of an image build."""
        return self._get_image_from_build_id(build_id)

    def list(self, team_id: Optional[str] = None) -> ImageListResponse:
        """List all images accessible to the current user."""
        if team_id is None:
            team_id = self.client.config.team_id

        params = {}
        if team_id:
            params["teamId"] = team_id

        response = self.client.request("GET", "/images", params=params or None)
        images_data = response.get("data", [])

        images = []
        for img in images_data:
            images.append(
                Image(
                    id=img["id"],
                    imageRef=img.get("fullImagePath", ""),
                    imageName=img["imageName"],
                    imageTag=img["imageTag"],
                    status=ImageBuildStatus(img["status"]),
                    sizeBytes=img.get("sizeBytes"),
                    isEphemeral=img.get("isEphemeral", False),
                    dockerfileHash=img.get("dockerfileHash"),
                    createdAt=img["createdAt"],
                    errorMessage=img.get("errorMessage"),
                    teamId=img.get("teamId"),
                    displayRef=img.get("displayRef"),
                )
            )

        return ImageListResponse(images=images, total=len(images))

    def delete(
        self,
        name: str,
        tag: str = "latest",
        team_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delete an image."""
        if team_id is None:
            team_id = self.client.config.team_id

        params = {}
        if team_id:
            params["teamId"] = team_id

        return self.client.request(
            "DELETE",
            f"/images/{name}/{tag}",
            params=params or None,
        )


class AsyncImageClient:
    """Async client for building and managing container images."""

    def __init__(self, api_client: Optional[AsyncAPIClient] = None):
        self.client = api_client or AsyncAPIClient()

    async def build(
        self,
        name: str,
        dockerfile_path: str = "Dockerfile",
        tag: str = "latest",
        context_path: Optional[str] = None,
        ephemeral: bool = False,
        team_id: Optional[str] = None,
        wait: bool = True,
        poll_interval: float = 2.0,
        timeout: int = 1800,
    ) -> Image:
        """Build a container image from a Dockerfile.

        Args:
            name: Image name
            dockerfile_path: Path to Dockerfile (default: "Dockerfile")
            tag: Image tag (default: "latest")
            context_path: Build context directory path. If not provided, only
                the Dockerfile is sent to the server (fast path, no COPY/ADD support).
            ephemeral: Auto-delete after 24 hours (default: False)
            team_id: Team ID for team images (optional)
            wait: Wait for build to complete (default: True)
            poll_interval: Seconds between status polls (default: 2.0)
            timeout: Maximum seconds to wait for build (default: 1800)

        Returns:
            Image object with build result

        Raises:
            FileNotFoundError: If Dockerfile not found
            APIError: If build fails
        """
        if team_id is None:
            team_id = self.client.config.team_id

        build_request: Dict[str, Any] = {
            "image_name": name,
            "image_tag": tag,
            "dockerfile_path": dockerfile_path,
            "is_ephemeral": ephemeral,
        }
        if team_id:
            build_request["team_id"] = team_id

        # Validate context_path exists before making API call to avoid orphaned builds
        if context_path and not os.path.isdir(context_path):
            raise FileNotFoundError(f"Build context directory not found: {context_path}")

        # TODO: Use aiofiles or run_in_executor for file I/O to avoid blocking event loop
        # No context_path means read Dockerfile and send content directly
        if not context_path:
            if not os.path.exists(dockerfile_path):
                raise FileNotFoundError(f"Dockerfile not found: {dockerfile_path}")

            with open(dockerfile_path, "r") as f:
                dockerfile_content = f.read()

            build_request["dockerfile_content"] = dockerfile_content

        response = await self.client.request("POST", "/images/build", json=build_request)
        build_response = ImageBuildResponse.model_validate(response)

        # Check if cached image was found, construct Image directly without extra API call
        if build_response.cached:
            return Image(
                id=build_response.image_id,
                image_ref=build_response.image_ref,
                name=build_response.image_name,
                tag=build_response.image_tag,
                status=ImageBuildStatus.COMPLETED,
                created_at=build_response.created_at,
            )

        # Upload build context and start build
        if build_response.upload_url and context_path:
            context_bytes = _create_context_tarball(context_path)
            await self._upload_context(build_response.upload_url, context_bytes)

            await self.client.request(
                "POST",
                f"/images/build/{build_response.build_id}/start",
                json={"context_uploaded": True},
            )

        if not wait:
            return await self._get_image_from_build_id(build_response.build_id)

        # Poll for completion
        return await self._wait_for_build(
            build_response.build_id,
            poll_interval=poll_interval,
            timeout=timeout,
        )

    async def _upload_context(self, upload_url: str, context_bytes: bytes) -> None:
        """Upload build context to presigned URL."""
        async with httpx.AsyncClient(timeout=600) as http_client:
            response = await http_client.put(
                upload_url,
                content=context_bytes,
                headers={"Content-Type": "application/gzip"},
            )
            response.raise_for_status()

    async def _get_image_from_build_id(self, build_id: str) -> Image:
        """Get Image object from build status."""
        response = await self.client.request("GET", f"/images/build/{build_id}")
        return Image(
            id=response["id"],
            imageRef=response.get("fullImagePath", ""),
            imageName=response["imageName"],
            imageTag=response["imageTag"],
            status=ImageBuildStatus(response["status"]),
            sizeBytes=response.get("sizeBytes"),
            isEphemeral=response.get("isEphemeral", False),
            dockerfileHash=response.get("dockerfileHash"),
            createdAt=response["createdAt"],
            errorMessage=response.get("errorMessage"),
            teamId=response.get("teamId"),
        )

    async def _wait_for_build(
        self,
        build_id: str,
        poll_interval: float = 2.0,
        timeout: int = 1800,
    ) -> Image:
        """Wait for build to complete and return Image."""
        start_time = time.time()

        while True:
            if time.time() - start_time > timeout:
                raise APIError(f"Build {build_id} timed out after {timeout} seconds")

            image = await self._get_image_from_build_id(build_id)

            if image.status == ImageBuildStatus.COMPLETED:
                return image
            elif image.status == ImageBuildStatus.FAILED:
                raise APIError(f"Build {build_id} failed: {image.error_message or 'Unknown error'}")
            elif image.status == ImageBuildStatus.CANCELLED:
                raise APIError(f"Build {build_id} was cancelled")

            await asyncio.sleep(poll_interval)

    async def get_build_status(self, build_id: str) -> Image:
        """Get the current status of an image build."""
        return await self._get_image_from_build_id(build_id)

    async def list(self, team_id: Optional[str] = None) -> ImageListResponse:
        """List all images accessible to the current user."""
        if team_id is None:
            team_id = self.client.config.team_id

        params = {}
        if team_id:
            params["teamId"] = team_id

        response = await self.client.request("GET", "/images", params=params or None)
        images_data = response.get("data", [])

        images = []
        for img in images_data:
            images.append(
                Image(
                    id=img["id"],
                    imageRef=img.get("fullImagePath", ""),
                    imageName=img["imageName"],
                    imageTag=img["imageTag"],
                    status=ImageBuildStatus(img["status"]),
                    sizeBytes=img.get("sizeBytes"),
                    isEphemeral=img.get("isEphemeral", False),
                    dockerfileHash=img.get("dockerfileHash"),
                    createdAt=img["createdAt"],
                    errorMessage=img.get("errorMessage"),
                    teamId=img.get("teamId"),
                    displayRef=img.get("displayRef"),
                )
            )

        return ImageListResponse(images=images, total=len(images))

    async def delete(
        self,
        name: str,
        tag: str = "latest",
        team_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delete an image."""
        if team_id is None:
            team_id = self.client.config.team_id

        params = {}
        if team_id:
            params["teamId"] = team_id

        return await self.client.request(
            "DELETE",
            f"/images/{name}/{tag}",
            params=params or None,
        )

    async def aclose(self) -> None:
        """Close the async client."""
        await self.client.aclose()

    async def __aenter__(self) -> "AsyncImageClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.aclose()
