"""Declarative image builders and Prime image build clients."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import re
import shlex
import tarfile
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import httpx

from .core import APIClient, APIError, APITimeoutError, AsyncAPIClient

PACKAGED_DOCKERFILE_PATH = ".__prime_dockerfile__"
DEFAULT_DECLARATIVE_IMAGE_NAME = "declarative-sandbox"
DEFAULT_IMAGE_PLATFORM = "linux/amd64"

BuildLogCallback = Callable[[str], None]

_ACTIVE_STATUSES = {"BUILDING", "PENDING", "UPLOADING"}
_TERMINAL_FAILURE_STATUSES = {"FAILED", "CANCELLED"}
_LATEST_ROW_KEYS: tuple[str, ...] = ("pushedAt", "completedAt", "startedAt", "createdAt")
_IMAGE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_IMAGE_TAG_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class Image:
    """Declarative Docker image definition for sandbox creation.

    Instances are immutable. Builder methods return a new image with the extra
    Dockerfile instruction appended in call order.
    """

    base_image: str
    _instructions: tuple[str, ...] = ()

    @classmethod
    def from_registry(cls, image: str) -> "Image":
        """Start from any registry image reference."""
        _require_single_line("image", image)
        if not image.strip():
            raise ValueError("image must not be empty")
        return cls(base_image=image.strip())

    @classmethod
    def debian_slim(cls, python_version: str = "3.12") -> "Image":
        """Start from Prime's Python-on-Debian slim base image."""
        _require_single_line("python_version", python_version)
        if not python_version.strip():
            raise ValueError("python_version must not be empty")
        return cls.from_registry(f"python:{python_version.strip()}-slim")

    @classmethod
    def python_slim(cls, python_version: str = "3.12") -> "Image":
        """Alias for :meth:`debian_slim` for users who prefer explicit Python naming."""
        return cls.debian_slim(python_version)

    def apt_install(self, packages: str | Iterable[str]) -> "Image":
        """Install Debian packages with apt."""
        package_list = _coerce_non_empty_list("packages", packages)
        quoted = " ".join(shlex.quote(package) for package in package_list)
        return self._append(
            "RUN apt-get update && apt-get install -y --no-install-recommends "
            f"{quoted} && rm -rf /var/lib/apt/lists/*"
        )

    def pip_install(self, packages: str | Iterable[str]) -> "Image":
        """Install Python packages with pip."""
        package_list = _coerce_non_empty_list("packages", packages)
        quoted = " ".join(shlex.quote(package) for package in package_list)
        return self._append(f"RUN python -m pip install --no-cache-dir {quoted}")

    def run(self, command: str) -> "Image":
        """Append a Dockerfile RUN instruction."""
        _require_single_line("command", command)
        if not command.strip():
            raise ValueError("command must not be empty")
        return self._append(f"RUN {command.strip()}")

    def env(self, variables: Optional[Mapping[str, str]] = None, **kwargs: str) -> "Image":
        """Set environment variables in the image."""
        merged: dict[str, str] = {}
        if variables:
            merged.update({str(key): str(value) for key, value in variables.items()})
        merged.update({str(key): str(value) for key, value in kwargs.items()})
        if not merged:
            raise ValueError("at least one environment variable is required")

        image = self
        for key, value in merged.items():
            if not _ENV_KEY_RE.match(key):
                raise ValueError(f"invalid environment variable name: {key!r}")
            _require_single_line(key, value)
            image = image._append(f"ENV {key}={_dockerfile_double_quote(value)}")
        return image

    def workdir(self, path: str) -> "Image":
        """Set the working directory."""
        _require_single_line("path", path)
        if not path.strip():
            raise ValueError("path must not be empty")
        return self._append(f"WORKDIR {path.strip()}")

    def cmd(self, command: str | Sequence[str]) -> "Image":
        """Set the image CMD instruction."""
        if isinstance(command, str):
            _require_single_line("command", command)
            if not command.strip():
                raise ValueError("command must not be empty")
            return self._append(f"CMD {command.strip()}")

        command_list = _coerce_non_empty_list("command", command)
        return self._append(f"CMD {json.dumps(command_list)}")

    def dockerfile(self) -> str:
        """Render this declarative image as a Dockerfile."""
        lines = [f"FROM {self.base_image}", *self._instructions]
        return "\n".join(lines) + "\n"

    def content_hash(self) -> str:
        """Stable hash of the rendered Dockerfile."""
        return hashlib.sha256(self.dockerfile().encode("utf-8")).hexdigest()

    def default_tag(self) -> str:
        """Stable default image tag derived from the image definition."""
        return f"sha-{self.content_hash()[:12]}"

    def _append(self, instruction: str) -> "Image":
        return replace(self, _instructions=(*self._instructions, instruction))


@dataclass(frozen=True)
class ImageBuildResult:
    """Result returned after building or reusing a declarative image."""

    image_reference: str
    image_name: str
    image_tag: str
    status: str
    build_id: Optional[str] = None
    reused: bool = False


class ImageClient:
    """Client for Prime image build APIs."""

    def __init__(self, api_client: Optional[APIClient] = None):
        self.client = api_client or APIClient()

    def build(
        self,
        image: Image,
        *,
        image_name: Optional[str] = None,
        image_tag: Optional[str] = None,
        platform: str = DEFAULT_IMAGE_PLATFORM,
        team_id: Optional[str] = None,
        visibility: Optional[str] = None,
        force_rebuild: bool = False,
        timeout_seconds: Optional[float] = 900,
        poll_interval_seconds: float = 5.0,
        on_build_log: Optional[BuildLogCallback] = None,
    ) -> ImageBuildResult:
        """Build a declarative image and return the registry reference.

        A ``timeout_seconds`` value of ``0`` or ``None`` waits indefinitely.
        """
        image_name, image_tag = _resolve_image_name_tag(image, image_name, image_tag)
        effective_team_id = _effective_team_id(self.client, team_id)

        if not force_rebuild:
            existing = self._latest_image_row(image_name, image_tag, effective_team_id)
            if existing is not None:
                status = str(existing.get("status") or "UNKNOWN")
                if status == "COMPLETED":
                    image_ref = _image_reference_from_row(existing, image_name, image_tag)
                    _emit(on_build_log, f"Using existing image {image_ref}")
                    return ImageBuildResult(
                        image_reference=image_ref,
                        image_name=image_name,
                        image_tag=image_tag,
                        status=status,
                        reused=True,
                    )
                if status in _ACTIVE_STATUSES:
                    _emit(on_build_log, f"Waiting for existing image build ({status})")
                    image_ref = self._wait_for_image_ready(
                        image_name,
                        image_tag,
                        effective_team_id,
                        timeout_seconds,
                        poll_interval_seconds,
                        on_build_log,
                    )
                    return ImageBuildResult(
                        image_reference=image_ref,
                        image_name=image_name,
                        image_tag=image_tag,
                        status="COMPLETED",
                        reused=True,
                    )

        dockerfile = image.dockerfile()
        _emit(on_build_log, f"Preparing declarative image {image_name}:{image_tag}")
        archive = _build_context_archive(dockerfile)

        payload: dict[str, Any] = {
            "image_name": image_name,
            "image_tag": image_tag,
            "dockerfile_path": PACKAGED_DOCKERFILE_PATH,
            "platform": platform,
        }
        if effective_team_id:
            payload["team_id"] = effective_team_id
        if visibility:
            payload["visibility"] = _normalize_visibility(visibility)

        _emit(on_build_log, "Initiating image build")
        build_response = self.client.request("POST", "/images/build", json=payload)
        build_id = build_response.get("build_id")
        upload_url = build_response.get("upload_url")
        if not build_id or not upload_url:
            raise APIError("Invalid response from image build API: missing build_id or upload_url")
        image_reference = str(build_response.get("fullImagePath") or f"{image_name}:{image_tag}")

        _emit(on_build_log, "Uploading image build context")
        upload_response = httpx.put(
            str(upload_url),
            content=archive,
            headers={"Content-Type": "application/octet-stream"},
            timeout=600.0,
        )
        upload_response.raise_for_status()

        _emit(on_build_log, "Starting image build")
        self.client.request(
            "POST",
            f"/images/build/{build_id}/start",
            json={"context_uploaded": True},
        )

        image_reference = self._wait_for_image_ready(
            image_name,
            image_tag,
            effective_team_id,
            timeout_seconds,
            poll_interval_seconds,
            on_build_log,
        )
        return ImageBuildResult(
            image_reference=image_reference,
            image_name=image_name,
            image_tag=image_tag,
            status="COMPLETED",
            build_id=str(build_id),
        )

    def _latest_image_row(
        self, image_name: str, image_tag: str, team_id: Optional[str]
    ) -> Optional[dict[str, Any]]:
        params: dict[str, str] = {"limit": "250", "offset": "0"}
        if team_id:
            params["teamId"] = team_id
        response = self.client.request("GET", "/images", params=params)
        return _latest_matching_row(response.get("data", []), image_name, image_tag, team_id)

    def _wait_for_image_ready(
        self,
        image_name: str,
        image_tag: str,
        team_id: Optional[str],
        timeout_seconds: Optional[float],
        poll_interval_seconds: float,
        on_build_log: Optional[BuildLogCallback],
    ) -> str:
        deadline = _deadline(timeout_seconds)
        last_status: Optional[str] = None
        while True:
            row = self._latest_image_row(image_name, image_tag, team_id)
            if row is not None:
                status = str(row.get("status") or "UNKNOWN")
                if status != last_status:
                    _emit(on_build_log, f"Image build status: {status}")
                    last_status = status
                if status == "COMPLETED":
                    image_ref = _image_reference_from_row(row, image_name, image_tag)
                    _emit(on_build_log, f"Image ready: {image_ref}")
                    return image_ref
                if status in _TERMINAL_FAILURE_STATUSES:
                    raise APIError(f"Image build {status.lower()} for {image_name}:{image_tag}")
            elif last_status is None:
                _emit(on_build_log, "Waiting for image build to appear")
                last_status = "UNKNOWN"

            _raise_if_timed_out(deadline, image_name, image_tag)
            time.sleep(max(0.0, poll_interval_seconds))


class AsyncImageClient:
    """Async client for Prime image build APIs."""

    def __init__(self, api_client: Optional[AsyncAPIClient] = None):
        self.client = api_client or AsyncAPIClient()

    async def build(
        self,
        image: Image,
        *,
        image_name: Optional[str] = None,
        image_tag: Optional[str] = None,
        platform: str = DEFAULT_IMAGE_PLATFORM,
        team_id: Optional[str] = None,
        visibility: Optional[str] = None,
        force_rebuild: bool = False,
        timeout_seconds: Optional[float] = 900,
        poll_interval_seconds: float = 5.0,
        on_build_log: Optional[BuildLogCallback] = None,
    ) -> ImageBuildResult:
        """Build a declarative image and return the registry reference."""
        image_name, image_tag = _resolve_image_name_tag(image, image_name, image_tag)
        effective_team_id = _effective_team_id(self.client, team_id)

        if not force_rebuild:
            existing = await self._latest_image_row(image_name, image_tag, effective_team_id)
            if existing is not None:
                status = str(existing.get("status") or "UNKNOWN")
                if status == "COMPLETED":
                    image_ref = _image_reference_from_row(existing, image_name, image_tag)
                    _emit(on_build_log, f"Using existing image {image_ref}")
                    return ImageBuildResult(
                        image_reference=image_ref,
                        image_name=image_name,
                        image_tag=image_tag,
                        status=status,
                        reused=True,
                    )
                if status in _ACTIVE_STATUSES:
                    _emit(on_build_log, f"Waiting for existing image build ({status})")
                    image_ref = await self._wait_for_image_ready(
                        image_name,
                        image_tag,
                        effective_team_id,
                        timeout_seconds,
                        poll_interval_seconds,
                        on_build_log,
                    )
                    return ImageBuildResult(
                        image_reference=image_ref,
                        image_name=image_name,
                        image_tag=image_tag,
                        status="COMPLETED",
                        reused=True,
                    )

        dockerfile = image.dockerfile()
        _emit(on_build_log, f"Preparing declarative image {image_name}:{image_tag}")
        archive = _build_context_archive(dockerfile)

        payload: dict[str, Any] = {
            "image_name": image_name,
            "image_tag": image_tag,
            "dockerfile_path": PACKAGED_DOCKERFILE_PATH,
            "platform": platform,
        }
        if effective_team_id:
            payload["team_id"] = effective_team_id
        if visibility:
            payload["visibility"] = _normalize_visibility(visibility)

        _emit(on_build_log, "Initiating image build")
        build_response = await self.client.request("POST", "/images/build", json=payload)
        build_id = build_response.get("build_id")
        upload_url = build_response.get("upload_url")
        if not build_id or not upload_url:
            raise APIError("Invalid response from image build API: missing build_id or upload_url")
        image_reference = str(build_response.get("fullImagePath") or f"{image_name}:{image_tag}")

        _emit(on_build_log, "Uploading image build context")
        async with httpx.AsyncClient(timeout=600.0) as http_client:
            upload_response = await http_client.put(
                str(upload_url),
                content=archive,
                headers={"Content-Type": "application/octet-stream"},
            )
            upload_response.raise_for_status()

        _emit(on_build_log, "Starting image build")
        await self.client.request(
            "POST",
            f"/images/build/{build_id}/start",
            json={"context_uploaded": True},
        )

        image_reference = await self._wait_for_image_ready(
            image_name,
            image_tag,
            effective_team_id,
            timeout_seconds,
            poll_interval_seconds,
            on_build_log,
        )
        return ImageBuildResult(
            image_reference=image_reference,
            image_name=image_name,
            image_tag=image_tag,
            status="COMPLETED",
            build_id=str(build_id),
        )

    async def _latest_image_row(
        self, image_name: str, image_tag: str, team_id: Optional[str]
    ) -> Optional[dict[str, Any]]:
        params: dict[str, str] = {"limit": "250", "offset": "0"}
        if team_id:
            params["teamId"] = team_id
        response = await self.client.request("GET", "/images", params=params)
        return _latest_matching_row(response.get("data", []), image_name, image_tag, team_id)

    async def _wait_for_image_ready(
        self,
        image_name: str,
        image_tag: str,
        team_id: Optional[str],
        timeout_seconds: Optional[float],
        poll_interval_seconds: float,
        on_build_log: Optional[BuildLogCallback],
    ) -> str:
        deadline = _deadline(timeout_seconds)
        last_status: Optional[str] = None
        while True:
            row = await self._latest_image_row(image_name, image_tag, team_id)
            if row is not None:
                status = str(row.get("status") or "UNKNOWN")
                if status != last_status:
                    _emit(on_build_log, f"Image build status: {status}")
                    last_status = status
                if status == "COMPLETED":
                    image_ref = _image_reference_from_row(row, image_name, image_tag)
                    _emit(on_build_log, f"Image ready: {image_ref}")
                    return image_ref
                if status in _TERMINAL_FAILURE_STATUSES:
                    raise APIError(f"Image build {status.lower()} for {image_name}:{image_tag}")
            elif last_status is None:
                _emit(on_build_log, "Waiting for image build to appear")
                last_status = "UNKNOWN"

            _raise_if_timed_out(deadline, image_name, image_tag)
            await asyncio.sleep(max(0.0, poll_interval_seconds))


def _build_context_archive(dockerfile: str) -> bytes:
    data = dockerfile.encode("utf-8")
    fileobj = io.BytesIO()
    with tarfile.open(fileobj=fileobj, mode="w:gz") as tar:
        info = tarfile.TarInfo(PACKAGED_DOCKERFILE_PATH)
        info.size = len(data)
        info.mtime = 0
        tar.addfile(info, io.BytesIO(data))
    return fileobj.getvalue()


def _coerce_non_empty_list(name: str, values: str | Iterable[str]) -> list[str]:
    if isinstance(values, str):
        result = [values]
    else:
        result = [str(value) for value in values]
    if not result:
        raise ValueError(f"{name} must not be empty")
    for value in result:
        _require_single_line(name, value)
        if not value.strip():
            raise ValueError(f"{name} must not contain empty values")
    return [value.strip() for value in result]


def _deadline(timeout_seconds: Optional[float]) -> Optional[float]:
    if timeout_seconds is None or timeout_seconds <= 0:
        return None
    return time.monotonic() + timeout_seconds


def _dockerfile_double_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _effective_team_id(client: APIClient | AsyncAPIClient, team_id: Optional[str]) -> Optional[str]:
    if team_id is not None:
        return team_id
    return client.config.team_id


def _emit(callback: Optional[BuildLogCallback], message: str) -> None:
    if callback is not None:
        callback(message)


def _image_reference_from_row(row: Mapping[str, Any], image_name: str, image_tag: str) -> str:
    return str(
        row.get("displayRef")
        or row.get("fullImagePath")
        or row.get("imageReference")
        or f"{image_name}:{image_tag}"
    )


def _latest_matching_row(
    rows: Any, image_name: str, image_tag: str, team_id: Optional[str]
) -> Optional[dict[str, Any]]:
    if not isinstance(rows, list):
        return None
    matches: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        artifact_type = str(row.get("artifactType") or "CONTAINER_IMAGE")
        if artifact_type != "CONTAINER_IMAGE":
            continue
        if row.get("imageName") != image_name or row.get("imageTag") != image_tag:
            continue
        if team_id and row.get("teamId") not in {None, team_id}:
            continue
        matches.append(row)
    if not matches:
        return None
    return max(matches, key=_row_timestamp)


def _normalize_visibility(visibility: str) -> str:
    normalized = visibility.upper()
    if normalized not in {"PUBLIC", "PRIVATE"}:
        raise ValueError("visibility must be PUBLIC or PRIVATE")
    return normalized


def _parse_ts(value: Any) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _raise_if_timed_out(
    deadline: Optional[float], image_name: str, image_tag: str
) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise APITimeoutError(f"Timed out waiting for image {image_name}:{image_tag} to build")


def _require_single_line(name: str, value: str) -> None:
    if "\n" in value or "\r" in value or "\0" in value:
        raise ValueError(f"{name} must be a single line")


def _resolve_image_name_tag(
    image: Image, image_name: Optional[str], image_tag: Optional[str]
) -> tuple[str, str]:
    resolved_name = image_name or DEFAULT_DECLARATIVE_IMAGE_NAME
    resolved_tag = image_tag
    if ":" in resolved_name and resolved_tag is None:
        resolved_name, resolved_tag = resolved_name.rsplit(":", 1)
    if resolved_tag is None:
        resolved_tag = image.default_tag()

    if "/" in resolved_name or ":" in resolved_name or not _IMAGE_NAME_RE.match(resolved_name):
        raise ValueError(
            "image_name must be a simple registry image name without '/' "
            "or ':' characters"
        )
    if not _IMAGE_TAG_RE.match(resolved_tag):
        raise ValueError("image_tag must be a valid Docker tag")
    return resolved_name, resolved_tag


def _row_timestamp(row: Mapping[str, Any]) -> datetime:
    for key in _LATEST_ROW_KEYS:
        parsed = _parse_ts(row.get(key))
        if parsed != datetime.min.replace(tzinfo=timezone.utc):
            return parsed
    return datetime.min.replace(tzinfo=timezone.utc)


__all__ = [
    "Image",
    "ImageBuildResult",
]
