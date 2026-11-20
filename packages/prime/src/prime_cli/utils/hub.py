"""Publishing a package through the Hub API (`/api/v1/hub`).

A release is published in three calls: start it (`POST .../releases`, declaring
each artifact's sha256 and size), PUT each artifact to the link that returns, and
finalize it (`POST .../finalize`), which checks the uploads against what was
declared. `prime env push` and `prime pipelines push` both publish this way.

Hub errors have one shape, `{"error": {"code", "message", "details"}}`, so callers
branch on `HubAPIError.code` rather than on message text.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx

from ..client import APIClient, APIError

CHUNK_BYTES = 1024 * 1024
UPLOAD_TIMEOUT = httpx.Timeout(600.0, connect=10.0)


class HubAPIError(APIError):
    """An error the Hub API returned, with its machine-readable code."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(f"HTTP {status_code} {code}: {message}")
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}


def _hub_error(response: httpx.Response) -> HubAPIError:
    try:
        error = response.json()["error"]
        return HubAPIError(
            response.status_code,
            str(error["code"]),
            str(error["message"]),
            error.get("details") or {},
        )
    except (ValueError, KeyError, TypeError):
        return HubAPIError(
            response.status_code,
            "http_error",
            response.text.strip() or response.reason_phrase,
        )


class HubClient:
    """Calls `/api/v1/hub` with the CLI's API key, keeping the Hub's error codes."""

    def __init__(self, api: APIClient):
        self.api = api

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.api._check_auth_required()
        url = f"{self.api.base_url}/api/v1/hub{path}"
        try:
            response = self.api.client.request(method, url, params=params, json=json)
        except httpx.TimeoutException as e:
            raise APIError(f"Request timed out: {e}") from e
        except httpx.RequestError as e:
            raise APIError(f"Request failed: {e.__class__.__name__} at {method} {url}: {e}") from e
        if not response.is_success:
            raise _hub_error(response)
        result = response.json()
        if not isinstance(result, dict):
            raise APIError("Hub response was not a JSON object")
        return result

    def get_package(self, owner: str, name: str) -> Optional[Dict[str, Any]]:
        """The package, or None if there is none by that name that you can read."""
        try:
            return self.request("GET", f"/packages/{owner}/{name}")
        except HubAPIError as e:
            if e.status_code == 404:
                return None
            raise


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def declare_artifact(kind: str, path: Path) -> Dict[str, Any]:
    return {
        "type": kind,
        "filename": path.name,
        "sha256": sha256_of(path),
        "size": path.stat().st_size,
    }


def upload_artifact(upload: Dict[str, Any], path: Path) -> None:
    """PUT one artifact to its signed link, streaming the file."""
    with path.open("rb") as file:
        response = httpx.put(
            upload["url"],
            content=file,
            headers=upload.get("headers") or {},
            timeout=UPLOAD_TIMEOUT,
        )
    if not response.is_success:
        raise APIError(
            f"Uploading {path.name} failed: HTTP {response.status_code} {response.text[:200]}"
        )


@dataclass
class PublishedRelease:
    package: Dict[str, Any]
    release: Dict[str, Any]
    created_package: bool
    owner: str
    name: str


@dataclass
class ReleaseRequest:
    """What a push declares about the release it starts."""

    version: str
    source: Path
    wheel: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    framework: Optional[str] = None
    components: List[Dict[str, Any]] = field(default_factory=list)


def find_package(hub: HubClient, owner: str, names: List[str]) -> Optional[Dict[str, Any]]:
    """The first of `names` that `owner` has a package by.

    The Hub matches names case-insensitively but not `_` against `-`, so a
    package created before names were normalized is still found by its own
    spelling.
    """
    for name in dict.fromkeys(names):
        package = hub.get_package(owner, name)
        if package is not None:
            return package
    return None


def publish_release(
    hub: HubClient,
    *,
    owner: str,
    names: List[str],
    request: ReleaseRequest,
    visibility: Optional[str] = None,
    upload: Callable[[Dict[str, Any], Path], None] = upload_artifact,
    on_step: Callable[[str], None] = lambda _message: None,
) -> PublishedRelease:
    """Create the package if it's new, then start, upload and finalize a release.

    `names` lists the package's name as the Hub may know it, most likely first;
    a new package takes the last, which must be the normalized form. The lookup
    comes first because a collaborator with write access can publish to a package
    without being allowed to create one under its owner.
    """
    package = find_package(hub, owner, names)
    created = False
    if package is None:
        name = names[-1]
        body: Dict[str, Any] = {"owner": owner, "name": name}
        if visibility:
            body["visibility"] = visibility
        description = request.metadata.get("description")
        if description:
            body["description"] = description
        on_step(f"Creating {owner}/{name}...")
        try:
            package = hub.request("POST", "/packages", json=body)
            created = True
        except HubAPIError as e:
            # Created by a concurrent push since the lookup.
            if e.code != "package_exists":
                raise
            package = hub.get_package(owner, name)
            if package is None:
                raise
    name = package["name"]

    paths = {"source": request.source}
    if request.wheel is not None:
        paths["wheel"] = request.wheel
    release_body: Dict[str, Any] = {
        "version": request.version,
        "artifacts": [declare_artifact(kind, path) for kind, path in paths.items()],
        "metadata": request.metadata,
        "components": request.components,
    }
    if request.framework:
        release_body["framework"] = request.framework

    on_step(f"Starting release {request.version}...")
    started = hub.request("POST", f"/packages/{owner}/{name}/releases", json=release_body)

    def upload_all() -> None:
        for item in started["uploads"]:
            on_step(f"Uploading {item['filename']}...")
            upload(item, paths[item["type"]])

    def finalize() -> Dict[str, Any]:
        return hub.request(
            "POST",
            f"/packages/{owner}/{name}/releases/{started['version']}/finalize",
            params={"content_hash": started["content_hash"]},
        )

    upload_all()
    on_step("Finalizing...")
    try:
        release = finalize()
    except HubAPIError as e:
        # Storage can take a moment to show an upload; send them again once.
        if e.code != "artifact_missing":
            raise
        upload_all()
        release = finalize()
    return PublishedRelease(
        package=package,
        release=release,
        created_package=created,
        owner=owner,
        name=name,
    )
