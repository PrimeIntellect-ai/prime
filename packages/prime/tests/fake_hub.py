"""A fake Hub write path for push tests, and the `hub` fixture (in conftest.py)."""

import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

API = "https://api.test"


class FakeHub:
    """The Hub write path, keeping what each call received."""

    def __init__(self) -> None:
        self.packages: Dict[str, Dict[str, Any]] = {}
        self.calls: List[tuple] = []
        self.releases: List[Dict[str, Any]] = []
        self.uploads: Dict[str, bytes] = {}
        self.finalize_errors: List[tuple] = []
        self.release_error: Optional[tuple] = None
        self.slug: Optional[str] = "alice"
        self.teams = [{"teamId": "c" + "t" * 24, "slug": "research", "name": "Research"}]

    @staticmethod
    def _error(status: int, code: str, message: str) -> httpx.Response:
        return httpx.Response(
            status, json={"error": {"code": code, "message": message, "details": {}}}
        )

    def _package(self, owner: str, name: str) -> Dict[str, Any]:
        return {
            "id": f"pkg-{owner}-{name}",
            "owner": {"kind": "user", "slug": owner},
            "name": name,
            "visibility": "private",
        }

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content) if request.content else None
        self.calls.append((request.method, path, body, dict(request.url.params)))
        if path == "/api/v1/user/whoami":
            return httpx.Response(200, json={"data": {"id": "u1", "slug": self.slug}})
        if path == "/api/v1/user/teams":
            return httpx.Response(200, json={"data": self.teams, "total_count": len(self.teams)})
        if path == "/api/v1/user/slug":
            self.slug = body["slug"]
            return httpx.Response(200, json={"data": {"slug": self.slug}})
        prefix = "/api/v1/hub"
        assert path.startswith(prefix), path
        path = path[len(prefix) :]
        parts = path.strip("/").split("/")
        if request.method == "POST" and path == "/packages":
            key = f"{body['owner']}/{body['name']}"
            if key in self.packages:
                return self._error(409, "package_exists", "exists")
            self.packages[key] = self._package(body["owner"], body["name"])
            if body.get("visibility"):
                self.packages[key]["visibility"] = body["visibility"]
            return httpx.Response(201, json=self.packages[key])
        owner, name = parts[1], parts[2]
        key = f"{owner}/{name}"
        if request.method == "GET" and len(parts) == 3:
            matches = [p for k, p in self.packages.items() if k.lower() == key.lower()]
            if not matches:
                return self._error(404, "package_not_found", "nope")
            return httpx.Response(200, json=matches[0])
        if request.method == "POST" and parts[3:] == ["releases"]:
            if self.release_error:
                return self._error(*self.release_error)
            self.releases.append(body)
            source = next(a for a in body["artifacts"] if a["type"] == "source")
            return httpx.Response(
                201,
                json={
                    "release_id": "rel-1",
                    "version": body["version"],
                    "content_hash": source["sha256"],
                    "uploads": [
                        {
                            "type": a["type"],
                            "filename": a["filename"],
                            "url": f"https://storage.test/{a['filename']}",
                            "headers": {"Content-Type": "application/octet-stream"},
                            "expires_at": "2026-11-20T00:00:00Z",
                        }
                        for a in body["artifacts"]
                    ],
                },
            )
        if request.method == "POST" and parts[-1] == "finalize":
            if self.finalize_errors:
                return self._error(*self.finalize_errors.pop(0))
            release = self.releases[-1]
            return httpx.Response(
                200,
                json={
                    "id": "rel-1",
                    "version": parts[4],
                    "content_hash": request.url.params["content_hash"],
                    "components": [
                        {**c, "source": "declared"} for c in release.get("components", [])
                    ],
                },
            )
        raise AssertionError(f"unexpected {request.method} {path}")


def write_wheel(dist: Path, distribution: str, version: str, requires: List[str]) -> Path:
    dist.mkdir(parents=True, exist_ok=True)
    wheel = dist / f"{distribution.replace('-', '_')}-{version}-py3-none-any.whl"
    metadata = "Metadata-Version: 2.1\nName: {}\nVersion: {}\n".format(distribution, version)
    metadata += "".join(f"Requires-Dist: {r}\n" for r in requires)
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"{distribution.replace('-', '_')}-{version}.dist-info/METADATA", metadata)
    return wheel
