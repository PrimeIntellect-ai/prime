"""Prime-owned environment metadata and Hub package operations."""

import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from gitignore_parser import parse_gitignore


def normalize_package_name(name: str) -> str:
    """Return the importable spelling used by environment packages."""
    return name.replace("-", "_").lower()


def parse_env_id(env_id: str) -> tuple[str, str, str | None]:
    """Parse a Prime Hub reference into owner, name, and optional version."""
    version = None
    if "@" in env_id:
        env_id, version = env_id.rsplit("@", 1)

    parts = env_id.split("/")
    valid_parts = len(parts) == 2 and all(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", part) for part in parts
    )
    valid_version = version is None or bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+_-]*", version))
    if not valid_parts or not valid_version:
        raise ValueError(
            f"Invalid environment ID: {env_id!r}. Expected 'owner/name' or 'owner/name@version'."
        )
    return parts[0], parts[1], version


def _safe_extract(tar: tarfile.TarFile, destination: Path) -> None:
    """Extract a source archive without links, absolute paths, or traversal."""
    destination = destination.resolve()
    for member in tar.getmembers():
        path = Path(member.name)
        if member.issym() or member.islnk():
            raise ValueError(f"Refusing to extract archive link: {member.name}")
        if not member.isfile() and not member.isdir():
            raise ValueError(f"Refusing to extract special archive entry: {member.name}")
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Refusing to extract unsafe path: {member.name}")
        if not (destination / path).resolve().is_relative_to(destination):
            raise ValueError(f"Archive path escapes destination: {member.name}")
    tar.extractall(destination, filter="data")


def download_environment_source(
    details: dict[str, Any],
    destination: Path,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Path:
    """Download and safely extract a Hub source package."""
    package_url = details.get("tracked_package_url") or details.get("package_url")
    parsed = urlparse(package_url or "")
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Environment has no valid downloadable source package")
    assert package_url is not None

    api_host = urlparse(base_url or "").netloc
    headers = (
        {"Authorization": f"Bearer {api_key}"} if api_key and parsed.netloc == api_host else {}
    )
    destination = destination.expanduser().absolute()
    if destination.is_symlink() or destination == Path(destination.anchor):
        raise ValueError(f"Unsafe extraction destination: {destination}")

    with tempfile.TemporaryDirectory(prefix="prime-env-source-") as directory:
        archive = Path(directory) / "environment.tar.gz"
        extracted = Path(directory) / "extracted"
        extracted.mkdir()
        with httpx.stream(
            "GET", package_url, headers=headers, timeout=60.0, follow_redirects=True
        ) as response:
            response.raise_for_status()
            with archive.open("wb") as handle:
                for chunk in response.iter_bytes(chunk_size=8192):
                    handle.write(chunk)
        with tarfile.open(archive, "r:gz") as tar:
            _safe_extract(tar, extracted)

        entries = list(extracted.iterdir())
        source = extracted
        if (
            not (extracted / "pyproject.toml").is_file()
            and len(entries) == 1
            and entries[0].is_dir()
        ):
            source = entries[0]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination, dirs_exist_ok=True)
    return destination


def _build_environment_wheel(source: Path, cache_dir: Path) -> Path:
    with tempfile.TemporaryDirectory(prefix="prime-env-wheel-") as directory:
        dist = Path(directory) / "dist"
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(dist)],
            cwd=source,
            check=True,
        )
        wheels = list(dist.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"Expected one built wheel for {source}, found {len(wheels)}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / wheels[0].name
        temporary = cache_dir / f".{wheels[0].name}.{os.getpid()}"
        shutil.copy2(wheels[0], temporary)
        temporary.replace(cached)
    return cached


def install_environment_from_hub(
    env_id: str,
    details: dict[str, Any],
    *,
    python_executable: str,
    api_key: str | None = None,
    base_url: str | None = None,
    prerelease: bool = False,
) -> str:
    """Install resolved Hub details and return the local importable module name."""
    owner, name, version = parse_env_id(env_id)
    simple_index_url = details.get("simple_index_url")
    wheel_url = details.get("wheel_url")
    url_dependencies = details.get("url_dependencies") or []
    package = normalize_package_name(name)
    pinned_version = version if version not in (None, "latest") else None
    command = [
        "uv",
        "pip",
        "install",
        "--python",
        python_executable,
        "-P",
        package,
    ]

    if isinstance(simple_index_url, str) and simple_index_url:
        command.append(f"{package}=={pinned_version}" if pinned_version else package)
        command.extend(str(dependency) for dependency in url_dependencies)
        command.extend(["--extra-index-url", simple_index_url])
    elif isinstance(wheel_url, str) and wheel_url:
        parsed = urlparse(wheel_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"Invalid wheel URL for {env_id!r}: {wheel_url}")
        command.append(wheel_url)
        command.extend(str(dependency) for dependency in url_dependencies)
    else:
        version_data = details.get("latest_version")
        content_hash = details.get("content_hash") or details.get("sha256")
        if not content_hash and isinstance(version_data, dict):
            content_hash = version_data.get("content_hash") or version_data.get("sha256")
        valid_hash = (
            isinstance(content_hash, str)
            and len(content_hash) == 64
            and all(character in "0123456789abcdef" for character in content_hash.lower())
        )
        cache_root = Path.home() / ".cache" / "prime" / "environments" / owner / name
        cache_dir = cache_root / str(content_hash) if valid_hash else None
        wheels = list(cache_dir.glob("*.whl")) if cache_dir else []
        if wheels:
            wheel = wheels[0]
        else:
            with tempfile.TemporaryDirectory(prefix="prime-env-build-") as directory:
                source = download_environment_source(
                    details,
                    Path(directory) / "source",
                    api_key=api_key,
                    base_url=base_url,
                )
                if cache_dir is None:
                    digest = hashlib.sha256()
                    for path in sorted(item for item in source.rglob("*") if item.is_file()):
                        digest.update(path.relative_to(source).as_posix().encode())
                        digest.update(b"\0")
                        digest.update(path.read_bytes())
                    cache_dir = cache_root / digest.hexdigest()
                cached = list(cache_dir.glob("*.whl"))
                wheel = cached[0] if cached else _build_environment_wheel(source, cache_dir)
        command.append(str(wheel))

    if prerelease:
        command.append("--prerelease=allow")
    subprocess.run(command, check=True)
    importlib.invalidate_caches()
    return package


def collect_archive_files(env_path: Path) -> List[Path]:
    """Collect source files deterministically while honoring root .gitignore."""
    gitignore = env_path / ".gitignore"
    ignore: Optional[Callable[[str], bool]] = (
        parse_gitignore(str(gitignore), base_dir=str(env_path)) if gitignore.exists() else None
    )
    files: Dict[str, Path] = {}

    def add(path: Path) -> None:
        if not path.is_file() or path.is_symlink():
            return
        relative = path.relative_to(env_path)
        if path.name.startswith(".") and relative != Path("proj/.build.json"):
            return
        if "__pycache__" in relative.parts:
            return
        if relative != Path("proj/.build.json") and ignore and ignore(str(path)):
            return
        files[relative.as_posix()] = path

    for pattern in ["README.md", "pyproject.toml", "*.py"]:
        for path in sorted(env_path.glob(pattern), key=lambda item: item.name):
            add(path)
    add(env_path / "proj" / ".build.json")

    for subdir in sorted(env_path.iterdir(), key=lambda item: item.name):
        if not subdir.is_dir() or subdir.name.startswith("."):
            continue
        if subdir.name in {"dist", "__pycache__", "build", "outputs"}:
            continue
        if subdir.name.endswith(".egg-info") or (ignore and ignore(str(subdir))):
            continue
        for root, dirnames, filenames in os.walk(subdir):
            root_path = Path(root)
            dirnames[:] = sorted(
                name
                for name in dirnames
                if not name.startswith(".")
                and name not in {"dist", "__pycache__", "build", "outputs"}
                and not name.endswith(".egg-info")
                and not (ignore and ignore(str(root_path / name)))
            )
            for filename in sorted(filenames):
                add(root_path / filename)
    return [files[path] for path in sorted(files)]


def compute_content_hash(env_path: Path) -> str:
    """Compute a deterministic hash of an environment's published source files."""
    digest = hashlib.sha256()
    for path in collect_archive_files(env_path):
        digest.update(f"file:{path.relative_to(env_path).as_posix()}".encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def get_environment_metadata(env_path: Path) -> Optional[Dict[str, Any]]:
    """Read Prime-owned metadata from ``.prime/.env-metadata.json``."""
    metadata_path = env_path / ".prime" / ".env-metadata.json"
    if not metadata_path.exists():
        return None

    try:
        with open(metadata_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def find_environment_metadata(
    env_name: Optional[str] = None,
    env_path: Optional[Path] = None,
    module_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Search for environment metadata in multiple common locations.

    Searches directories in the following order:
    1. env_path (if provided)
    2. ./environments/{module_name} (if module_name provided)
    3. ./environments/{env_name} (if env_name provided)
    4. ./{env_name} (if env_name provided)
    5. ./{module_name} (if module_name provided)
    6. ./ (current directory)

    Args:
        env_name: Environment name (e.g., "simpleqa")
        env_path: Optional explicit path to check first
        module_name: Optional module name (e.g., "simple_qa" for env_name "simpleqa")

    Returns:
        Dictionary containing environment metadata from the first location found, or None
    """
    possible_env_dirs = []

    # 1. Check explicit path first if provided
    if env_path:
        possible_env_dirs.append(env_path)

    # 2-5. Check common locations based on provided names
    if module_name:
        possible_env_dirs.append(Path("./environments") / module_name)
    if env_name:
        possible_env_dirs.append(Path("./environments") / env_name)
        possible_env_dirs.append(Path(".") / env_name)
    if module_name:
        possible_env_dirs.append(Path(".") / module_name)

    # 6. Check current directory last
    possible_env_dirs.append(Path("."))

    # Search through directories and return first match
    for env_dir in possible_env_dirs:
        metadata = get_environment_metadata(env_dir)
        if metadata:
            return metadata

    return None
