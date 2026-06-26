"""Utilities for reading and managing environment metadata."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from gitignore_parser import parse_gitignore


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
