"""The files a package push uploads, and the source archive built from them.

`prime env push` and `prime pipelines push` upload the same thing: the package
folder's own files (README, pyproject.toml, Python sources and any other file
that isn't hidden, a build output or matched by the root .gitignore), as a flat
.tar.gz.
"""

import gzip
import hashlib
import os
import tarfile
from pathlib import Path
from typing import Callable, Dict, List, Optional

from gitignore_parser import parse_gitignore

# Every member of a source archive gets this timestamp, and the gzip header none,
# so the same files always make the same bytes. A retried push then has the same
# hash as the try it retries. 1980-01-02 rather than 0, because a zip (a wheel
# built from the unpacked archive) can't hold a timestamp before 1980.
ARCHIVE_MTIME = 315619200


def should_include_file_in_archive(file_path: Path, base_path: Path) -> bool:
    """Determine if a file should be included in the archive based on filtering rules."""
    if not file_path.is_file():
        return False

    # Skip symlinks - they cause extraction failures in _safe_tar_extract
    if file_path.is_symlink():
        return False

    # Hardlinks are fine to include: _add_file_to_archive stores them as
    # regular file content so pull/install safe-extract never sees LNKTYPE.

    rel_path = file_path.relative_to(base_path)

    # Skip hidden files
    if file_path.name.startswith("."):
        return False

    # Skip files in __pycache__ directories
    if "__pycache__" in rel_path.parts:
        return False

    return True


def should_include_directory_in_archive(dir_path: Path) -> bool:
    """Determine if a directory should be included in the archive based on filtering rules."""
    if not dir_path.is_dir():
        return False

    # Skip directory symlinks, as file symlinks are: one could point anywhere, and its
    # files would be uploaded as the package's own.
    if dir_path.is_symlink():
        return False

    # Skip hidden directories (includes .prime/, .git/, etc.)
    if dir_path.name.startswith("."):
        return False

    # Skip build artifacts, cache directories, and outputs
    if dir_path.name in ["dist", "__pycache__", "build", "outputs"]:
        return False

    # Skip egg-info directories
    if dir_path.name.endswith(".egg-info"):
        return False

    return True


def _build_gitignore_matcher(env_path: Path) -> Optional[Callable[[str], bool]]:
    """Build a matcher for the root .gitignore file, if present."""
    gitignore_path = env_path / ".gitignore"
    if not gitignore_path.exists():
        return None
    return parse_gitignore(str(gitignore_path), base_dir=str(env_path))


def _collect_archive_files(env_path: Path) -> List[Path]:
    """Collect archive file paths in deterministic order, honoring .gitignore."""
    ignore_matcher = _build_gitignore_matcher(env_path)
    files_by_rel_path: Dict[str, Path] = {}

    def maybe_add_file(file_path: Path) -> None:
        if not should_include_file_in_archive(file_path, env_path):
            return
        if ignore_matcher is not None and ignore_matcher(str(file_path)):
            return

        rel_path = str(file_path.relative_to(env_path)).replace("\\", "/")
        files_by_rel_path[rel_path] = file_path

    for pattern in ["README.md", "pyproject.toml", "*.py"]:
        for file_path in sorted(env_path.glob(pattern), key=lambda p: p.name):
            maybe_add_file(file_path)

    def is_nested_dir_ignored(dir_path: Path) -> bool:
        """Check if a nested directory should be pruned from traversal."""
        if not should_include_directory_in_archive(dir_path):
            return True
        if ignore_matcher is not None and ignore_matcher(str(dir_path)):
            return True
        return False

    for subdir in sorted(env_path.iterdir(), key=lambda path: path.name):
        if not should_include_directory_in_archive(subdir):
            continue
        if is_nested_dir_ignored(subdir):
            continue

        for root, dirnames, filenames in os.walk(subdir):
            root_path = Path(root)
            dirnames[:] = sorted(
                dirname for dirname in dirnames if not is_nested_dir_ignored(root_path / dirname)
            )
            for filename in sorted(filenames):
                maybe_add_file(root_path / filename)

    return [files_by_rel_path[rel_path] for rel_path in sorted(files_by_rel_path)]


def _add_file_to_archive(
    tar: tarfile.TarFile, file_path: Path, arcname: str, *, mtime: Optional[int] = None
) -> None:
    """Add a file as REGTYPE content (never symlink/hardlink members).

    ``tar.add()`` records a hardlink (``LNKTYPE``) when another file with the
    same inode is already in the archive. ``_safe_tar_extract`` rejects
    hardlinks, so push must always store byte content as a regular file —
    same rationale as skipping symlinks at collect time.

    With `mtime`, the member also gets that timestamp, no owner, and mode 0755 or
    0644 by the file's executable bit, so the archive is reproducible.
    """
    if file_path.is_symlink():
        return

    info = tar.gettarinfo(str(file_path), arcname=arcname)
    info.type = tarfile.REGTYPE
    info.linkname = ""
    info.size = file_path.stat().st_size
    if mtime is not None:
        info.mtime = mtime
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mode = 0o755 if info.mode & 0o111 else 0o644
    with open(file_path, "rb") as handle:
        tar.addfile(info, handle)


def compute_content_hash(env_path: Path) -> str:
    """Compute deterministic, cross-platform content hash for environment files.

    Args:
        env_path: Path to the environment directory

    Returns:
        SHA256 hexdigest of the environment content
    """
    content_hasher = hashlib.sha256()

    for file_path in _collect_archive_files(env_path):
        normalized_path = str(file_path.relative_to(env_path)).replace("\\", "/")
        content_hasher.update(f"file:{normalized_path}".encode("utf-8"))
        try:
            with open(file_path, "rb") as f:
                content_hasher.update(f.read())
        except IOError:
            # Skip files that can't be read
            pass

    return content_hasher.hexdigest()


def write_source_archive(env_path: Path, destination: Path) -> Path:
    """Write the package's files to `destination` as a reproducible .tar.gz.

    Paths are relative to the package folder, with no top-level folder. Owners,
    timestamps and permissions are normalized, so only the files' names, bytes and
    executable bits decide the archive's sha256.
    """
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for file_path in _collect_archive_files(env_path):
                    arcname = file_path.relative_to(env_path).as_posix()
                    _add_file_to_archive(tar, file_path, arcname, mtime=ARCHIVE_MTIME)
    return destination
