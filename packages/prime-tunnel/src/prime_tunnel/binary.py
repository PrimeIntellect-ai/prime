import os
import platform
import shutil
import stat
import tarfile
import tempfile
from pathlib import Path

import httpx

from prime_tunnel.core.config import Config
from prime_tunnel.exceptions import BinaryDownloadError

FRPC_VERSION = "0.66.0"
FRPC_URLS = {
    (
        "Darwin",
        "arm64",
    ): f"https://github.com/fatedier/frp/releases/download/v{FRPC_VERSION}/frp_{FRPC_VERSION}_darwin_arm64.tar.gz",
    (
        "Darwin",
        "x86_64",
    ): f"https://github.com/fatedier/frp/releases/download/v{FRPC_VERSION}/frp_{FRPC_VERSION}_darwin_amd64.tar.gz",
    (
        "Linux",
        "x86_64",
    ): f"https://github.com/fatedier/frp/releases/download/v{FRPC_VERSION}/frp_{FRPC_VERSION}_linux_amd64.tar.gz",
    (
        "Linux",
        "aarch64",
    ): f"https://github.com/fatedier/frp/releases/download/v{FRPC_VERSION}/frp_{FRPC_VERSION}_linux_arm64.tar.gz",
    (
        "Linux",
        "arm64",
    ): f"https://github.com/fatedier/frp/releases/download/v{FRPC_VERSION}/frp_{FRPC_VERSION}_linux_arm64.tar.gz",
}


def _get_platform_key() -> tuple[str, str]:
    system = platform.system()
    machine = platform.machine()

    if machine in ("AMD64", "x86_64"):
        machine = "x86_64"
    elif machine in ("arm64", "aarch64"):
        machine = "arm64" if system == "Darwin" else "aarch64"

    return (system, machine)


def _download_frpc(dest: Path) -> None:
    platform_key = _get_platform_key()
    url = FRPC_URLS.get(platform_key)

    if not url:
        raise BinaryDownloadError(f"Unsupported platform: {platform_key[0]} {platform_key[1]}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        archive_path = tmpdir_path / "frp.tar.gz"

        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
                response.raise_for_status()
                with open(archive_path, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=8192):
                        f.write(chunk)

        except httpx.HTTPError as e:
            raise BinaryDownloadError(f"Failed to download frpc: {e}") from e

        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.endswith("/frpc") or member.name == "frpc":
                        member.name = "frpc"
                        tar.extract(member, tmpdir_path)
                        break
                else:
                    raise BinaryDownloadError("frpc binary not found in archive")

        except tarfile.TarError as e:
            raise BinaryDownloadError(f"Failed to extract frpc: {e}") from e

        extracted_path = tmpdir_path / "frpc"
        if not extracted_path.exists():
            raise BinaryDownloadError("frpc binary not found after extraction")

        dest.parent.mkdir(parents=True, exist_ok=True)

        # Set executable permissions on extracted file before moving
        extracted_path.chmod(
            extracted_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        )

        # Copy to temp file in same directory, then rename
        # This prevents corruption if multiple processes download simultaneously
        temp_dest = dest.parent / f".frpc.{os.getpid()}.tmp"
        try:
            shutil.copy2(extracted_path, temp_dest)
            os.replace(temp_dest, dest)  # Atomic on POSIX
        finally:
            # Clean up temp file if rename failed
            if temp_dest.exists():
                temp_dest.unlink()


def get_frpc_path() -> Path:
    config = Config()
    frpc_path = config.bin_dir / "frpc"
    version_file = config.bin_dir / ".frpc_version"

    if frpc_path.exists():
        if version_file.exists():
            current_version = version_file.read_text().strip()
            if current_version == FRPC_VERSION:
                return frpc_path

    _download_frpc(frpc_path)

    # Write to temp file in same directory, then rename to prevent partial reads
    temp_version = version_file.parent / f".frpc_version.{os.getpid()}.tmp"
    try:
        temp_version.write_text(FRPC_VERSION)
        os.replace(temp_version, version_file)
    finally:
        if temp_version.exists():
            temp_version.unlink()

    return frpc_path
