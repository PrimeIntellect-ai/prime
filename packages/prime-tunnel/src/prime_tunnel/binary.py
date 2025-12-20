import hashlib
import platform
import shutil
import stat
import tarfile
import tempfile
from pathlib import Path

import httpx

from prime_tunnel.core.config import Config
from prime_tunnel.exceptions import BinaryDownloadError

FRPC_VERSION = "0.65.0"
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
FRPC_CHECKSUMS = {
    ("Darwin", "arm64"): None,  # TODO: Add checksums
    ("Darwin", "x86_64"): None,
    ("Linux", "x86_64"): None,
    ("Linux", "aarch64"): None,
    ("Linux", "arm64"): None,
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

    expected_checksum = FRPC_CHECKSUMS.get(platform_key)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        archive_path = tmpdir_path / "frp.tar.gz"

        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
                response.raise_for_status()
                downloaded = 0

                with open(archive_path, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)

        except httpx.HTTPError as e:
            raise BinaryDownloadError(f"Failed to download frpc: {e}") from e

        if expected_checksum:
            actual_checksum = _compute_sha256(archive_path)
            if actual_checksum != expected_checksum:
                raise BinaryDownloadError(
                    f"Checksum mismatch: expected {expected_checksum}, got {actual_checksum}"
                )

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
        shutil.copy2(extracted_path, dest)
        dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _compute_sha256(path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


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
    version_file.write_text(FRPC_VERSION)

    return frpc_path
