"""Utility functions for sandbox commands."""

import os
import tarfile
import tempfile
from typing import Optional, Tuple

from rich.console import Console

console = Console()


def _format_bytes(num_bytes: int) -> str:
    """Return human-friendly byte size (e.g. 1.2 MB)."""
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(num_bytes)
    unit_idx = 0
    while size >= 1024.0 and unit_idx < len(units) - 1:
        size /= 1024.0
        unit_idx += 1
    if unit_idx == 0:
        return f"{int(size)} {units[unit_idx]}"
    return f"{size:.1f} {units[unit_idx]}"


def _calculate_file_size(file_path: str) -> int:
    """Calculate the size of a file in bytes."""
    try:
        return os.path.getsize(file_path)
    except (OSError, IOError):
        return 0


def _calculate_directory_size(dir_path: str) -> int:
    """Calculate the total size of a directory in bytes."""
    total_size = 0
    try:
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                total_size += _calculate_file_size(file_path)
    except Exception:
        pass
    return total_size


def _should_use_compression(path: str, compress: bool) -> bool:
    """Determine if compression should be used based on file/directory size."""
    if not compress:
        return False

    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        return True  # Default to compression if we can't determine size

    if os.path.isfile(abs_path):
        size = _calculate_file_size(abs_path)
        if size < 100 * 1024 * 1024:  # 100MB
            console.print(
                f"[yellow]File size ({size / 1024 / 1024:.1f}MB) < 100MB, "
                "disabling compression for faster transfer[/yellow]"
            )
            return False
    else:
        size = _calculate_directory_size(abs_path)
        if size < 100 * 1024 * 1024:  # 100MB
            console.print(
                f"[yellow]Directory size ({size / 1024 / 1024:.1f}MB) < 100MB, "
                "disabling compression for faster transfer[/yellow]"
            )
            return False

    return True


def _cleanup_temp_file(file_path: str) -> None:
    """Safely cleanup a temporary file."""
    if not file_path:
        return

    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except Exception:
        pass


def _create_tar_archive(source_path: str, destination_path: str, compress: bool) -> Tuple[str, int]:
    """Create a tar archive from source path and return (temp_path, bytes_written)."""
    src_abs = os.path.abspath(source_path)
    if not os.path.exists(src_abs):
        raise FileNotFoundError(f"Source not found: {src_abs}")

    mode = "w:gz" if compress else "w:"
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        with tarfile.open(tmp_path, mode=mode) as tf:  # type: ignore[call-overload]
            if os.path.isfile(src_abs):
                tf.add(src_abs, arcname=os.path.basename(destination_path))
            else:
                base = os.path.basename(src_abs.rstrip("/"))
                tf.add(src_abs, arcname=base)

        bytes_written = os.path.getsize(tmp_path)
        return tmp_path, bytes_written

    except Exception as e:
        _cleanup_temp_file(tmp_path)
        raise Exception(f"Failed to create tar archive: {e}")


def _extract_tar_archive(
    temp_file_path: str, destination_path: str, compress: bool, source_path: str
) -> None:
    """Extract a tar archive to the destination path."""
    dst_abs = os.path.abspath(destination_path)
    mode = "r:gz" if compress else "r:"

    try:
        with tarfile.open(temp_file_path, mode=mode) as tf:
            members = list(tf.getmembers())

            if not members:
                raise Exception(
                    f"Tar archive is empty. Source path '{source_path}' "
                    "may not exist in the sandbox"
                )

            # Detect content type from tar members
            is_file = len(members) == 1 and members[0].isfile()

            if is_file:
                # Handle single file extraction
                if os.path.isdir(dst_abs):
                    os.rmdir(dst_abs)

                for member in members:
                    if member.isfile():
                        with open(dst_abs, "wb") as f:
                            f.write(tf.extractfile(member).read())
                        break
                else:
                    raise Exception("No file found in archive")
            else:
                # Handle directory extraction
                for member in members:
                    if "/" in member.name:
                        member.name = "/".join(member.name.split("/")[1:])
                    tf.extract(member, path=dst_abs)

    except tarfile.ReadError as e:
        raise Exception(
            f"Failed to read tar archive: {e}. "
            "This might be due to corrupted data from the sandbox."
        )
    except Exception as e:
        raise Exception(f"Failed to extract tar archive: {e}")


def _parse_cp_arg(arg: str) -> Tuple[Optional[str], str]:
    """Parse cp-style arg: either "<sandbox-id>:<path>" or local path.

    Returns (sandbox_id, path). sandbox_id is None if local.
    """
    if ":" in arg and not arg.startswith(":"):
        sandbox_id, path = arg.split(":", 1)
        if sandbox_id:
            return sandbox_id, path
    return None, arg


def _expand_home_in_path(path: str) -> str:
    """
    Safely expand $HOME to /sandbox-workspace in paths.
    Only allows $HOME expansion for security - no other environment variables.
    """
    if "$HOME" in path:
        # Replace $HOME with the sandbox workspace path
        expanded_path = path.replace("$HOME", "/sandbox-workspace")
        return expanded_path
    return path
