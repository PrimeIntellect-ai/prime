"""Read a volume's S3 object namespace without creating an SSH session."""

from dataclasses import dataclass
from datetime import datetime


def is_internal(path: str) -> bool:
    path = path.rstrip("/")
    return path == "runs/.sessions" or path.startswith("runs/.sessions/")


@dataclass(frozen=True)
class VolumeObject:
    path: str
    size: int | None
    modified: datetime | None
    directory: bool = False

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "size": self.size,
            "modified": self.modified.isoformat() if self.modified else None,
            "type": "directory" if self.directory else "file",
        }


def _relative(key: str, prefix: str) -> str:
    if not key.startswith(prefix):
        raise ValueError("Storage returned an object outside the volume prefix")
    return key[len(prefix) :]


def _pages(s3, bucket: str, prefix: str, *, recursive: bool):
    options = {"Bucket": bucket, "Prefix": prefix}
    if not recursive:
        options["Delimiter"] = "/"
    yield from s3.get_paginator("list_objects_v2").paginate(**options)


def _objects(s3, bucket: str, prefix: str):
    for page in _pages(s3, bucket, prefix, recursive=True):
        for obj in page.get("Contents", []):
            path = _relative(obj["Key"], prefix)
            if path:
                yield path, obj


def list_entries(
    s3, bucket: str, prefix: str, path: str, *, recursive: bool = False
) -> list[VolumeObject]:
    """List files and directories; markers keep otherwise empty directories visible."""
    from botocore.exceptions import ClientError

    base = path.rstrip("/")
    if base and not path.endswith("/"):
        try:
            head = s3.head_object(Bucket=bucket, Key=prefix + base)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") not in ("404", "NoSuchKey", "NotFound"):
                raise
        else:
            return [VolumeObject(base, head["ContentLength"], head.get("LastModified"))]

    directory = base + "/" if base else ""
    entries: dict[str, VolumeObject] = {}
    exists = not base  # An empty volume still has a root directory.
    for page in _pages(s3, bucket, prefix + directory, recursive=recursive):
        for item in page.get("CommonPrefixes", []):
            rel = _relative(item["Prefix"], prefix)
            exists = True  # Hidden children still establish that their parent exists.
            if is_internal(rel):
                continue
            entries.setdefault(rel, VolumeObject(rel, None, None, True))
        for obj in page.get("Contents", []):
            rel = _relative(obj["Key"], prefix)
            exists = True
            if is_internal(rel):
                continue
            if rel == directory:
                continue  # The requested directory's own marker isn't a child.
            is_dir = rel.endswith("/")
            entries[rel] = VolumeObject(
                rel, None if is_dir else obj["Size"], obj.get("LastModified"), is_dir
            )
            if recursive:
                # Directories need not have explicit S3 marker objects.
                remainder = rel[len(directory) :].rstrip("/")
                parts = remainder.split("/")
                for i in range(1, len(parts)):
                    parent = directory + "/".join(parts[:i]) + "/"
                    entries.setdefault(parent, VolumeObject(parent, None, None, True))
    if not exists:
        raise ValueError(f"No such file or directory on the volume: /{base}")
    return sorted(entries.values(), key=lambda entry: entry.path)


def read_usage(s3, bucket: str, prefix: str, path: str) -> dict:
    """Count visible objects and bytes, retaining only counters across listing pages.

    A single full-volume scan gives both the selected path's size and the
    whole-volume usage that must be compared with the volume's capacity.
    Zero-byte directory markers count as objects, never inferred directories.
    """
    base = path.rstrip("/")
    directory = base + "/" if base else ""
    file_size = None
    selected_bytes = selected_objects = total_bytes = total_objects = 0
    exists = not base
    for rel, obj in _objects(s3, bucket, prefix):
        if rel.startswith(directory):
            exists = True
        if is_internal(rel):
            continue
        size = obj["Size"]
        total_bytes += size
        total_objects += 1
        if base and rel == base and not path.endswith("/"):
            file_size = size
        if rel.startswith(directory):
            selected_bytes += size
            selected_objects += 1
    if file_size is not None:
        selected_bytes, selected_objects, exists = file_size, 1, True
    if not exists:
        raise ValueError(f"No such file or directory on the volume: /{base}")
    return {
        "path": path or "/",
        "bytes": selected_bytes,
        "objects": selected_objects,
        "volumeBytes": total_bytes,
        "volumeObjects": total_objects,
    }
