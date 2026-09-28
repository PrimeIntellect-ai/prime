"""In-pod staging script for ``prime volumes stage``.

This module is not used by the CLI at runtime. ``prime volumes stage``
copies its source into a short-lived CPU pod (mounted volume at
``/volume``) and runs it with the pinned prime-rl image's Python:

    /app/.venv/bin/python -c <this source> stage --source <repo> ...

The same source doubles as the offline verifier, re-executed by itself in
a *fresh* interpreter with an empty HF/datasets cache, ``HF_HUB_OFFLINE=1``
and ``HF_DATASETS_OFFLINE=1`` and no token, so a cached ``load_dataset``
in the downloading process never stands in for verification.

Phases (``stage``): resolve revision -> download snapshot -> offline
fresh-process verify -> publish with ``renameat2(RENAME_NOREPLACE)``.

    datasets/<name>/           <- published, immutable (never replaced)
      .prime-stage-manifest    <- hidden JSON, no data-file extension
    datasets/.prime-stage-<op>/[-cache/]  <- operation-owned scratch

Tests import this module and drive the phases directly against tiny local
snapshots; it deliberately depends only on the standard library at import
time (``datasets``/``huggingface_hub`` are imported lazily inside the
phases that need them).
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import Any, Optional, cast

# The CLI tail()s container logs and looks for this marker; it also
# requires exit code 0. Neither alone is success.
RESULT_MARKER = "PRIME_STAGE_RESULT:"

MANIFEST_NAME = ".prime-stage-manifest"
MANIFEST_SCHEMA_VERSION = 1

# Data-only, locally loadable repositories: no loader scripts, no
# save_to_disk Arrow caches, no streaming-only shards.
DATA_EXTENSIONS = (".parquet", ".json", ".jsonl")

SAVE_TO_DISK_MARKERS = ("state.json", "dataset_info.json")

_TOKEN_ENV_KEYS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGING_FACE_API_TOKEN")

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
# Fallback for libc without a renameat2 symbol (musl/old glibc).
_RENAMEAT2_SYSCALL_NR = {"x86_64": 316, "aarch64": 276, "arm64": 276}


class StageError(Exception):
    """A clean, expected failure: message is printed verbatim by the CLI."""


class UnsupportedRenameError(StageError):
    """The target filesystem does not support renameat2(RENAME_NOREPLACE).

    Live-verified on the production ceph-filesystem (CephFS) volumes:
    renameat2 with RENAME_NOREPLACE returns EINVAL, so publication uses a
    locked no-overwrite rename fallback (see publish_no_replace)."""


def _progress(message: str) -> None:
    print(message, flush=True)


def _fail(message: str) -> "StageError":
    return StageError(message)


def _emit(payload: dict[str, Any]) -> None:
    print(RESULT_MARKER + json.dumps(payload, default=str), flush=True)


def _library_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    for dist in ("datasets", "huggingface_hub", "pyarrow"):
        try:
            versions[dist] = version(dist)
        except PackageNotFoundError:  # pragma: no cover - image always has them
            versions[dist] = "unknown"
    return versions


def _self_source() -> str:
    """The full source of this module, for re-exec in a fresh interpreter."""
    explicit = os.environ.get("PRIME_STAGE_SCRIPT")
    if explicit:
        return explicit
    import inspect

    import prime_cli.commands.volumes_stage_script as module

    return inspect.getsource(module)


def _rename_noreplace(src: Path, dst: Path) -> None:
    """Atomic, non-overwriting directory rename via renameat2.

    Fails safely (never falls back to os.replace/os.rename, which can
    overwrite a concurrent publisher's directory or an existing dataset).
    """
    libc = ctypes.CDLL(None, use_errno=True)
    csrc = os.fsencode(str(src))
    cdst = os.fsencode(str(dst))
    func = getattr(libc, "renameat2", None)
    if func is not None:
        func.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        func.restype = ctypes.c_int
        result = func(_AT_FDCWD, csrc, _AT_FDCWD, cdst, _RENAME_NOREPLACE)
    else:
        nr = _RENAMEAT2_SYSCALL_NR.get(platform.machine())
        if nr is None:
            raise UnsupportedRenameError(
                f"no renameat2 available on {platform.machine()}; "
                "cannot publish without an atomic no-replace rename"
            )
        libc.syscall.restype = ctypes.c_long
        result = libc.syscall(
            ctypes.c_long(nr),
            ctypes.c_int(_AT_FDCWD),
            ctypes.c_char_p(csrc),
            ctypes.c_int(_AT_FDCWD),
            ctypes.c_char_p(cdst),
            ctypes.c_uint(_RENAME_NOREPLACE),
        )
    if result == 0:
        return
    err = ctypes.get_errno()
    if err == errno.EEXIST:
        raise FileExistsError(err, os.strerror(err), str(src), None, str(dst))
    if err in (errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP, errno.ENOTSUP):
        raise UnsupportedRenameError(
            f"filesystem does not support renameat2(RENAME_NOREPLACE) "
            f"(errno {err}, {os.strerror(err)}); refusing to publish with a "
            "rename that could overwrite existing data"
        )
    raise OSError(err, os.strerror(err), str(src), None, str(dst))


def _publish_lock_path(datasets_root: Path) -> Path:
    return datasets_root / ".prime-stage-publish.lock"


def publish_no_replace(src: Path, dst: Path) -> str:
    """Publish `src` as `dst` without ever overwriting existing data.

    Preferred: renameat2(RENAME_NOREPLACE) - atomic no-replace. The
    production CephFS (ceph-filesystem RWX) volumes do not implement
    renameat2 flags (EINVAL), so for those filesystems fall back to a
    locked reservation + directory rename:

    - an flock on datasets/.prime-stage-publish.lock serializes staging
      operations targeting the same datasets/ root. A lock failure is a
      HARD failure: publication aborts without any rename, because an
      unlocked check-then-rename could overwrite a racing writer.
    - under the lock, an empty destination marker directory is created
      atomically (os.mkdir fails EEXIST if anything - including an
      unmanaged manual directory - already occupies the destination).
    - the candidate directory is renamed onto OUR OWN empty marker: a
      same-filesystem atomic replace of an empty, operation-owned
      directory. A foreign/managed destination is never the rename
      target; anything that appears between reservation and rename
      surfaces as EEXIST/ENOTEMPTY and is treated as a conflict.

    Returns the mechanism used ("renameat2" or "locked_rename").
    """
    try:
        _rename_noreplace(src, dst)
        return "renameat2"
    except UnsupportedRenameError:
        pass
    import fcntl

    lock_path = _publish_lock_path(dst.parent)
    lock_file = None
    have_lock = False
    try:
        lock_file = open(lock_path, "a")
        try:
            os.chmod(lock_path, 0o644)
        except OSError:
            pass
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except OSError as exc:
            # Hard fail-closed: without the advisory lock the check-then-
            # rename sequence below could overwrite a racing publisher,
            # so publication aborts entirely. Nothing was published.
            raise StageError(
                f"publisher lock unavailable on this filesystem ({exc}); "
                "refusing to publish without an overwrite guarantee"
            ) from exc
        have_lock = True
        if dst.exists() or dst.is_symlink():
            raise FileExistsError(errno.EEXIST, "exists", str(src), None, str(dst))
        reserved = False
        try:
            os.mkdir(dst)  # atomic reservation; EEXIST if already occupied
            reserved = True
            os.rename(src, dst)  # replaces only OUR empty marker
        except BaseException as exc:
            if reserved:
                # Cancellation (Ctrl+C, the SIGTERM handler) or a filesystem
                # error in the reservation window: remove the marker so a
                # retry is not permanently blocked by an empty final dir.
                # Only our own still-empty marker is removed: os.rmdir
                # refuses non-empty directories, so anything that raced
                # into the marker survives untouched.
                try:
                    os.rmdir(dst)
                except OSError:
                    pass
            if isinstance(exc, OSError) and exc.errno in (errno.EEXIST, errno.ENOTEMPTY):
                raise FileExistsError(errno.EEXIST, "exists", str(src), None, str(dst)) from exc
            raise
        return "locked_rename"
    finally:
        if lock_file is not None:
            try:
                if have_lock:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            finally:
                lock_file.close()


def _inventory_problems(root: Path, manifest: dict[str, Any]) -> list[str]:
    """The published tree must still match the manifest inventory exactly:
    every recorded file present with the recorded size, nothing unlisted."""
    problems: list[str] = []
    recorded: dict[str, int] = {}
    for entry in manifest.get("files") or []:
        if not isinstance(entry, dict) or "path" not in entry:
            continue
        recorded[str(entry["path"])] = int(entry.get("bytes", -1))
    for rel, expected_bytes in recorded.items():
        parts = PurePosixPath(rel).parts
        if not rel or rel.startswith(("/", "..", "\\")) or ".." in parts:
            problems.append(f"manifest inventory entry escapes the dataset root: {rel}")
            continue
        path = root / rel
        if path.is_symlink() or not path.is_file():
            problems.append(f"manifest inventory entry is missing (or a symlink): {rel}")
        elif path.stat().st_size != expected_bytes:
            problems.append(
                f"manifest inventory entry has changed size: {rel} "
                f"(expected {expected_bytes} bytes)"
            )
    for path in root.rglob("*"):
        if (
            path.is_file()
            and not path.is_symlink()
            and path.name != MANIFEST_NAME
            and path.relative_to(root).as_posix() not in recorded
        ):
            problems.append(
                f"dataset file is not in the stage manifest: {path.relative_to(root).as_posix()}"
            )
    return problems


def _existing_dataset_problems(final: Path, name: str) -> list[str]:
    """Shared containment validator for existing staged directories, used
    by BOTH the idempotent re-stage path and the concurrent-winner branch:
    a symlink destination is never followed, adopted, or replaced."""
    if final.is_symlink():
        return [
            f"destination datasets/{name} is a symlink; refusing to stage "
            "onto (or through) a symlink destination"
        ]
    return []


def _front_matter_block(text: str) -> Optional[str]:
    """The YAML metadata block of an HF dataset card, if present."""
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end < 0:
        end = text.find("\r\n---", 3)
    return text[3:end] if end >= 0 else text[3:]


def _iter_data_file_paths(data_files: Any) -> list[str]:
    """Normalize every supported data_files spelling to a list of path
    strings: a bare scalar, a flow list, a block list of scalars or
    {split: ..., path: ...} mappings, {split: path} dicts, and multi-line
    (folded) strings are all normal YAML, so this walks the parsed
    structure instead of scanning lines."""
    paths: list[str] = []
    if data_files is None:
        return paths
    if isinstance(data_files, str):
        return [data_files]
    if isinstance(data_files, list):
        for item in data_files:
            paths.extend(_iter_data_file_paths(item))
        return paths
    if isinstance(data_files, dict):
        for key, value in data_files.items():
            paths.extend(_iter_data_file_paths(value))
        return paths
    return paths


def _data_file_problems(root: Path, path_str: str) -> list[str]:
    value = str(path_str).strip()
    if not value:
        return []
    if value.startswith(("http://", "https://", "file://", "ftp://", "s3://", "gs://")):
        return [f"data_files references the external URL {value!r}"]
    if value.startswith(("/", "~")) or "\\" in value:
        return [f"data_files references {value!r} outside the dataset snapshot"]
    if ".." in PurePosixPath(value.replace("\\", "/")).parts:
        return [f"data_files references {value!r} outside the dataset snapshot"]
    root_resolved = root.resolve()
    resolved = (root / value).resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        return [f"data_files path {value!r} resolves outside the dataset snapshot"]
    if not (root / value.split("*")[0]).exists() and "*" not in value:
        # Missing shard files fail in the fresh load anyway; external
        # references are the containment problem handled here.
        pass
    return []


def check_metadata_references(root: Path) -> list[str]:
    """Reject dataset card (README) configs that point data_files at
    anything outside the snapshot: the card is parsed as YAML (all normal
    spellings - scalars, flow lists, folded strings - are structural), and
    every resolved file path must stay inside the snapshot root. HF
    offline mode blocks the network, not other local paths."""
    readme = root / "README.md"
    if not readme.is_file():
        return []
    try:
        text = readme.read_text()
    except OSError:
        return []
    block = _front_matter_block(text)
    if block is None:
        return []
    try:
        import yaml
    except ImportError:  # pragma: no cover - huggingface_hub depends on pyyaml
        return ["cannot validate the dataset card (pyyaml unavailable)"]
    try:
        metadata = yaml.safe_load(block)
    except Exception as exc:  # noqa: BLE001 - any card parse problem fails closed
        return [f"dataset card metadata is not valid YAML: {exc}"]
    if metadata is None:
        return []
    if not isinstance(metadata, dict):
        return ["dataset card metadata block is not a mapping"]
    configs = metadata.get("configs")
    if configs is None:
        return []
    if not isinstance(configs, list):
        return ["dataset card 'configs' is not a list"]
    problems: list[str] = []
    for config in configs:
        if not isinstance(config, dict):
            problems.append("dataset card config entry is not a mapping")
            continue
        for path_str in _iter_data_file_paths(config.get("data_files")):
            if isinstance(path_str, str):
                problems.extend(_data_file_problems(root, path_str))
            else:
                problems.append("dataset card data_files entry is not a path")
    return problems


def check_layout(root: Path) -> list[str]:
    """Static layout check of a candidate snapshot. Fail-closed problems."""
    problems: list[str] = []
    has_data = False
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if path.is_symlink():
            problems.append(f"symlink is not a supported layout entry: {rel}")
            continue
        if not path.is_file():
            continue
        name = path.name.lower()
        if name.endswith(".py"):
            problems.append(
                f"loading script {rel}: script-based datasets are not supported "
                "(data-only Parquet/JSON/JSONL repositories only)"
            )
        if name in SAVE_TO_DISK_MARKERS:
            problems.append(
                f"{rel} looks like a save_to_disk Arrow cache; stage the "
                "original data-only repository instead"
            )
        if name.endswith(DATA_EXTENSIONS) and not path.name.startswith("."):
            has_data = True
    if not has_data:
        problems.append(
            "no supported data files found (expected Parquet/JSON/JSONL in the repository snapshot)"
        )
    return problems


def _apply_trainer_readable_permissions(root: Path) -> None:
    """New operation files only: dirs 0755, files 0644. No recursive
    chown/chmod of anything outside the operation's own candidate."""
    os.chmod(root, 0o755)
    for path in root.rglob("*"):
        if path.is_symlink():
            continue
        os.chmod(path, 0o755 if path.is_dir() else 0o644)


def _tree_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        total += path.stat().st_size
    return total


def _file_inventory(root: Path) -> list[dict[str, Any]]:
    inventory = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            inventory.append(
                {
                    "path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                }
            )
    return inventory


def _read_manifest(root: Path) -> Optional[dict[str, Any]]:
    manifest_path = root / MANIFEST_NAME
    try:
        raw = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        return None
    return raw if isinstance(raw, dict) else None


def _manifest_matches(manifest: Optional[dict[str, Any]], source: str, sha: str) -> bool:
    return bool(
        manifest
        and manifest.get("source") == source
        and manifest.get("revision") == sha
        and manifest.get("schemaVersion") == MANIFEST_SCHEMA_VERSION
        and isinstance(manifest.get("files"), list)
    )


def _run_verifier(dataset: Path, cache: Path) -> dict[str, Any]:
    """Re-exec this source in a fresh offline interpreter and return its
    config/split summary. The fresh process has an empty writable cache,
    no token, and offline HF env, so only local files can satisfy loads."""
    cache.mkdir(parents=True, exist_ok=True)
    child_env = {key: value for key, value in os.environ.items() if key not in _TOKEN_ENV_KEYS}
    child_env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HOME": str(cache / "hf"),
            "HF_DATASETS_CACHE": str(cache / "datasets"),
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _self_source(),
            "verify",
            "--dataset",
            str(dataset),
        ],
        env=child_env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        raise _fail(
            "offline verification failed in a fresh process: "
            + (detail[-1] if detail else f"exit code {result.returncode}")
        )
    payload = _parse_result_payload(result.stdout)
    if not payload or payload.get("status") != "verified":
        raise _fail("verifier produced no usable result")
    return payload


def _parse_result_payload(log_output: str) -> Optional[dict[str, Any]]:
    for line in reversed(log_output.strip().splitlines()):
        line = line.strip()
        if line.startswith(RESULT_MARKER):
            try:
                payload = json.loads(line[len(RESULT_MARKER) :])
            except ValueError:
                return None
            return payload if isinstance(payload, dict) else None
    return None


def _read_token(token_file: Optional[str]) -> Optional[str]:
    if not token_file:
        return None
    return Path(token_file).read_text().strip() or None


def _expected_bytes(info: Any) -> Optional[int]:
    """Best-effort repo size from the hub metadata, when the API reports it."""
    total = 0
    known = False
    for sibling in getattr(info, "siblings", None) or []:
        size = getattr(sibling, "size", None)
        if size is not None:
            total += size
            known = True
    if not known:
        return None
    return total


def _snapshot(expected: Optional[int], free: int) -> None:
    if expected is not None and expected > free:
        raise _fail(
            f"not enough space on the volume: dataset needs about {expected} "
            f"bytes, volume has {free} free. Grow it with "
            "`prime volumes resize <volume> --size <larger-size>` and retry."
        )


# ---------------------------------------------------------------------------
# verify subcommand: runs in a fresh, offline, token-less process
# ---------------------------------------------------------------------------


def verify_dataset(dataset: Path) -> dict[str, Any]:
    """Load every config offline and summarize splits/columns/counts.

    Never prints row values. Fails closed on any loading error, missing
    shard, external reference, script requirement, or all-empty splits.
    """
    from datasets import get_dataset_config_names, load_dataset

    problems = check_metadata_references(dataset)
    if problems:
        raise _fail("; ".join(problems))

    configs = get_dataset_config_names(str(dataset))
    if not configs:
        raise _fail("no dataset configs discovered")
    summary: dict[str, Any] = {}
    total_rows = 0
    for config in configs:
        loaded = cast(Any, load_dataset(str(dataset), name=config))
        splits: dict[str, Any] = {}
        for split, ds in loaded.items():
            rows = len(ds)
            columns = list(ds.column_names)
            if rows:
                _ = ds[0]  # read one real row; values are never printed
            total_rows += rows
            splits[str(split)] = {"rows": rows, "columns": columns}
        summary[config] = {"splits": splits}
    if total_rows < 1:
        raise _fail(f"all splits are empty (configs: {', '.join(configs)})")
    if len(configs) == 1:
        # The public recipe form for single-config datasets.
        load_dataset(str(dataset))
    return {"status": "verified", "configs": summary}


def _cmd_verify(args: argparse.Namespace) -> int:
    try:
        payload = verify_dataset(Path(args.dataset))
    except StageError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 1
    except Exception as exc:  # noqa: BLE001 - sanitized, fail closed
        detail = f"offline verification error: {type(exc).__name__}: {exc}"
        print(detail, file=sys.stderr, flush=True)
        return 1
    _emit(payload)
    return 0


# ---------------------------------------------------------------------------
# stage subcommand: download -> verify -> publish
# ---------------------------------------------------------------------------


def _cmd_stage(args: argparse.Namespace) -> int:
    # CLI cancellation deletes the pod, which SIGTERMs this container: a
    # plain SIGTERM would skip Python's finally blocks and leave operation
    # scratch behind (live-probed: leftover .prime-stage-* on SIGTERM 143).
    # Route SIGTERM through the KeyboardInterrupt path so owned scratch is
    # removed on unwind - installed FIRST, so the early re-stage path is
    # covered too (signal handlers only install from the main thread).
    def _sigterm(signum, frame):
        raise KeyboardInterrupt()

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, _sigterm)

    volume_root = Path(args.volume_root)
    datasets_root = volume_root / "datasets"
    datasets_root.mkdir(parents=True, exist_ok=True)
    if datasets_root.is_symlink():
        # A symlinked datasets/ root (e.g. datasets -> runs) would redirect
        # operation scratch and published datasets somewhere else entirely.
        raise _fail(
            f"volume path '{datasets_root}/datasets' is a symlink; refusing to stage through it"
        )
    final = datasets_root / args.dataset_name
    operation = args.operation_id
    token = _read_token(args.hf_token_file)

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - image always has it
        raise _fail(f"huggingface_hub is unavailable in the staging image: {exc}") from exc

    api = HfApi(token=token)
    try:
        info = api.dataset_info(args.source, revision=args.revision)
    except Exception as exc:  # noqa: BLE001 - sanitized, fail closed
        # HF deliberately conflates 404 with private/gated: never claim the
        # repo does not exist.
        raise _fail(
            f"could not resolve dataset '{args.source}' at revision "
            f"'{args.revision}' (not found or private/inaccessible; set "
            "HF_TOKEN / --env-file, and accept gated-dataset terms if it is gated)"
        ) from exc
    sha = str(getattr(info, "sha", "") or "")
    if not sha:
        raise _fail(f"revision '{args.revision}' of '{args.source}' has no commit SHA")
    _progress(f"resolved {args.source}@{args.revision} -> {sha}")

    expected = _expected_bytes(info)
    free = shutil.disk_usage(datasets_root).free
    if expected is not None:
        _progress(
            f"repository size about {expected} bytes (advisory; verification "
            f"cache needs more) · volume free {free} bytes"
        )
        _snapshot(expected, free)
    else:
        _progress(f"repository size unknown · volume free {free} bytes")

    # A symlink destination is never followed, replaced, or removed:
    # the API-visible path must be the real dataset directory.
    destination_problems = _existing_dataset_problems(final, args.dataset_name)
    if destination_problems:
        raise _fail("; ".join(destination_problems))

    # Idempotent re-stage: same repo + same resolved SHA verifies and
    # returns already_staged without touching existing data.
    if final.exists():
        manifest = _read_manifest(final)
        if _manifest_matches(manifest, args.source, sha):
            problems = check_layout(final)
            if manifest:
                problems += _inventory_problems(final, manifest)
            if problems:
                raise _fail(
                    "existing staged dataset no longer matches its manifest: " + "; ".join(problems)
                )
            reverify_cache = datasets_root / f".prime-stage-{operation}-cache"
            try:
                verified = _run_verifier(final, reverify_cache)
            finally:
                # A verifier failure must not leak the re-verification
                # cache; cleanup problems surface, never silently pass.
                cache_problems = _remove_scratch([reverify_cache])
                for problem in cache_problems:
                    print(f"scratch cleanup failed: {problem}", file=sys.stderr, flush=True)
            if cache_problems:
                raise _fail(
                    "staged data verified but scratch cleanup failed: "
                    + "; ".join(cache_problems)
                    + " (remove the leftover .prime-stage-* directory)"
                )
            _emit(
                {
                    "status": "already_staged",
                    "operationId": operation,
                    "source": args.source,
                    "requestedRevision": args.revision,
                    "revision": sha,
                    "datasetName": args.dataset_name,
                    "bytes": _tree_bytes(final),
                    "files": len(manifest.get("files", [])) if manifest else 0,
                    "configs": verified.get("configs", {}),
                }
            )
            return 0
        raise _fail(
            f"destination datasets/{args.dataset_name} already exists with a "
            "different source/revision (or without a valid staging manifest). "
            "Existing data is never overwritten: choose a new --path, or "
            "--revision matching the recorded one."
        )

    candidate = datasets_root / f".prime-stage-{operation}"
    cache = datasets_root / f".prime-stage-{operation}-cache"
    # A retry always starts clean: leftover scratch from a hard-killed pod
    # is never resumed blindly.
    _remove_scratch([candidate, cache])
    candidate.mkdir(parents=True)

    started = time.monotonic()  # elapsed time in the emitted result
    cleanup_problems: list[str] = []
    rc = [1]
    try:
        from huggingface_hub import snapshot_download

        _progress(f"downloading snapshot of {args.source} at {sha} ...")
        try:
            snapshot_download(
                repo_id=args.source,
                repo_type="dataset",
                revision=sha,
                local_dir=str(candidate),
                token=token,
            )
        except OSError as exc:
            if exc.errno in (errno.ENOSPC, errno.EDQUOT):
                raise _fail(
                    f"download failed, the volume is full ({exc}); grow it with "
                    "`prime volumes resize <volume> --size <larger-size>` and "
                    "retry"
                ) from exc
            raise _fail(f"download failed (partial data is not published): {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - sanitized, fail closed
            raise _fail(f"download failed (partial data is not published): {exc}") from exc

        problems = check_layout(candidate)
        if problems:
            raise _fail(
                "unsupported or corrupt dataset layout: "
                + "; ".join(problems)
                + " (supported: data-only Parquet/JSON/JSONL repositories)"
            )

        _progress("verifying offline with a fresh process ...")
        verified = _run_verifier(candidate, cache)
        configs = verified.get("configs", {})

        # Remove download bookkeeping/verification scratch, keep dataset
        # metadata (README card etc.). The loader must not see hub
        # metadata; if the bookkeeping cannot be removed, fail closed.
        bookkeeping_problems = _remove_scratch([candidate / ".cache"])
        if bookkeeping_problems:
            raise _fail(
                "could not remove download bookkeeping before publication: "
                + "; ".join(bookkeeping_problems)
            )
        _apply_trainer_readable_permissions(candidate)

        dataset_bytes = _tree_bytes(candidate)
        manifest = {
            "schemaVersion": MANIFEST_SCHEMA_VERSION,
            "operationId": operation,
            "source": args.source,
            "requestedRevision": args.revision,
            "revision": sha,
            "stagedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "bytes": dataset_bytes,
            "files": _file_inventory(candidate),
            "libraryVersions": _library_versions(),
            "configs": configs,
        }
        (candidate / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
        os.chmod(candidate / MANIFEST_NAME, 0o644)

        published: Optional[dict[str, Any]] = None
        try:
            publish_no_replace(candidate, final)
        except FileExistsError:
            winner_problems = _existing_dataset_problems(final, args.dataset_name)
            if winner_problems:
                raise _fail("; ".join(winner_problems))
            winner = _read_manifest(final)
            if not _manifest_matches(winner, args.source, sha):
                raise _fail(
                    f"destination datasets/{args.dataset_name} was published "
                    "by a different source/revision concurrently; existing "
                    "data was left untouched"
                )
            # Same repo+SHA won the race: verify the winner's directory as
            # strictly as a re-stage would, never trust the manifest alone.
            winner_problems = check_layout(final)
            if winner:
                winner_problems += _inventory_problems(final, winner)
            if winner_problems:
                raise _fail(
                    "another operation published this revision but the "
                    "published data does not match its manifest: " + "; ".join(winner_problems)
                )
            winner_verified = _run_verifier(final, cache)
            _progress("another staging operation published the same revision")
            published = {
                "status": "already_staged",
                "operationId": operation,
                "source": args.source,
                "requestedRevision": args.revision,
                "revision": sha,
                "datasetName": args.dataset_name,
                "bytes": _tree_bytes(final),
                "files": len(winner.get("files", [])) if winner else 0,
                "configs": winner_verified.get("configs", configs),
            }
        except OSError as exc:
            # A filesystem-level failure (e.g. EPERM) must not be evaded by
            # a weaker publication path, nor crash with a raw traceback.
            raise _fail(f"publication failed on this filesystem: {exc}") from exc
        else:
            _progress("published")
            published = {
                "status": "staged",
                "operationId": operation,
                "source": args.source,
                "requestedRevision": args.revision,
                "revision": sha,
                "datasetName": args.dataset_name,
                "bytes": dataset_bytes,
                "files": len(cast(list, manifest["files"])),
                "configs": configs,
                "elapsedSeconds": round(time.monotonic() - started, 1),
            }
        rc[0] = 0
    finally:
        # Owned scratch only: the candidate, its cache, and nothing else.
        # Cleanup errors are NOT swallowed: they are printed on EVERY
        # path (failure and cancellation included, where the already-
        # pending exception would otherwise skip the reporting below) and
        # fail the operation instead of pretending success.
        cleanup_problems.extend(_remove_scratch([candidate, cache]))
        for problem in cleanup_problems:
            print(f"scratch cleanup failed: {problem}", file=sys.stderr, flush=True)
    if rc[0] != 0:
        return 1
    if cleanup_problems:
        # Publication may already have happened (the rename commits before
        # this check): report it as incomplete, never as a green success -
        # and emit no success result line.
        print(
            "published, but scratch cleanup incomplete: "
            + "; ".join(cleanup_problems)
            + f" - remove the leftover {candidate} directory",
            file=sys.stderr,
            flush=True,
        )
        return 1
    _emit(published if published is not None else {})
    return 0


def _remove_scratch(paths: list[Path]) -> list[str]:
    """Best-effort removal of operation-owned scratch directories. A
    directory that is already gone counts as cleaned; anything else is
    reported so the operation can fail instead of pretending success."""
    problems: list[str] = []
    for path in paths:
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            continue
        except OSError as exc:
            problems.append(f"{path}: {exc}")
    return problems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prime-volume-stage")
    sub = parser.add_subparsers(dest="command", required=True)

    stage = sub.add_parser("stage", help="download, verify, publish")
    stage.add_argument("--source", required=True)
    stage.add_argument("--revision", default="main")
    stage.add_argument("--dataset-name", required=True)
    stage.add_argument("--volume-root", default="/volume")
    stage.add_argument("--operation-id", required=True)
    stage.add_argument("--hf-token-file", default=None)
    stage.set_defaults(func=_cmd_stage)

    verify = sub.add_parser("verify", help="offline fresh-process verification")
    verify.add_argument("--dataset", required=True)
    verify.set_defaults(func=_cmd_verify)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except StageError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - signal path
        print("staging interrupted", file=sys.stderr, flush=True)
        return 130


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
