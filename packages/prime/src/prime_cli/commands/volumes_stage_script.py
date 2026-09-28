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
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
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
    """The target filesystem does not support renameat2(RENAME_NOREPLACE)."""


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
    if err in (errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP, errno.ENOTSUP, errno.EPERM):
        raise UnsupportedRenameError(
            f"filesystem does not support renameat2(RENAME_NOREPLACE) "
            f"(errno {err}, {os.strerror(err)}); refusing to publish with a "
            "rename that could overwrite existing data"
        )
    raise OSError(err, os.strerror(err), str(src), None, str(dst))


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
    volume_root = Path(args.volume_root)
    datasets_root = volume_root / "datasets"
    datasets_root.mkdir(parents=True, exist_ok=True)
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

    # Idempotent re-stage: same repo + same resolved SHA verifies and
    # returns already_staged without touching existing data.
    if final.exists():
        manifest = _read_manifest(final)
        if _manifest_matches(manifest, args.source, sha):
            problems = check_layout(final)
            if problems:
                raise _fail(
                    "existing staged dataset no longer matches its manifest: " + "; ".join(problems)
                )
            reverify_cache = datasets_root / f".prime-stage-{operation}-cache"
            try:
                verified = _run_verifier(final, reverify_cache)
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
            finally:
                shutil.rmtree(reverify_cache, ignore_errors=True)
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
    shutil.rmtree(candidate, ignore_errors=True)
    shutil.rmtree(cache, ignore_errors=True)
    candidate.mkdir(parents=True)

    started = time.monotonic()  # elapsed time in the emitted result
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
        # metadata (README card etc.). The loader must not see hub metadata.
        shutil.rmtree(candidate / ".cache", ignore_errors=True)
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

        try:
            _rename_noreplace(candidate, final)
        except FileExistsError:
            winner = _read_manifest(final)
            if _manifest_matches(winner, args.source, sha):
                _progress("another staging operation published the same revision")
                _emit(
                    {
                        "status": "already_staged",
                        "operationId": operation,
                        "source": args.source,
                        "requestedRevision": args.revision,
                        "revision": sha,
                        "datasetName": args.dataset_name,
                        "bytes": _tree_bytes(final),
                        "files": len(winner.get("files", [])) if winner else 0,
                        "configs": (winner or {}).get("configs", configs),
                    }
                )
                return 0
            raise _fail(
                f"destination datasets/{args.dataset_name} was published by a "
                "different source/revision concurrently; existing data was left "
                "untouched"
            )

        _emit(
            {
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
        )
        return 0
    finally:
        # Owned scratch only: the candidate, its cache, and nothing else.
        shutil.rmtree(candidate, ignore_errors=True)
        shutil.rmtree(cache, ignore_errors=True)


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
