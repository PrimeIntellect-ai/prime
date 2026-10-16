"""Isolated CPU schema worker for experimental hosted config validation."""

import contextlib
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

if os.name == "nt":
    import msvcrt
else:
    import fcntl


class CompilationError(Exception):
    pass


SFT_ONLINE_EVAL_KEYS = frozenset({"eval", "inference", "weight_broadcast"})
# Each schema environment is 250-650 MB; keep a few recent ones.
_CACHE_KEEP = 4
_CACHE_MAX_AGE = 14 * 86400
_CACHE_SWEEP_INTERVAL = 86400
_CACHE_KEY = re.compile(r"[0-9a-f]{64}")


_WORKER = Path(__file__).with_name("training_schema_worker.py").read_text()
_BYTECODE_COMPILER = """
import compileall
import sysconfig

for path in sorted({sysconfig.get_path("purelib"), sysconfig.get_path("platlib")}):
    # Match uv: an invalid, unused source file must not prevent validation.
    compileall.compile_dir(path, quiet=2)
"""


def _worker_environment() -> dict[str, str]:
    # Selected schemas execute locally. Do not give the subprocess API tokens
    # or inherit PYTHONPATH, which could select a different schema installation.
    return {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "HOME", "SYSTEMROOT", "TMPDIR", "TEMP", "LANG", "LC_ALL"}
    } | {"PYTHONDONTWRITEBYTECODE": "1"}


_NETWORK_VARIABLES = {
    name
    for base in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY")
    for name in (base, base.lower())
} | {"SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE"}


def _installer_environment() -> dict[str, str]:
    # Installers fetch from the network, so they keep proxy and CA settings.
    # prime-rl registers its submodules with SSH URLs; rewrite them to HTTPS so
    # a fresh install needs no GitHub keys. Deliberately no UV_* variables.
    rewrites = ("git@github.com:", "ssh://git@github.com/")
    git_config = {"GIT_CONFIG_COUNT": str(len(rewrites))}
    for index, prefix in enumerate(rewrites):
        git_config[f"GIT_CONFIG_KEY_{index}"] = "url.https://github.com/.insteadOf"
        git_config[f"GIT_CONFIG_VALUE_{index}"] = prefix
    return (
        _worker_environment()
        | {k: v for k, v in os.environ.items() if k in _NETWORK_VARIABLES}
        | git_config
    )


def _run_process(
    command: list[str],
    *,
    timeout: float,
    input: str | None = None,
    env: dict[str, str] | None = None,
    pass_fds: tuple[int, ...] = (),
) -> subprocess.CompletedProcess[str]:
    # Installers spawn Git and build backends. They must stop before the
    # caller releases the cache lock and a retry replaces their environment.
    with subprocess.Popen(
        command,
        stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env if env is not None else _worker_environment(),
        pass_fds=pass_fds,
        start_new_session=os.name != "nt",
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    ) as process:
        try:
            stdout, stderr = process.communicate(input=input, timeout=timeout)
        except BaseException:
            if os.name == "nt":
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=10,
                        check=False,
                    )
                finally:
                    process.kill()
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.kill()
            process.communicate()
            raise
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _run(command: list[str], *, lock: int | None, timeout: int = 300) -> None:
    # Installers inherit the cache lock: if the CLI is killed, its surviving
    # installer tree keeps the cache locked until it exits.
    try:
        result = _run_process(
            command,
            timeout=timeout,
            env=_installer_environment(),
            pass_fds=() if lock is None else (lock,),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CompilationError(f"Could not build local schema environment: {exc}") from exc
    if result.returncode:
        raise CompilationError(
            "Could not build local schema environment:\n" + result.stderr[-2000:]
        )


@contextlib.contextmanager
def _cache_lock(path: Path, timeout: float, *, shared: bool = False) -> Iterator[int | None]:
    # An OS lock dies with its holder, so a killed install never wedges the cache.
    # Windows region locks are exclusive only, so shared holders take turns there.
    deadline = time.monotonic() + timeout
    with path.open("a+b") as handle:
        while True:
            try:
                if os.name == "nt":
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
                    fcntl.flock(handle, mode | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise CompilationError(f"Timed out waiting for schema cache lock: {path}")
                time.sleep(0.2)
        try:
            # Windows region locks belong to the handle, so children cannot share them.
            yield None if os.name == "nt" else handle.fileno()
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def _schema_paths(inputs: dict[str, Any]) -> tuple[Path, str, Path, Path]:
    identity = json.dumps({"install": inputs, "cache_version": 2}, sort_keys=True).encode()
    key = hashlib.sha256(identity).hexdigest()
    cache = (
        Path(os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        / "prime"
        / "training-schemas"
    )
    target = cache / key
    python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return cache, key, target, python


def _schema_python(preparation: dict[str, Any]) -> Path:
    if not shutil.which("uv"):
        raise CompilationError("Local config validation requires uv on PATH.")
    inputs = preparation["schema_install"]
    cache, key, target, python = _schema_paths(inputs)
    cache.mkdir(parents=True, exist_ok=True)
    bytecode_ready = target / ".bytecode-v1.ready"
    lock_path = cache / (key + ".lockfile")
    with _cache_lock(lock_path, 300) as lock:
        if not (target / ".ready").exists() or not python.is_file():
            # Install in the final location: venv scripts contain absolute paths.
            shutil.rmtree(target, ignore_errors=True)
            # The selected runtime supplies the install inputs. Ambient uv
            # settings (for example, a project's package cooldown or indexes)
            # must not change how this separate schema environment is built.
            _run(
                ["uv", "venv", "--no-config", "--python", inputs["python"], str(target)],
                lock=lock,
                timeout=60,
            )
            with tempfile.TemporaryDirectory(prefix="prime-schema-") as temp:
                constraints = Path(temp) / "constraints.txt"
                constraints.write_text("\n".join(inputs["constraints"]) + "\n")
                _run(
                    [
                        "uv",
                        "pip",
                        "install",
                        "--no-config",
                        "--compile-bytecode",
                        "--python",
                        str(python),
                        "--no-sources",
                        "--prerelease",
                        "allow",
                        "--constraints",
                        str(constraints),
                        *inputs["requirements"],
                    ],
                    lock=lock,
                )
            bytecode_ready.touch()
            (target / ".ready").touch()
        elif not bytecode_ready.is_file():
            # Upgrade existing schema caches without resolving or reinstalling
            # dependencies. The cache lock also covers this one-time step.
            _run([str(python), "-I", "-B", "-c", _BYTECODE_COMPILER], lock=lock, timeout=120)
            bytecode_ready.touch()
        return python


@contextlib.contextmanager
def _schema_environment(preparation: dict[str, Any]) -> Iterator[Path]:
    # The worker runs under a shared lock so a cache sweep cannot remove its environment.
    cache, key, target, python = _schema_paths(preparation["schema_install"])
    while True:
        cache.mkdir(parents=True, exist_ok=True)
        with _cache_lock(cache / (key + ".lockfile"), 300, shared=True):
            if (
                (target / ".ready").exists()
                and (target / ".bytecode-v1.ready").exists()
                and python.is_file()
            ):
                (target / ".last-used").touch()
                yield python
                break
        _schema_python(preparation)
    with contextlib.suppress(OSError):
        _sweep_cache(cache, key, time.time())


def envs_to_delete(last_used: dict[str, float], current: str, now: float) -> list[str]:
    """Cached environments to remove: unused for too long, or beyond the newest few."""
    others = sorted((key for key in last_used if key != current), key=last_used.get, reverse=True)
    return [
        key
        for rank, key in enumerate(others)
        if rank >= _CACHE_KEEP - 1 or now - last_used[key] > _CACHE_MAX_AGE
    ]


def _last_used(target: Path) -> float:
    for marker in (".last-used", ".ready"):
        with contextlib.suppress(FileNotFoundError):
            return (target / marker).stat().st_mtime
    return target.stat().st_mtime


def _sweep_cache(cache: Path, current: str, now: float) -> None:
    stamp = cache / ".last-sweep"
    with contextlib.suppress(FileNotFoundError):
        if now - stamp.stat().st_mtime < _CACHE_SWEEP_INTERVAL:
            return
    stamp.touch()
    last_used = {
        path.name: _last_used(path)
        for path in cache.iterdir()
        if path.is_dir() and _CACHE_KEY.fullmatch(path.name)
    }
    for key in envs_to_delete(last_used, current, now):
        try:
            # Busy environments are skipped. Lock files stay: removing one could
            # let two processes lock different files for the same environment.
            with _cache_lock(cache / (key + ".lockfile"), 0):
                (cache / key).rename(cache / f"{key}.trash-{os.getpid()}")
        except (CompilationError, OSError):
            continue
    for trash in cache.glob("*.trash-*"):
        shutil.rmtree(trash, ignore_errors=True)
    for legacy in cache.glob("*.lock"):
        # Directory locks from CLI versions before the OS lock.
        if legacy.is_dir() and now - legacy.stat().st_mtime > _CACHE_MAX_AGE:
            with contextlib.suppress(OSError):
                legacy.rmdir()


def _schema_request(
    request: dict[str, Any], preparation: dict[str, Any], schema_python: Path | None
) -> dict[str, Any]:
    if preparation.get("contract_version") != 1:
        raise CompilationError("Unsupported local validation contract version.")
    runtime = preparation["runtime"]
    if runtime["schema_commit"] != (runtime.get("source_commit") or runtime["image_commit"]):
        raise CompilationError("Preparation schema commit does not match the selected runtime.")
    request = {**request, "mode": preparation["mode"], "schema_commit": runtime["schema_commit"]}
    environment = (
        contextlib.nullcontext(schema_python.absolute())
        if schema_python
        else _schema_environment(preparation)
    )
    try:
        with environment as python:
            result = _run_process(
                [str(python), "-I", "-B", "-c", _WORKER], input=json.dumps(request), timeout=60
            )
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        raise CompilationError(f"Local schema worker failed: {exc}") from exc
    try:
        response = json.loads(result.stdout)
    except ValueError as exc:
        raise CompilationError(
            "Local schema worker returned invalid JSON:\n" + result.stderr[-2000:]
        ) from exc
    if result.returncode or "error" in response:
        raise CompilationError(response.get("error", "Local schema worker failed."))
    return response


def compile_locally(
    config: dict[str, Any], preparation: dict[str, Any], *, schema_python: Path | None = None
) -> dict[str, Any]:
    response = _schema_request(
        {"config": config, "sft_online_eval_keys": sorted(SFT_ONLINE_EVAL_KEYS)},
        preparation,
        schema_python,
    )
    return {**response["artifact"], "runtime": preparation["runtime"]}
