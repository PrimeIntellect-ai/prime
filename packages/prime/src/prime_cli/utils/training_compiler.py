"""Isolated CPU schema worker for experimental hosted config validation."""

import contextlib
import hashlib
import json
import os
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


def _run(command: list[str], *, timeout: int = 300) -> None:
    try:
        result = _run_process(command, timeout=timeout, env=_installer_environment())
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CompilationError(f"Could not build local schema environment: {exc}") from exc
    if result.returncode:
        raise CompilationError(
            "Could not build local schema environment:\n" + result.stderr[-2000:]
        )


@contextlib.contextmanager
def _cache_lock(path: Path, timeout: float) -> Iterator[None]:
    # An OS lock dies with its holder, so a killed install never wedges the cache.
    deadline = time.monotonic() + timeout
    with path.open("a+b") as handle:
        while True:
            try:
                if os.name == "nt":
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise CompilationError(f"Timed out waiting for schema cache lock: {path}")
                time.sleep(0.2)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def _schema_python(preparation: dict[str, Any]) -> Path:
    if not shutil.which("uv"):
        raise CompilationError("Local config validation requires uv on PATH.")
    inputs = preparation["schema_install"]
    identity = json.dumps({"install": inputs, "cache_version": 2}, sort_keys=True).encode()
    key = hashlib.sha256(identity).hexdigest()
    cache = (
        Path(os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        / "prime"
        / "training-schemas"
    )
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / key
    python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    bytecode_ready = target / ".bytecode-v1.ready"
    lock = cache / (key + ".lockfile")
    with _cache_lock(lock, 300):
        if not (target / ".ready").exists() or not python.is_file():
            # Install in the final location: venv scripts contain absolute paths.
            shutil.rmtree(target, ignore_errors=True)
            # The selected runtime supplies the install inputs. Ambient uv
            # settings (for example, a project's package cooldown or indexes)
            # must not change how this separate schema environment is built.
            _run(
                ["uv", "venv", "--no-config", "--python", inputs["python"], str(target)],
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
                    ]
                )
            bytecode_ready.touch()
            (target / ".ready").touch()
        elif not bytecode_ready.is_file():
            # Upgrade existing schema caches without resolving or reinstalling
            # dependencies. The cache lock also covers this one-time step.
            _run([str(python), "-I", "-B", "-c", _BYTECODE_COMPILER], timeout=120)
            bytecode_ready.touch()
        return python


def _schema_request(
    request: dict[str, Any], preparation: dict[str, Any], schema_python: Path | None
) -> dict[str, Any]:
    if preparation.get("contract_version") != 1:
        raise CompilationError("Unsupported local validation contract version.")
    runtime = preparation["runtime"]
    if runtime["schema_commit"] != (runtime.get("source_commit") or runtime["image_commit"]):
        raise CompilationError("Preparation schema commit does not match the selected runtime.")
    python = schema_python.absolute() if schema_python else _schema_python(preparation)
    request = {**request, "mode": preparation["mode"], "schema_commit": runtime["schema_commit"]}
    try:
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
    response = _schema_request({"config": config}, preparation, schema_python)
    return {**response["artifact"], "runtime": preparation["runtime"]}
