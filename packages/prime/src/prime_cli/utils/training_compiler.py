"""Isolated CPU schema worker for experimental hosted config validation."""

import hashlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


class CompilationError(Exception):
    pass


_WORKER = Path(__file__).with_name("training_schema_worker.py").read_text()


def _worker_environment() -> dict[str, str]:
    # Selected schemas execute locally. Do not give the subprocess API tokens
    # or inherit PYTHONPATH, which could select a different schema installation.
    return {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "HOME", "SYSTEMROOT", "TMPDIR", "TEMP", "LANG", "LC_ALL"}
    } | {"PYTHONDONTWRITEBYTECODE": "1"}


def _run_process(
    command: list[str], *, timeout: float, input: str | None = None
) -> subprocess.CompletedProcess[str]:
    # Installers spawn Git and build backends. They must stop before the
    # caller releases the cache lock and a retry replaces their environment.
    with subprocess.Popen(
        command,
        stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_worker_environment(),
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
        result = _run_process(command, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CompilationError(f"Could not build local schema environment: {exc}") from exc
    if result.returncode:
        raise CompilationError(
            "Could not build local schema environment:\n" + result.stderr[-2000:]
        )


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
    lock = cache / (key + ".lock")
    deadline = time.monotonic() + 300
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise CompilationError(f"Timed out waiting for schema cache lock: {lock}")
            time.sleep(0.2)
    try:
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
            (target / ".ready").touch()
        return python
    finally:
        lock.rmdir()


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
