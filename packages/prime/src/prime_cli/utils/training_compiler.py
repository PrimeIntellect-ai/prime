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


_WORKER = r"""
import json
import sys
from importlib import metadata

try:
    request = json.load(sys.stdin)
    sys.path.insert(0, request["compiler_path"])
    dist = metadata.distribution("prime-rl-configs")
    install = json.loads(dist.read_text("direct_url.json") or "{}")
    actual = install.get("vcs_info", {}).get("commit_id")
    expected = request["schema_commit"]
    if actual != expected:
        raise ValueError(f"schema provenance mismatch: expected {expected}, installed {actual}")
    if request.get("deployment"):
        from rl_k8s.compiled import compile_deployment as compile_fn
    else:
        from rl_k8s.compiled import compile_config as compile_fn
    artifact = compile_fn(request["config"], mode=request["mode"], schema_commit=expected)
    print(json.dumps({"artifact": artifact}))
except Exception as exc:
    try:
        from pydantic import ValidationError
    except ImportError:
        ValidationError = ()
    if isinstance(exc, ValidationError):
        error = "\n".join(
            f"{'.'.join(map(str, item['loc']))}: {item['msg']}"
            for item in exc.errors(include_input=False)
        )
    else:
        error = str(exc)
    print(json.dumps({"error": error}))
    sys.exit(1)
"""


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


def _schema_python(preparation: dict[str, Any], compiler_path: Path) -> Path:
    if not shutil.which("uv"):
        raise CompilationError("Local config validation requires uv on PATH.")
    inputs = preparation["schema_install"]
    identity = json.dumps(
        {
            "install": inputs,
            "compiler_path": str(compiler_path),
            "pyproject": (compiler_path / "pyproject.toml").read_text(),
        },
        sort_keys=True,
    ).encode()
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
            _run(["uv", "venv", "--python", inputs["python"], str(target)], timeout=60)
            with tempfile.TemporaryDirectory(prefix="prime-schema-") as temp:
                constraints = Path(temp) / "constraints.txt"
                constraints.write_text("\n".join(inputs["constraints"]) + "\n")
                _run(
                    [
                        "uv",
                        "pip",
                        "install",
                        "--python",
                        str(python),
                        "--no-sources",
                        "--prerelease",
                        "allow",
                        "--constraints",
                        str(constraints),
                        *inputs["requirements"],
                        str(compiler_path),
                    ]
                )
            (target / ".ready").touch()
        return python
    finally:
        lock.rmdir()


def compile_locally(
    config: dict[str, Any],
    preparation: dict[str, Any],
    *,
    compiler_path: Path,
    schema_python: Path | None = None,
    deployment: bool = False,
) -> dict[str, Any]:
    if preparation.get("contract_version") != 1:
        raise CompilationError("Unsupported local validation contract version.")
    runtime = preparation["runtime"]
    if runtime["schema_commit"] != (runtime.get("source_commit") or runtime["image_commit"]):
        raise CompilationError("Preparation schema commit does not match the selected runtime.")
    compiler_path = compiler_path.resolve()
    if not (compiler_path / "rl_k8s" / "compiled.py").is_file():
        raise CompilationError(
            "--compiler-path must point to this branch's FFT prime-rl-k8s directory."
        )
    # Resolve relative paths, but preserve venv executable symlinks: resolving
    # one to the base interpreter silently loses the selected environment.
    python = (
        schema_python.absolute() if schema_python else _schema_python(preparation, compiler_path)
    )
    request = {
        "compiler_path": str(compiler_path),
        "config": config,
        "mode": preparation["mode"],
        "schema_commit": preparation["runtime"]["schema_commit"],
        "deployment": deployment,
    }
    try:
        result = _run_process(
            [str(python), "-I", "-B", "-c", _WORKER],
            input=json.dumps(request),
            timeout=60,
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
    return {**response["artifact"], "runtime": preparation["runtime"]}
