#!/usr/bin/env python3
"""Run Prime release e2e checks inside a Prime sandbox.

The host script creates an isolated sandbox, uploads the current checkout, and runs
live release smoke checks there so the GitHub runner stays clean while the tested
CLI is the candidate source tree from the PR/release branch.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tarfile
import tempfile
import textwrap
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from prime_sandboxes import APIClient, CreateSandboxRequest, SandboxClient

DEFAULT_MODEL = "deepseek/deepseek-chat"
DEFAULT_HOSTED_TIMEOUT_MINUTES = "120"
DEFAULT_SANDBOX_IMAGE = "python:3.11-bookworm"

ARCHIVE_SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}
ARCHIVE_SKIP_SUFFIXES = {".pyc", ".pyo"}
SECRET_FILE_PATTERNS = (
    ".env",
    ".env.",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "credentials.json",
    "service-account",
)
SECRET_SOURCE_MODULE_NAMES = ("secrets.py",)
SECRET_FILE_SUFFIXES = (".key", ".pem")


@dataclass(frozen=True)
class RemoteConfig:
    model: str
    hosted_mode: str
    env_prefix: str
    hosted_timeout_minutes: int
    cleanup_remote_env: bool
    run_suffix: str


def env_int(name: str, default: str) -> int:
    return int(os.environ.get(name) or default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Prime monorepo checkout to test (default: repository root).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("PRIME_E2E_MODEL", DEFAULT_MODEL),
        help="Prime Inference model used for one-example eval smoke tests.",
    )
    parser.add_argument(
        "--hosted-mode",
        choices=("skip", "submit", "wait"),
        default=os.environ.get("PRIME_E2E_HOSTED_MODE", "submit"),
        help=(
            "Hosted eval behavior: skip, submit and cancel after API visibility, "
            "or wait for terminal completion."
        ),
    )
    parser.add_argument(
        "--hosted-timeout-minutes",
        type=int,
        default=env_int("PRIME_E2E_HOSTED_TIMEOUT_MINUTES", DEFAULT_HOSTED_TIMEOUT_MINUTES),
        help="Timeout passed to hosted evals and used while waiting for completion.",
    )
    parser.add_argument(
        "--sandbox-timeout-minutes",
        type=int,
        default=env_int("PRIME_E2E_SANDBOX_TIMEOUT_MINUTES", "120"),
        help="Sandbox lifetime in minutes.",
    )
    parser.add_argument(
        "--command-timeout-seconds",
        type=int,
        default=env_int("PRIME_E2E_COMMAND_TIMEOUT_SECONDS", "7200"),
        help="Maximum time for the in-sandbox e2e command.",
    )
    parser.add_argument(
        "--sandbox-image",
        default=os.environ.get("PRIME_E2E_SANDBOX_IMAGE", DEFAULT_SANDBOX_IMAGE),
        help="Docker image used for the Prime sandbox.",
    )
    parser.add_argument(
        "--sandbox-region",
        default=os.environ.get("PRIME_E2E_SANDBOX_REGION") or None,
        help="Optional sandbox region.",
    )
    parser.add_argument(
        "--env-prefix",
        default=os.environ.get("PRIME_E2E_ENV_PREFIX", "prime-e2e"),
        help="Prefix for the temporary Environment Hub package name.",
    )
    parser.add_argument(
        "--keep-sandbox",
        action="store_true",
        default=os.environ.get("PRIME_E2E_KEEP_SANDBOX") == "1",
        help="Leave the sandbox running for debugging.",
    )
    parser.add_argument(
        "--no-cleanup-remote-env",
        action="store_true",
        help="Do not delete the temporary Environment Hub environment at the end.",
    )
    return parser.parse_args()


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is required for release e2e checks")
    return value


def sanitize_slug_part(value: str, *, max_length: int = 32) -> str:
    sanitized = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    sanitized = re.sub(r"-+", "-", sanitized)
    return (sanitized or "run")[:max_length].strip("-") or "run"


def default_run_suffix() -> str:
    github_run = os.environ.get("GITHUB_RUN_ID")
    github_attempt = os.environ.get("GITHUB_RUN_ATTEMPT")
    if github_run:
        suffix = github_run if not github_attempt else f"{github_run}-{github_attempt}"
        return sanitize_slug_part(suffix, max_length=24)
    return f"{time.strftime('%m%d%H%M')}-{uuid.uuid4().hex[:8]}"


def is_secret_file(path: Path) -> bool:
    name = path.name.lower()
    if name in SECRET_SOURCE_MODULE_NAMES:
        return False
    return (
        name.startswith("secrets.")
        or name.endswith(SECRET_FILE_SUFFIXES)
        or any(name == pattern or name.startswith(pattern) for pattern in SECRET_FILE_PATTERNS)
    )


def should_archive(path: Path, repo_root: Path) -> bool:
    if path.is_symlink():
        return False
    rel = path.relative_to(repo_root)
    if any(part in ARCHIVE_SKIP_DIRS for part in rel.parts):
        return False
    if path.is_file() and path.suffix in ARCHIVE_SKIP_SUFFIXES:
        return False
    if path.is_file() and is_secret_file(path):
        return False
    return True


def create_source_archive(repo_root: Path) -> Path:
    repo_root = repo_root.resolve()
    archive = Path(tempfile.mkdtemp(prefix="prime-release-e2e-")) / "prime-src.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(repo_root.rglob("*")):
            if not should_archive(path, repo_root):
                continue
            rel = path.relative_to(repo_root)
            tar.add(path, arcname=Path("prime") / rel, recursive=False)
    return archive


def remote_script(config: RemoteConfig) -> str:
    config_literal = repr(config.__dict__)
    template = """
        from __future__ import annotations

        import json
        import os
        import re
        import shlex
        import subprocess
        import sys
        import textwrap
        import time
        from pathlib import Path

        CONFIG = __CONFIG_JSON__
        WORKSPACE = Path("/workspace")
        SOURCE_ROOT = WORKSPACE / "prime"
        LAB_ROOT = WORKSPACE / "release-e2e"
        ENVS_ROOT = LAB_ROOT / "environments"
        HOME = WORKSPACE / "home"
        VENV = WORKSPACE / "venv"
        ENV_NAME = f"{CONFIG['env_prefix']}-{CONFIG['run_suffix']}"
        MODULE_NAME = ENV_NAME.replace("-", "_")
        ENV_DIR = ENVS_ROOT / MODULE_NAME
        REMOTE_ENV_SLUG = None
        HOSTED_EVAL_IDS: list[str] = []
        TERMINAL_HOSTED_STATUSES = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED"}


        class CommandFailed(RuntimeError):
            def __init__(self, cmd: list[str], returncode: int) -> None:
                super().__init__(f"Command failed ({returncode}): {shlex.join(cmd)}")
                self.returncode = returncode


        def command_env() -> dict[str, str]:
            env = os.environ.copy()
            env["HOME"] = str(HOME)
            env["PATH"] = f"{VENV / 'bin'}:{env.get('PATH', '')}"
            env["PRIME_DISABLE_VERSION_CHECK"] = "1"
            env["PYTHONUNBUFFERED"] = "1"
            return env


        def run(
            cmd: list[str],
            *,
            cwd: Path | None = None,
            check: bool = True,
            timeout: int | None = None,
        ) -> str:
            print(f"\\n$ {shlex.join(cmd)}", flush=True)
            result = subprocess.run(
                cmd,
                cwd=cwd,
                env=command_env(),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
            if result.stdout:
                print(result.stdout, end="" if result.stdout.endswith("\\n") else "\\n")
            if check and result.returncode != 0:
                raise CommandFailed(cmd, result.returncode)
            return result.stdout or ""


        def write_file(path: Path, content: str) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")


        def write_smoke_environment() -> None:
            ENV_DIR.mkdir(parents=True, exist_ok=True)
            write_file(
                ENV_DIR / "pyproject.toml",
                f'''
                [project]
                name = "{ENV_NAME}"
                version = "0.0.1"
                description = "Temporary Prime release e2e smoke environment"
                requires-python = ">=3.11"
                dependencies = [
                  "datasets>=2.18.0",
                  "verifiers>=0.2.0",
                ]

                [build-system]
                requires = ["hatchling"]
                build-backend = "hatchling.build"

                [tool.hatch.build]
                include = ["{MODULE_NAME}.py", "README.md"]

                [project.entry-points."verifiers.environments"]
                {MODULE_NAME} = "{MODULE_NAME}:load_environment"
                ''',
            )
            write_file(
                ENV_DIR / "README.md",
                f'''
                # {ENV_NAME}

                Temporary release e2e environment created by the Prime CLI release smoke test.
                It contains one deterministic single-turn example and is deleted after the test.
                ''',
            )
            write_file(
                ENV_DIR / f"{MODULE_NAME}.py",
                '''
                from __future__ import annotations

                from datasets import Dataset
                import verifiers as vf

                SYSTEM_PROMPT = 'Answer with the exact requested token inside <answer> tags.'


                def _rows(split: str, variant: str) -> list[dict[str, str]]:
                    return [
                        {
                            "id": f"{split}-{variant}-0",
                            "question": "Return only the token prime inside <answer> tags.",
                            "answer": "prime",
                        }
                    ]


                def load_environment(
                    split: str = "eval",
                    variant: str = "default",
                    num_examples: int = -1,
                    **kwargs,
                ) -> vf.Environment:
                    dataset = Dataset.from_list(_rows(split, variant))
                    if num_examples > 0:
                        dataset = dataset.select(range(min(num_examples, len(dataset))))

                    parser = vf.XMLParser(["answer"])

                    async def exact_answer(completion, answer, parser) -> float:
                        parsed = parser.parse_answer(completion)
                        if parsed is None:
                            return 0.0
                        return 1.0 if str(parsed).strip().lower() == str(answer).strip() else 0.0

                    rubric = vf.Rubric(
                        funcs=[exact_answer, parser.get_format_reward_func()],
                        weights=[1.0, 0.2],
                        parser=parser,
                    )
                    return vf.SingleTurnEnv(
                        dataset=dataset,
                        eval_dataset=dataset,
                        rubric=rubric,
                        system_prompt=SYSTEM_PROMPT,
                        **kwargs,
                    )
                ''',
            )


        def install_candidate_cli() -> None:
            run([sys.executable, "-m", "venv", str(VENV)], timeout=300)
            python = str(VENV / "bin" / "python")
            run([python, "-m", "pip", "install", "--upgrade", "pip", "uv"], timeout=600)
            uv = str(VENV / "bin" / "uv")
            run(
                [
                    uv,
                    "pip",
                    "install",
                    "-e",
                    str(SOURCE_ROOT / "packages" / "prime-evals"),
                    "-e",
                    str(SOURCE_ROOT / "packages" / "prime-sandboxes"),
                    "-e",
                    str(SOURCE_ROOT / "packages" / "prime-tunnel"),
                    "-e",
                    str(SOURCE_ROOT / "packages" / "prime"),
                ],
                timeout=1800,
            )
            run(["prime", "--version"], timeout=120)


        def installed_runtime_regression_checks() -> None:
            run(
                [
                    str(VENV / "bin" / "python"),
                    "-c",
                    textwrap.dedent(
                        '''
                        import inspect

                        from prime_cli.commands import evals
                        from prime_tunnel import Tunnel

                        if "labels" not in inspect.signature(Tunnel).parameters:
                            raise SystemExit("prime_tunnel.Tunnel must accept labels")

                        calls = []

                        class OldEvalsClient:
                            def push_samples(self, evaluation_id, samples):
                                calls.append((evaluation_id, samples))

                        class TerminalConsole:
                            is_terminal = True

                        original_console = evals.console
                        evals.console = TerminalConsole()
                        try:
                            samples = [{"example_id": "1"}]
                            evals._push_samples_with_progress(
                                OldEvalsClient(), "eval-123", samples
                            )
                        finally:
                            evals.console = original_console

                        if calls != [("eval-123", [{"example_id": "1"}])]:
                            raise SystemExit("old prime-evals push_samples compatibility failed")
                        '''
                    ),
                ],
                timeout=120,
            )


        def read_remote_env_slug() -> str:
            metadata_path = ENV_DIR / ".prime" / ".env-metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            owner = metadata["owner"]
            name = metadata["name"]
            return f"{owner}/{name}"


        def latest_eval_dir() -> Path:
            output_roots = (LAB_ROOT / "outputs", ENV_DIR / "outputs")
            candidates = sorted(
                (
                    path.parent
                    for output_root in output_roots
                    for path in output_root.glob("evals/**/results.jsonl")
                ),
                key=lambda path: path.stat().st_mtime,
            )
            if not candidates:
                searched = ", ".join(str(path) for path in output_roots)
                raise RuntimeError(f"No eval output directory found under: {searched}")
            return candidates[-1]


        def parse_hosted_eval_ids(output: str) -> list[str]:
            ids: list[str] = []
            for line in output.splitlines():
                if "Evaluation ID:" in line or "Evaluation IDs:" in line:
                    value = line.rsplit(":", 1)[-1]
                    ids.extend(part.strip() for part in value.split(",") if part.strip())
            return [eval_id for eval_id in ids if re.fullmatch(r"[A-Za-z0-9_-]+", eval_id)]


        def extract_json_object(raw: str) -> dict:
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1 or end < start:
                raise RuntimeError(f"Could not find JSON object in output: {raw}")
            return json.loads(raw[start : end + 1])


        def eval_status(eval_id: str) -> str:
            last_error: Exception | None = None
            for attempt in range(6):
                try:
                    raw = run(["prime", "eval", "get", eval_id, "--output", "json"], timeout=120)
                    data = extract_json_object(raw)
                    return str(
                        data.get("status") or data.get("data", {}).get("status") or "UNKNOWN"
                    )
                except Exception as exc:
                    last_error = exc
                    if attempt == 5:
                        break
                    time.sleep(5)
            raise RuntimeError(f"Could not fetch status for hosted eval {eval_id}") from last_error


        def wait_for_hosted_evals() -> None:
            deadline = time.monotonic() + CONFIG["hosted_timeout_minutes"] * 60
            pending = set(HOSTED_EVAL_IDS)
            while pending:
                for eval_id in list(pending):
                    status = eval_status(eval_id)
                    print(f"Hosted eval {eval_id} status: {status}", flush=True)
                    if status in TERMINAL_HOSTED_STATUSES:
                        if status != "COMPLETED":
                            raise RuntimeError(f"Hosted eval {eval_id} ended with {status}")
                        pending.remove(eval_id)
                if not pending:
                    return
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Timed out waiting for hosted evals: {sorted(pending)}")
                time.sleep(20)


        def best_effort_cancel_hosted_evals() -> None:
            for eval_id in HOSTED_EVAL_IDS:
                try:
                    run(["prime", "eval", "stop", eval_id], check=False, timeout=120)
                except Exception as exc:
                    print(
                        f"Warning: failed to cancel hosted eval {eval_id}: {exc}",
                        file=sys.stderr,
                    )


        def hosted_eval_checks() -> None:
            if CONFIG["hosted_mode"] == "skip":
                print("Skipping hosted eval checks by request.", flush=True)
                return

            direct = run(
                [
                    "prime",
                    "eval",
                    "run",
                    ENV_NAME,
                    "--hosted",
                    "--env-path",
                    str(ENV_DIR),
                    "-m",
                    CONFIG["model"],
                    "-n",
                    "1",
                    "-r",
                    "1",
                    "--timeout-minutes",
                    str(CONFIG["hosted_timeout_minutes"]),
                    "--eval-name",
                    f"{ENV_NAME} direct hosted",
                ],
                cwd=LAB_ROOT,
                timeout=600,
            )
            HOSTED_EVAL_IDS.extend(parse_hosted_eval_ids(direct))

            single_config = LAB_ROOT / "hosted-single.toml"
            write_file(
                single_config,
                f'''
                model = "{CONFIG["model"]}"
                num_examples = 1
                rollouts_per_example = 1
                timeout_minutes = {CONFIG["hosted_timeout_minutes"]}
                max_tokens = 64
                temperature = 0.0
                eval_name = "{ENV_NAME} single config hosted"

                [[eval]]
                env_id = "{REMOTE_ENV_SLUG}"
                env_args = {{ split = "eval", variant = "single" }}
                ''',
            )
            single = run(["prime", "eval", "run", str(single_config), "--hosted"], timeout=600)
            HOSTED_EVAL_IDS.extend(parse_hosted_eval_ids(single))

            multi_config = LAB_ROOT / "hosted-multi.toml"
            write_file(
                multi_config,
                f'''
                model = "{CONFIG["model"]}"
                timeout_minutes = {CONFIG["hosted_timeout_minutes"]}
                max_tokens = 64
                temperature = 0.0

                [[eval]]
                env_id = "{REMOTE_ENV_SLUG}"
                num_examples = 1
                rollouts_per_example = 1
                env_args = {{ split = "eval", variant = "multi-a" }}
                eval_name = "{ENV_NAME} multi config A"

                [[eval]]
                env_id = "{REMOTE_ENV_SLUG}"
                num_examples = 1
                rollouts_per_example = 2
                env_args = {{ split = "eval", variant = "multi-b" }}
                eval_name = "{ENV_NAME} multi config B"
                ''',
            )
            multi = run(["prime", "eval", "run", str(multi_config), "--hosted"], timeout=600)
            HOSTED_EVAL_IDS.extend(parse_hosted_eval_ids(multi))

            if not HOSTED_EVAL_IDS:
                raise RuntimeError(
                    "Hosted eval commands succeeded but no evaluation IDs were parsed"
                )

            # Verify records are visible before either waiting or cancelling.
            for eval_id in HOSTED_EVAL_IDS:
                print(f"Hosted eval {eval_id} initial status: {eval_status(eval_id)}", flush=True)

            if CONFIG["hosted_mode"] == "wait":
                wait_for_hosted_evals()
                HOSTED_EVAL_IDS.clear()
            else:
                best_effort_cancel_hosted_evals()
                HOSTED_EVAL_IDS.clear()


        def main() -> int:
            global REMOTE_ENV_SLUG
            HOME.mkdir(parents=True, exist_ok=True)
            LAB_ROOT.mkdir(parents=True, exist_ok=True)
            run(["tar", "-xzf", str(WORKSPACE / "prime-src.tar.gz"), "-C", str(WORKSPACE)])
            install_candidate_cli()
            installed_runtime_regression_checks()
            write_smoke_environment()

            try:
                run(
                    ["prime", "env", "push", "--path", str(ENV_DIR), "--visibility", "PRIVATE"],
                    timeout=900,
                )
                REMOTE_ENV_SLUG = read_remote_env_slug()
                print(f"Temporary Environment Hub slug: {REMOTE_ENV_SLUG}", flush=True)

                run(["prime", "env", "info", REMOTE_ENV_SLUG], timeout=180)
                run(["prime", "env", "install", REMOTE_ENV_SLUG, "--with", "pip"], timeout=900)

                run(
                    [
                        "prime",
                        "eval",
                        "run",
                        ENV_NAME,
                        "-m",
                        CONFIG["model"],
                        "-n",
                        "1",
                        "-r",
                        "1",
                        "--env-path",
                        str(ENV_DIR),
                        "--save-results",
                        "--skip-upload",
                    ],
                    cwd=LAB_ROOT,
                    timeout=900,
                )
                eval_dir = latest_eval_dir()
                run(
                    [
                        "prime",
                        "eval",
                        "push",
                        str(eval_dir),
                        "--env",
                        REMOTE_ENV_SLUG,
                        "--name",
                        f"{ENV_NAME} explicit eval push",
                        "--output",
                        "json",
                    ],
                    cwd=LAB_ROOT,
                    timeout=600,
                )

                run(
                    [
                        "prime",
                        "eval",
                        "run",
                        ENV_NAME,
                        "-m",
                        CONFIG["model"],
                        "-n",
                        "1",
                        "-r",
                        "1",
                        "--env-path",
                        str(ENV_DIR),
                    ],
                    cwd=LAB_ROOT,
                    timeout=900,
                )

                hosted_eval_checks()
                print("Prime release e2e checks passed.", flush=True)
                return 0
            finally:
                if HOSTED_EVAL_IDS:
                    best_effort_cancel_hosted_evals()
                if REMOTE_ENV_SLUG and CONFIG["cleanup_remote_env"]:
                    try:
                        run(
                            ["prime", "env", "delete", REMOTE_ENV_SLUG, "--force"],
                            check=False,
                            timeout=180,
                        )
                    except Exception as exc:
                        print(
                            f"Warning: failed to delete temporary environment "
                            f"{REMOTE_ENV_SLUG}: {exc}",
                            file=sys.stderr,
                        )


        if __name__ == "__main__":
            try:
                raise SystemExit(main())
            except CommandFailed as exc:
                raise SystemExit(exc.returncode) from exc
        """
    return textwrap.dedent(template).replace("__CONFIG_JSON__", config_literal)


def sandbox_environment_vars() -> dict[str, str]:
    names = [
        "PRIME_API_BASE_URL",
        "PRIME_BASE_URL",
        "PRIME_FRONTEND_URL",
        "PRIME_INFERENCE_URL",
        "PRIME_TEAM_ID",
        "PRIME_USER_ID",
    ]
    return {name: value for name in names if (value := os.environ.get(name))}


def sandbox_secrets() -> dict[str, str]:
    return {"PRIME_API_KEY": require_env("PRIME_API_KEY")}


def create_sandbox(client: SandboxClient, args: argparse.Namespace, run_suffix: str) -> str:
    name = f"prime-release-e2e-{run_suffix}"
    request = CreateSandboxRequest(
        name=name,
        docker_image=args.sandbox_image,
        cpu_cores=2,
        memory_gb=4,
        disk_size_gb=12,
        timeout_minutes=args.sandbox_timeout_minutes,
        environment_vars=sandbox_environment_vars() or None,
        secrets=sandbox_secrets(),
        labels=["prime-release-e2e"],
        region=args.sandbox_region,
    )
    sandbox = client.create(request)
    print(f"Created sandbox {sandbox.id} ({name})", flush=True)
    client.wait_for_creation(sandbox.id, max_attempts=max(60, args.sandbox_timeout_minutes * 30))
    return sandbox.id


def upload_bytes(client: SandboxClient, sandbox_id: str, path: str, content: str) -> None:
    data = content.encode("utf-8")
    client.upload_bytes(sandbox_id, path, data, filename=Path(path).name, timeout=300)


def run_in_sandbox(
    client: SandboxClient,
    sandbox_id: str,
    archive_path: Path,
    config: RemoteConfig,
    timeout_seconds: int,
) -> int:
    client.upload_file(sandbox_id, "/workspace/prime-src.tar.gz", str(archive_path), timeout=600)
    upload_bytes(client, sandbox_id, "/workspace/release_e2e_remote.py", remote_script(config))
    response = client.run_background_job(
        sandbox_id,
        "python /workspace/release_e2e_remote.py",
        timeout=timeout_seconds,
        poll_interval=5,
    )
    if response.stdout:
        print(response.stdout, end="" if response.stdout.endswith("\n") else "\n")
    if response.stderr:
        print(response.stderr, file=sys.stderr, end="" if response.stderr.endswith("\n") else "\n")
    return response.exit_code


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    if not (repo_root / "packages" / "prime" / "pyproject.toml").exists():
        raise SystemExit(f"Not a Prime monorepo checkout: {repo_root}")

    run_suffix = default_run_suffix()
    remote_config = RemoteConfig(
        model=args.model,
        hosted_mode=args.hosted_mode,
        env_prefix=sanitize_slug_part(args.env_prefix, max_length=24),
        hosted_timeout_minutes=args.hosted_timeout_minutes,
        cleanup_remote_env=not args.no_cleanup_remote_env,
        run_suffix=run_suffix,
    )

    archive = create_source_archive(repo_root)
    api_client = APIClient(api_key=require_env("PRIME_API_KEY"))
    sandbox_client = SandboxClient(api_client)
    sandbox_id: str | None = None
    try:
        sandbox_id = create_sandbox(sandbox_client, args, run_suffix)
        return run_in_sandbox(
            sandbox_client,
            sandbox_id,
            archive,
            remote_config,
            args.command_timeout_seconds,
        )
    finally:
        if sandbox_id and not args.keep_sandbox:
            print(f"Deleting sandbox {sandbox_id}", flush=True)
            try:
                sandbox_client.delete(sandbox_id)
            except Exception as exc:  # noqa: BLE001 - cleanup best effort
                print(f"Warning: failed to delete sandbox {sandbox_id}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
