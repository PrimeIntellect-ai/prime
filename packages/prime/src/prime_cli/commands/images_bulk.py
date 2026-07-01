import json
import re
import sys
import tarfile
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import click
import httpx
import typer
from gitignore_parser import parse_gitignore
from prime_sandboxes import (
    APIClient,
    APIError,
    Config,
    ImageVisibility,
    UnauthorizedError,
)
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from ..utils import get_console

console = get_console()

# Use a synthetic archive path to avoid collisions with Dockerfiles already in the context.
PACKAGED_DOCKERFILE_PATH = ".__prime_dockerfile__"

SUPPORTED_PLATFORMS = ("linux/amd64", "linux/arm64")

DEFAULT_MAX_IN_FLIGHT = 64
DEFAULT_BUILD_TIMEOUT_SECONDS = 1800
POLL_INTERVAL_SECONDS = 10.0
UPLOAD_TIMEOUT_SECONDS = 600.0

# Backoff schedule for rate-limit 429s
RATE_LIMIT_MAX_ATTEMPTS = 5
RATE_LIMIT_BACKOFF_INITIAL_SECONDS = 2.0
RATE_LIMIT_BACKOFF_MAX_SECONDS = 60.0

TERMINAL_BUILD_STATUSES = {"COMPLETED", "FAILED", "CANCELLED"}

FAILURE_TABLE_MAX_ROWS = 20

_TAG_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$")

_MANIFEST_KEYS = {"image", "context", "dockerfile", "platform"}


class BulkPushValidationError(Exception):
    """Raised when the build list cannot be resolved; carries all problems."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


class QuotaExceededError(APIError):
    """A 429 caused by the wallet's image count/storage quota, not the rate limiter."""


@dataclass
class BuildSpec:
    """One resolved build: where the context lives and what to call the image."""

    image_name: str
    image_tag: str
    context: Path
    dockerfile: Path
    platform: str
    source: str

    @property
    def image_ref(self) -> str:
        return f"{self.image_name}:{self.image_tag}"

    def to_manifest_line(self) -> dict[str, str]:
        """Serialize as a manifest entry (absolute paths, so it re-runs from any cwd)."""
        return {
            "image": self.image_ref,
            "context": str(self.context),
            "dockerfile": str(self.dockerfile),
            "platform": self.platform,
        }


@dataclass
class BuildOutcome:
    """Terminal result for one spec.

    ``status`` is a backend terminal status (COMPLETED/FAILED/CANCELLED) or a
    client-side one: SUBMIT_FAILED (initiate/upload/start failed), TIMEOUT
    (no terminal status within --build-timeout), SKIPPED (never submitted
    because the quota was exhausted mid-run).
    """

    spec: BuildSpec
    status: str
    build_id: Optional[str] = None
    full_image_path: Optional[str] = None
    error: Optional[str] = None


def package_build_context(context_path: Path, dockerfile_path: Path) -> str:
    """Create a tar.gz of the build context with the Dockerfile packaged at
    ``PACKAGED_DOCKERFILE_PATH``.

    Returns the temp tar path; the caller must unlink it. A .dockerignore
    matcher is applied so ignored paths (e.g. local .venv, node_modules)
    aren't uploaded. BuildKit looks for <Dockerfile>.dockerignore next to the
    Dockerfile first and falls back to <context>/.dockerignore, so mirror that.
    """
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
        tar_path = tmp_file.name

    per_dockerfile_ignore = dockerfile_path.with_name(dockerfile_path.name + ".dockerignore")
    root_dockerignore = context_path / ".dockerignore"
    if per_dockerfile_ignore.is_file():
        dockerignore_path: Optional[Path] = per_dockerfile_ignore
    elif root_dockerignore.is_file():
        dockerignore_path = root_dockerignore
    else:
        dockerignore_path = None
    ignore_matcher = (
        parse_gitignore(str(dockerignore_path), base_dir=str(context_path))
        if dockerignore_path is not None
        else None
    )

    def tar_filter(tarinfo: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
        if ignore_matcher is None:
            return tarinfo
        rel = tarinfo.name
        if rel.startswith("./"):
            rel = rel[2:]
        if not rel or rel == ".":
            return tarinfo
        if ignore_matcher(str(context_path / rel)):
            return None
        return tarinfo

    try:
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(context_path, arcname=".", filter=tar_filter)
            tar.add(dockerfile_path, arcname=PACKAGED_DOCKERFILE_PATH)
    except Exception:
        Path(tar_path).unlink(missing_ok=True)
        raise
    return tar_path


# ---------------------------------------------------------------------------
# Build-list resolution: JSONL manifest
# ---------------------------------------------------------------------------


def _duplicate_ref_problems(specs: list[BuildSpec], hint: str = "") -> list[str]:
    by_ref: dict[str, list[str]] = {}
    for spec in specs:
        by_ref.setdefault(spec.image_ref, []).append(spec.source)
    return [
        f"duplicate image reference '{ref}' ({', '.join(sources)}){hint}"
        for ref, sources in by_ref.items()
        if len(sources) > 1
    ]


def load_manifest(manifest_path: Path, default_platform: str) -> list[BuildSpec]:
    """Parse and fully validate a JSONL manifest."""
    base = manifest_path.parent
    problems: list[str] = []
    specs: list[BuildSpec] = []

    for lineno, raw_line in enumerate(manifest_path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        where = f"{manifest_path.name}:{lineno}"
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as e:
            problems.append(f"{where}: invalid JSON ({e})")
            continue
        if not isinstance(entry, dict):
            problems.append(f"{where}: expected a JSON object")
            continue
        unknown = sorted(set(entry) - _MANIFEST_KEYS)
        if unknown:
            problems.append(
                f"{where}: unknown key(s) {', '.join(unknown)} "
                f"(expected: {', '.join(sorted(_MANIFEST_KEYS))})"
            )
            continue

        image = entry.get("image")
        context = entry.get("context")
        if not image or not isinstance(image, str):
            problems.append(f"{where}: 'image' is required")
            continue
        if not context or not isinstance(context, str):
            problems.append(f"{where}: 'context' is required")
            continue
        if ":" in image:
            image_name, image_tag = image.rsplit(":", 1)
        else:
            image_name, image_tag = image, "latest"
        if not image_name or not image_tag:
            problems.append(f"{where}: invalid image reference '{image}'")
            continue
        if "/" in image_name:
            problems.append(
                f"{where}: image name cannot contain '/' ('{image_name}'); "
                "use simple names like 'myapp:v1'"
            )
            continue

        platform = entry.get("platform") or default_platform
        if platform not in SUPPORTED_PLATFORMS:
            problems.append(
                f"{where}: unsupported platform '{platform}' "
                f"(supported: {', '.join(SUPPORTED_PLATFORMS)})"
            )
            continue

        context_path = (base / context).resolve()
        dockerfile = entry.get("dockerfile")
        dockerfile_path = (
            (base / dockerfile).resolve() if dockerfile else context_path / "Dockerfile"
        )
        if not context_path.is_dir():
            problems.append(f"{where}: build context is not a directory: {context_path}")
            continue
        if not dockerfile_path.is_file():
            problems.append(f"{where}: Dockerfile not found: {dockerfile_path}")
            continue

        specs.append(
            BuildSpec(
                image_name=image_name,
                image_tag=image_tag,
                context=context_path,
                dockerfile=dockerfile_path,
                platform=platform,
                source=where,
            )
        )

    problems.extend(_duplicate_ref_problems(specs))
    if not specs and not problems:
        problems.append(f"{manifest_path.name}: manifest contains no builds")
    if problems:
        raise BulkPushValidationError(problems)
    return specs


# ---------------------------------------------------------------------------
# Build-list resolution: Harbor task directories
# ---------------------------------------------------------------------------


def _is_harbor_task_dir(path: Path) -> bool:
    return (path / "task.toml").is_file() and (path / "environment").is_dir()


def discover_harbor_tasks(root: Path) -> list[Path]:
    """Return Harbor task directories under root (or root itself if it is a task)."""
    if _is_harbor_task_dir(root):
        return [root]
    return sorted(
        (p for p in root.iterdir() if p.is_dir() and _is_harbor_task_dir(p)),
        key=lambda p: p.name,
    )


def sanitize_image_name(raw: str) -> str:
    """Normalize to a valid image name: lowercase [a-z0-9._-], alphanumeric ends."""
    name = raw.strip().lower()
    name = re.sub(r"[^a-z0-9._-]+", "-", name)
    name = re.sub(r"^[^a-z0-9]+", "", name)
    name = re.sub(r"[^a-z0-9]+$", "", name)
    return name


def _render_name_template(template: str, *, task_dir_name: str, toml_name: Optional[str]) -> str:
    values = {"dir": task_dir_name, "name": toml_name or task_dir_name}
    try:
        rendered = template.format(**values)
    except (KeyError, IndexError):
        raise BulkPushValidationError(
            [
                f"invalid --name-template '{template}': "
                "supported placeholders are {dir} and {name}"
            ]
        )
    return sanitize_image_name(rendered)


def load_harbor_specs(
    root: Path, *, tag: str, name_template: str, platform: str
) -> tuple[list[BuildSpec], list[tuple[str, str]]]:
    """Resolve build specs from a Harbor tasks directory.

    Returns (specs, skipped) where skipped is a list of (task name, reason)
    for tasks that have nothing to build: prebuilt ``docker_image`` tasks and
    tasks whose environment/ has no Dockerfile (e.g. compose-only).
    """
    tasks = discover_harbor_tasks(root)
    if not tasks:
        raise BulkPushValidationError(
            [
                f"no Harbor tasks found under {root} "
                "(a task directory contains task.toml and environment/)"
            ]
        )

    problems: list[str] = []
    specs: list[BuildSpec] = []
    skipped: list[tuple[str, str]] = []
    for task_dir in tasks:
        try:
            with open(task_dir / "task.toml", "rb") as f:
                config = tomllib.load(f)
        except Exception as e:
            problems.append(f"{task_dir.name}: failed to parse task.toml ({e})")
            continue

        environment = config.get("environment") or {}
        if isinstance(environment, dict) and environment.get("docker_image"):
            skipped.append((task_dir.name, f"uses prebuilt image {environment['docker_image']}"))
            continue
        dockerfile = task_dir / "environment" / "Dockerfile"
        if not dockerfile.is_file():
            skipped.append((task_dir.name, "no environment/Dockerfile"))
            continue

        task_section = config.get("task")
        toml_name = task_section.get("name") if isinstance(task_section, dict) else None
        image_name = _render_name_template(
            name_template, task_dir_name=task_dir.name, toml_name=toml_name
        )
        if not image_name:
            problems.append(f"{task_dir.name}: image name is empty after sanitization")
            continue

        specs.append(
            BuildSpec(
                image_name=image_name,
                image_tag=tag,
                context=task_dir / "environment",
                dockerfile=dockerfile,
                platform=platform,
                source=task_dir.name,
            )
        )

    problems.extend(_duplicate_ref_problems(specs, hint=" — use --name-template to disambiguate"))
    if not specs and not problems:
        problems.append(
            f"no buildable tasks under {root} "
            f"({len(skipped)} skipped: {', '.join(name for name, _ in skipped)})"
        )
    if problems:
        raise BulkPushValidationError(problems)
    return specs, skipped


# ---------------------------------------------------------------------------
# Submission + polling
# ---------------------------------------------------------------------------

_QUOTA_DETAIL_MARKERS = ("image limit exceeded", "image storage limit exceeded")


def _is_http_429(error: APIError) -> bool:
    return "HTTP 429" in str(error)


def _is_quota_429(error: APIError) -> bool:
    message = str(error).lower()
    return _is_http_429(error) and any(marker in message for marker in _QUOTA_DETAIL_MARKERS)


def _request_with_rate_limit_retry(
    client: APIClient, method: str, path: str, *, json_body: dict[str, Any]
) -> dict[str, Any]:
    """client.request with exponential backoff on rate-limit 429s.

    Wallet-quota 429s are re-raised as QuotaExceededError immediately — they
    cannot succeed on retry and the caller must stop submitting new builds.
    """
    delay = RATE_LIMIT_BACKOFF_INITIAL_SECONDS
    for attempt in range(1, RATE_LIMIT_MAX_ATTEMPTS + 1):
        try:
            return client.request(method, path, json=json_body)
        except UnauthorizedError:
            raise
        except APIError as e:
            if _is_quota_429(e):
                raise QuotaExceededError(str(e)) from e
            if not _is_http_429(e) or attempt == RATE_LIMIT_MAX_ATTEMPTS:
                raise
            time.sleep(delay)
            delay = min(delay * 2, RATE_LIMIT_BACKOFF_MAX_SECONDS)
    raise AssertionError("unreachable")


def _submit_build(
    client: APIClient,
    spec: BuildSpec,
    *,
    team_id: Optional[str],
    visibility: Optional[ImageVisibility],
) -> tuple[str, str]:
    """Run the initiate -> upload context -> start flow for one build.

    Returns (build_id, full_image_path).
    """
    tar_path = package_build_context(spec.context, spec.dockerfile)
    try:
        payload: dict[str, Any] = {
            "image_name": spec.image_name,
            "image_tag": spec.image_tag,
            "dockerfile_path": PACKAGED_DOCKERFILE_PATH,
            "platform": spec.platform,
        }
        if team_id:
            payload["team_id"] = team_id
        if visibility is not None:
            payload["visibility"] = visibility.value

        response = _request_with_rate_limit_retry(
            client, "POST", "/images/build", json_body=payload
        )
        build_id = response.get("build_id")
        upload_url = response.get("upload_url")
        if not build_id or not upload_url:
            raise APIError("invalid response from server (missing build_id or upload_url)")
        full_image_path = response.get("fullImagePath") or spec.image_ref

        with open(tar_path, "rb") as f:
            upload_response = httpx.put(
                upload_url,
                content=f,
                headers={"Content-Type": "application/octet-stream"},
                timeout=UPLOAD_TIMEOUT_SECONDS,
            )
            upload_response.raise_for_status()

        _request_with_rate_limit_retry(
            client,
            "POST",
            f"/images/build/{build_id}/start",
            json_body={"context_uploaded": True},
        )
        return build_id, full_image_path
    finally:
        Path(tar_path).unlink(missing_ok=True)


@dataclass
class _InFlightBuild:
    spec: BuildSpec
    full_image_path: str
    deadline: float


def run_bulk_push(
    client: APIClient,
    specs: list[BuildSpec],
    *,
    team_id: Optional[str],
    visibility: Optional[ImageVisibility],
    concurrency: int,
    build_timeout: int,
) -> list[BuildOutcome]:
    """Submit builds with a sliding window and poll them to terminal status.

    A finished build immediately frees a slot for the next queued build, so
    one slow build never blocks the rest of a "batch". Once the wallet quota
    is hit, submission stops (remaining specs become SKIPPED) but builds
    already in flight are still polled to completion.
    """
    pending: deque[BuildSpec] = deque(specs)
    in_flight: dict[str, _InFlightBuild] = {}
    outcomes: list[BuildOutcome] = []
    quota_error: Optional[str] = None

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )
    with progress:
        task_id = progress.add_task("Pushing images", total=len(specs))

        def record(outcome: BuildOutcome) -> None:
            outcomes.append(outcome)
            progress.advance(task_id)
            if outcome.status == "COMPLETED":
                progress.console.print(
                    f"[green]✓[/green] {outcome.full_image_path or outcome.spec.image_ref}"
                )
            elif outcome.status != "SKIPPED":  # skips are summarized once at the end
                progress.console.print(
                    f"[red]✗ {outcome.spec.image_ref}[/red] "
                    f"[dim]({outcome.status.lower()}: {outcome.error})[/dim]"
                )

        while pending or in_flight:
            while pending and quota_error is None and len(in_flight) < concurrency:
                spec = pending.popleft()
                try:
                    build_id, full_image_path = _submit_build(
                        client, spec, team_id=team_id, visibility=visibility
                    )
                except QuotaExceededError as e:
                    quota_error = str(e)
                    record(BuildOutcome(spec=spec, status="SUBMIT_FAILED", error=str(e)))
                except UnauthorizedError:
                    raise
                except (APIError, httpx.HTTPError, OSError) as e:
                    record(BuildOutcome(spec=spec, status="SUBMIT_FAILED", error=str(e)))
                else:
                    in_flight[build_id] = _InFlightBuild(
                        spec=spec,
                        full_image_path=full_image_path,
                        deadline=time.monotonic() + build_timeout,
                    )

            if quota_error is not None and pending:
                for spec in pending:
                    record(
                        BuildOutcome(
                            spec=spec,
                            status="SKIPPED",
                            error="not submitted: image quota exceeded",
                        )
                    )
                pending.clear()

            if not in_flight:
                break

            time.sleep(POLL_INTERVAL_SECONDS)

            for build_id, entry in list(in_flight.items()):
                try:
                    status_response = client.request("GET", f"/images/build/{build_id}")
                    build_status = str(status_response.get("status") or "")
                except UnauthorizedError:
                    raise
                except APIError:
                    build_status = ""  # transient poll failure; the deadline still applies

                if build_status in TERMINAL_BUILD_STATUSES:
                    del in_flight[build_id]
                    record(
                        BuildOutcome(
                            spec=entry.spec,
                            status=build_status,
                            build_id=build_id,
                            full_image_path=entry.full_image_path,
                            error=(
                                None
                                if build_status == "COMPLETED"
                                else f"build ended as {build_status}"
                            ),
                        )
                    )
                elif time.monotonic() > entry.deadline:
                    del in_flight[build_id]
                    record(
                        BuildOutcome(
                            spec=entry.spec,
                            status="TIMEOUT",
                            build_id=build_id,
                            full_image_path=entry.full_image_path,
                            error=(
                                f"no terminal status after {build_timeout}s "
                                "(the build may still finish server-side)"
                            ),
                        )
                    )

    return outcomes


def _write_failures_manifest(path: Path, failures: list[BuildOutcome]) -> None:
    lines = [json.dumps(outcome.spec.to_manifest_line()) for outcome in failures]
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


def push_bulk(
    manifest: Optional[Path] = typer.Option(
        None,
        "--manifest",
        "-m",
        help=(
            "JSONL manifest of builds; each line is "
            '{"image": "name:tag", "context": "./dir", "dockerfile"?: "...", "platform"?: "..."} '
            "with paths relative to the manifest file"
        ),
    ),
    harbor: Optional[Path] = typer.Option(
        None,
        "--harbor",
        help=(
            "Harbor task directory (or directory of tasks); each task's "
            "environment/ folder is the build context"
        ),
    ),
    tag: str = typer.Option("latest", "--tag", help="Image tag for Harbor mode"),
    name_template: str = typer.Option(
        "{dir}",
        "--name-template",
        help=(
            "Image name template for Harbor mode; placeholders: {dir} (task directory "
            "name) and {name} (task.toml [task].name). The result is sanitized to a "
            "valid image name"
        ),
    ),
    platform: str = typer.Option(
        "linux/amd64",
        "--platform",
        click_type=click.Choice(list(SUPPORTED_PLATFORMS)),
        help="Default target platform (manifest lines may override per build)",
    ),
    public: bool = typer.Option(
        False, "--public", help="Make the images public when the builds complete"
    ),
    private: bool = typer.Option(
        False, "--private", help="Make the images private when the builds complete"
    ),
    concurrency: int = typer.Option(
        DEFAULT_MAX_IN_FLIGHT, "--concurrency", help="Maximum builds in flight at once"
    ),
    build_timeout: int = typer.Option(
        DEFAULT_BUILD_TIMEOUT_SECONDS,
        "--build-timeout",
        help="Seconds to wait for a single build before giving up on it",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Resolve and print the build list without pushing"
    ),
    failures_out: Path = typer.Option(
        Path("push-bulk-failures.jsonl"),
        "--failures-out",
        help="Where to write a re-runnable manifest of failed builds",
    ),
):
    """
    Build and push many Docker images in one command.

    Reads builds from a JSONL manifest (--manifest) or a Harbor tasks
    directory (--harbor), validates everything up front, then keeps up to
    --concurrency builds running server-side, starting the next build as soon
    as one finishes. Failed builds are written to a manifest you can re-run.

    \b
    Manifest format (one JSON object per line; paths relative to the manifest):
        {"image": "myapp:v1", "context": "./apps/myapp"}
        {"image": "other:v2", "context": "./other", "dockerfile": "./docker/Dockerfile.prod"}

    \b
    Harbor mode finds tasks (directories containing task.toml and environment/)
    and uses each task's environment/ folder as the build context. Images are
    named after the task directory (override with --name-template) and tagged
    with --tag. Tasks with a prebuilt docker_image or no environment/Dockerfile
    are skipped.

    \b
    Examples:
        prime images push-bulk --manifest builds.jsonl
        prime images push-bulk --harbor ./tasks --tag v1
        prime images push-bulk --harbor ./tasks --name-template "swe-{dir}" --dry-run
        prime images push-bulk --manifest push-bulk-failures.jsonl
    """
    try:
        if (manifest is None) == (harbor is None):
            console.print("[red]Error: Provide exactly one of --manifest or --harbor[/red]")
            raise typer.Exit(1)
        if public and private:
            console.print("[red]Error: --public and --private cannot be used together[/red]")
            raise typer.Exit(1)
        if concurrency < 1:
            console.print("[red]Error: --concurrency must be at least 1[/red]")
            raise typer.Exit(1)
        if build_timeout < 1:
            console.print("[red]Error: --build-timeout must be at least 1[/red]")
            raise typer.Exit(1)
        if manifest is not None and (tag != "latest" or name_template != "{dir}"):
            console.print("[red]Error: --tag and --name-template only apply to --harbor mode[/red]")
            raise typer.Exit(1)

        skipped: list[tuple[str, str]] = []
        try:
            if manifest is not None:
                manifest_path = manifest.resolve()
                if not manifest_path.is_file():
                    raise BulkPushValidationError([f"manifest not found: {manifest_path}"])
                source_desc = f"manifest {manifest_path}"
                specs = load_manifest(manifest_path, default_platform=platform)
            else:
                assert harbor is not None
                harbor_root = harbor.resolve()
                if not harbor_root.is_dir():
                    raise BulkPushValidationError(
                        [f"Harbor task directory not found: {harbor_root}"]
                    )
                if not _TAG_RE.match(tag):
                    raise BulkPushValidationError([f"invalid image tag '{tag}'"])
                source_desc = f"Harbor tasks in {harbor_root}"
                specs, skipped = load_harbor_specs(
                    harbor_root, tag=tag, name_template=name_template, platform=platform
                )
        except BulkPushValidationError as e:
            console.print(
                f"[red]Error: cannot start bulk push ({len(e.problems)} problem(s)):[/red]"
            )
            for problem in e.problems:
                console.print(f"[red]  - {problem}[/red]")
            raise typer.Exit(1)

        for task_name, reason in skipped:
            console.print(f"[yellow]Skipping {task_name}: {reason}[/yellow]")

        if dry_run:
            table = Table(title=f"Resolved {len(specs)} build(s)")
            table.add_column("Image", style="cyan", no_wrap=True)
            table.add_column("Context")
            table.add_column("Dockerfile")
            table.add_column("Platform", no_wrap=True)
            table.add_column("From", style="dim")
            for spec in specs:
                table.add_row(
                    spec.image_ref,
                    str(spec.context),
                    str(spec.dockerfile),
                    spec.platform,
                    spec.source,
                )
            console.print(table)
            console.print("[dim]Dry run only — re-run without --dry-run to push.[/dim]")
            return

        config = Config()
        visibility: Optional[ImageVisibility] = None
        if public:
            visibility = ImageVisibility.PUBLIC
        elif private:
            visibility = ImageVisibility.PRIVATE

        console.print(
            f"[bold blue]Bulk pushing {len(specs)} image(s)[/bold blue] [dim]({source_desc})[/dim]"
        )
        if config.team_id:
            console.print(f"[dim]Team: {config.team_id}[/dim]")
        if visibility is not None:
            console.print(f"[dim]Visibility: {visibility.value}[/dim]")
        else:
            console.print(
                "[dim]Visibility: PRIVATE for new images "
                "(existing tags keep their current visibility)[/dim]"
            )
        console.print(
            f"[dim]Up to {concurrency} builds in flight; "
            f"polling every {int(POLL_INTERVAL_SECONDS)}s[/dim]"
        )
        console.print()

        client = APIClient()
        outcomes = run_bulk_push(
            client,
            specs,
            team_id=config.team_id or None,
            visibility=visibility,
            concurrency=concurrency,
            build_timeout=build_timeout,
        )

        failures = [o for o in outcomes if o.status != "COMPLETED"]
        completed_count = len(outcomes) - len(failures)
        console.print()
        console.print(f"[bold]{completed_count}/{len(outcomes)} builds completed[/bold]")
        if not failures:
            console.print("[bold green]All images pushed successfully![/bold green]")
            console.print()
            console.print("[dim]Use them with: prime sandbox create <image reference>[/dim]")
            return

        failure_table = Table(title=f"{len(failures)} build(s) did not complete")
        failure_table.add_column("Image", style="cyan", no_wrap=True)
        failure_table.add_column("Status", no_wrap=True)
        failure_table.add_column("From", style="dim")
        failure_table.add_column("Error")
        for outcome in failures[:FAILURE_TABLE_MAX_ROWS]:
            failure_table.add_row(
                outcome.spec.image_ref, outcome.status, outcome.spec.source, outcome.error or ""
            )
        console.print(failure_table)
        if len(failures) > FAILURE_TABLE_MAX_ROWS:
            console.print(
                f"[dim]... and {len(failures) - FAILURE_TABLE_MAX_ROWS} more, "
                f"all included in {failures_out}[/dim]"
            )
        if any("limit exceeded" in (o.error or "").lower() for o in failures):
            console.print(
                "[red]Image quota reached — delete unused images (prime images delete) "
                "or request a higher limit, then retry.[/red]"
            )

        _write_failures_manifest(failures_out, failures)
        console.print()
        console.print(f"Wrote {len(failures)} failed build(s) to [bold]{failures_out}[/bold]")
        console.print(f"[dim]Retry with: prime images push-bulk --manifest {failures_out}[/dim]")
        raise typer.Exit(1)

    except UnauthorizedError:
        console.print("[red]Error: Not authenticated. Please run 'prime login' first.[/red]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print(
            "\n[yellow]Cancelled. Builds already started keep running server-side; "
            "check them with 'prime images list'.[/yellow]"
        )
        raise typer.Exit(1)
