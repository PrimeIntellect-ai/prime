"""Commands for managing Docker images in Prime Intellect registry."""

import json
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import click
import httpx
import typer
from gitignore_parser import parse_gitignore
from prime_sandboxes import APIClient, APIError, Config, UnauthorizedError
from rich.table import Table

from ..utils import PlainTyper, get_console, json_output_help, validate_output_format

app = PlainTyper(help="Manage Docker images in Prime Intellect registry", no_args_is_help=True)
console = get_console()
# Use a synthetic archive path to avoid collisions with Dockerfiles already in the context.
PACKAGED_DOCKERFILE_PATH = ".__prime_dockerfile__"

config = Config()

LIST_IMAGES_JSON_HELP = json_output_help(
    "Raw API response is printed unchanged.",
    ".data[] = {displayRef?, fullImagePath?, imageName, imageTag, status, "
    "artifactType, ownerType, sizeBytes?, createdAt, pushedAt?}",
)

# ---------------------------------------------------------------------------
# Helpers for rendering `prime images list`
# ---------------------------------------------------------------------------

# Raw artifact row as returned by ``GET /v1/images``. The backend schema is
# documented in ``LIST_IMAGES_JSON_HELP`` above; we keep the dict shape loose
# here because the server may add new optional fields over time.
ImageRow = dict[str, Any]


@dataclass
class ArtifactPartition:
    """Per-artifact-type view of a grouped image.

    Holds the single most recently updated row for this artifact type. The
    status of that row is what gets rendered — we intentionally ignore older
    rows (including stale ``BUILDING`` / ``PENDING`` entries the backend may
    have orphaned) so the display reflects the current truth rather than a
    composite derived from history.
    """

    latest: Optional[ImageRow] = None

    def is_empty(self) -> bool:
        """True when no row exists for this artifact type."""
        return self.latest is None


# Mapping of artifact type (e.g. ``CONTAINER_IMAGE``) to its partition bucket.
PartitionMap = dict[str, ArtifactPartition]

# Timestamp priority used when picking the single latest row per artifact
# type. ``pushedAt`` wins for completed uploads; ``completedAt`` covers
# failed/cancelled terminal states; ``startedAt`` and ``createdAt`` are
# ultimate fallbacks for rows that never finished.
_LATEST_ROW_KEYS: tuple[str, ...] = ("pushedAt", "completedAt", "startedAt", "createdAt")


def _parse_ts(value: Any) -> Optional[datetime]:
    """Parse an ISO8601 timestamp (possibly ``Z`` suffixed) as a tz-aware UTC datetime.

    Naive timestamps (the backend emits ``createdAt`` as naive UTC, e.g. from
    ``datetime.utcnow()``) are treated as UTC so comparisons across the dataset
    are consistent. Returns ``None`` on failure.
    """
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _latest(rows: list[ImageRow], *keys: str) -> Optional[tuple[ImageRow, Optional[datetime]]]:
    """Return the row whose first parseable timestamp (across ``keys``) is newest.

    Each row is evaluated by walking ``keys`` in order and taking the first
    value that ``_parse_ts`` accepts. This means an unparseable-but-truthy
    value (e.g. a malformed date string) is skipped rather than short-circuiting
    selection — and the *same* parsed timestamp is returned alongside the row
    so callers don't re-read a potentially different field.

    The returned ``datetime`` is ``None`` only in the fallback case where no
    row had any parseable timestamp; we still return the first row so Size /
    Reference fields can be derived, but callers should treat "no timestamp"
    as a signal to fall through to a lower-priority tier.

    Returns ``None`` if ``rows`` is empty.
    """
    if not rows:
        return None
    best: Optional[ImageRow] = None
    best_ts: Optional[datetime] = None
    for row in rows:
        ts: Optional[datetime] = None
        for k in keys:
            ts = _parse_ts(row.get(k))
            if ts is not None:
                break
        if ts is None:
            continue
        if best_ts is None or ts > best_ts:
            best = row
            best_ts = ts
    if best is None:
        return rows[0], None
    return best, best_ts


def _coerce_artifact_type(value: Any) -> str:
    """Normalise an ``artifactType`` value into a usable string key.

    Defensive against a malformed backend payload (``null``, missing, or a
    non-string value) so that downstream ``sorted(partition)`` and label
    rendering never blow up on mixed key types.
    """
    if isinstance(value, str) and value:
        return value
    return "CONTAINER_IMAGE"


def _pick_row(rows: list[ImageRow], *keys: str) -> Optional[ImageRow]:
    """Return just the row component from ``_latest`` (drops the timestamp)."""
    result = _latest(rows, *keys)
    return result[0] if result is not None else None


def _partition_group(artifacts: list[ImageRow]) -> PartitionMap:
    """Group artifact rows by type, keeping only the most recent row per type.

    For a single ``imageName:imageTag`` the backend returns one row per
    (build, artifact type) plus any completed artifacts from the user images
    table. We pick the single newest row per artifact type (ordered by
    ``pushedAt → completedAt → startedAt → createdAt``) and render its raw
    ``status`` verbatim. Older rows — including stuck ``BUILDING`` zombies
    and stale failures — are simply not considered.
    """
    by_type: dict[str, list[ImageRow]] = {}
    for art in artifacts:
        t = _coerce_artifact_type(art.get("artifactType"))
        by_type.setdefault(t, []).append(art)

    result: PartitionMap = {}
    for art_type, rows in by_type.items():
        latest_row = _pick_row(rows, *_LATEST_ROW_KEYS)
        result[art_type] = ArtifactPartition(latest=latest_row)
    return result


_TYPE_LABELS: tuple[tuple[str, str], ...] = (
    ("CONTAINER_IMAGE", "[cyan]Container[/cyan]"),
    ("VM_SANDBOX", "[magenta]VM[/magenta]"),
)


def _ordered_present_types(partition: PartitionMap) -> list[tuple[str, str]]:
    """Return artifact types present in ``partition`` in display order.

    Container first, then VM, then any future types in sorted order. Types
    whose per-artifact partition bucket is completely empty (no completed,
    no active, no failed_only, no other) are skipped so we don't render
    dead slots.
    """
    ordered: list[tuple[str, str]] = []
    for art_type, label in _TYPE_LABELS:
        part = partition.get(art_type)
        if part is not None and not part.is_empty():
            ordered.append((art_type, label))
    for art_type in sorted(partition):
        if art_type in {"CONTAINER_IMAGE", "VM_SANDBOX"}:
            continue
        if not partition[art_type].is_empty():
            label = f"[white]{str(art_type).replace('_', ' ').title()}[/white]"
            ordered.append((art_type, label))
    return ordered


def _render_type_column(partition: PartitionMap) -> str:
    """Build the Type cell: ``Container / VM`` with color, only for types present."""
    parts = [label for _art_type, label in _ordered_present_types(partition)]
    return " / ".join(parts) if parts else "[dim]—[/dim]"


_STATUS_LABELS: dict[str, str] = {
    "COMPLETED": "[green]Ready[/green]",
    "BUILDING": "[yellow]Building[/yellow]",
    "UPLOADING": "[yellow]Uploading[/yellow]",
    "PENDING": "[blue]Pending[/blue]",
    "FAILED": "[red]Failed[/red]",
    "CANCELLED": "[dim]Cancelled[/dim]",
}


def _render_status_slot(part: Optional[ArtifactPartition]) -> str:
    """Render the raw status of the latest row for this artifact type.

    Unknown statuses (e.g. a future backend addition) are rendered as a
    dim title-cased label rather than being dropped.
    """
    if part is None or part.latest is None:
        return "[dim]—[/dim]"
    status = str(part.latest.get("status") or "UNKNOWN")
    return _STATUS_LABELS.get(status, f"[dim]{status.title()}[/dim]")


def _render_status_column(partition: PartitionMap) -> str:
    """Build the Status cell as positional slots aligned with the Type column.

    Example: if Type is ``Container / VM``, Status for ``rehl:latest`` with a
    container that's Ready and a VM that's Failed becomes ``Ready / Failed``.
    When an artifact has an active rebuild on top of a completed image, its
    slot is ``(rebuilding)``.
    """
    ordered = _ordered_present_types(partition)
    if not ordered:
        return "[dim]—[/dim]"
    return " / ".join(_render_status_slot(partition.get(art_type)) for art_type, _ in ordered)


def _render_image_reference(img: ImageRow, *, is_team_listing: bool) -> str:
    """Render the user-facing image reference.

    The owner prefix (``{userId}/`` or ``team-{teamId}/``) is **always**
    preserved so the full string is a valid reference that can be pasted
    directly into downstream commands (e.g. ``prime sandbox create``),
    which require the fully-qualified ``<userId>/<imageName>:<tag>`` form.

    Visual truncation (if any) is applied separately by
    :func:`_truncate_ref_left` so that when the terminal is too narrow the
    *owner prefix* is what gets clipped, not the image ``name:tag``.
    """
    del is_team_listing
    return (
        img.get("displayRef")
        or img.get("fullImagePath")
        or f"{img.get('imageName', 'unknown')}:{img.get('imageTag', 'latest')}"
    )


def _truncate_ref_left(ref: str, max_width: Optional[int]) -> str:
    """Left-ellipsize ``ref`` so the tail (``name:tag``) stays visible.

    If ``ref`` fits within ``max_width`` it's returned unchanged. Otherwise,
    the head of the string (the owner prefix) is dropped and replaced with a
    single ``…`` character. This matches the requested display style:

        cmkrcib4x00004kjyxq48…/nvidia-basic-dev:latest → …/nvidia-basic-dev:latest

    ``max_width`` must be at least 2; smaller values are clamped.
    """
    if max_width is None or max_width <= 0:
        return ref
    if len(ref) <= max_width:
        return ref
    max_width = max(max_width, 2)
    return "…" + ref[-(max_width - 1) :]


def _image_ref_column_width(console_width: int, is_team_listing: bool) -> int:
    """Compute a budget for the Image Reference column based on terminal width.

    The remaining columns have roughly fixed widths (worst-case labels):

        Type     ~14 chars ("Container / VM")
        Status   ~20 chars ("Uploading / Uploading")
        Size     ~10 chars
        Created  ~17 chars ("YYYY-MM-DD HH:MM ")
        Owner    ~10 chars (team listings only)
        borders + padding ~ 3 chars per column

    We subtract these from the terminal width and clamp to a reasonable range
    so the column is neither absurdly wide on large terminals nor unusable on
    tiny ones.
    """
    reserved = 14 + 20 + 10 + 17 + (10 if is_team_listing else 0)
    num_cols = 5 + (1 if is_team_listing else 0)
    reserved += 3 * num_cols
    budget = console_width - reserved
    return max(30, min(80, budget))


def _completed_size_mb(partition: PartitionMap) -> str:
    """Sum sizes of the latest COMPLETED rows per artifact type.

    Only completed artifacts carry a meaningful ``sizeBytes``; in-flight and
    failed rows contribute nothing, so summing across the latest-per-type
    gives the current on-disk footprint of the image.
    """
    total = 0
    for part in partition.values():
        row = part.latest
        if row is None or row.get("status") != "COMPLETED":
            continue
        total += row.get("sizeBytes") or 0
    if total <= 0:
        return "[dim]—[/dim]"
    return f"{total / 1024 / 1024:.1f} MB"


def _pick_display_datetime(partition: PartitionMap) -> Optional[datetime]:
    """Return the newest timestamp across the latest row of each artifact type."""
    latest_rows: list[ImageRow] = [
        part.latest for part in partition.values() if part.latest is not None
    ]
    if not latest_rows:
        return None
    result = _latest(latest_rows, *_LATEST_ROW_KEYS)
    if result is None:
        return None
    return result[1]


def _display_created(partition: PartitionMap) -> str:
    """Format the Created column value for a grouped image row."""
    ts = _pick_display_datetime(partition)
    return ts.strftime("%Y-%m-%d %H:%M") if ts is not None else ""


def _group_sort_key(partition: PartitionMap) -> datetime:
    """Key function to sort groups newest-first by their display timestamp."""
    ts = _pick_display_datetime(partition)
    return ts if ts is not None else datetime.min.replace(tzinfo=timezone.utc)


@app.command("push")
def push_image(
    image_reference: str = typer.Argument(
        ..., help="Image reference (e.g., 'myapp:v1.0.0' or 'myapp:latest')"
    ),
    context: str = typer.Option(".", "--context", "-c", help="Build context directory"),
    dockerfile: Optional[str] = typer.Option(
        None,
        "--dockerfile",
        "-f",
        help="Path to Dockerfile",
        show_default="<context>/Dockerfile",
    ),
    platform: str = typer.Option(
        "linux/amd64",
        "--platform",
        click_type=click.Choice(["linux/amd64", "linux/arm64"]),
        help="Target platform (defaults to linux/amd64 for Kubernetes compatibility)",
    ),
):
    """
    Build and push a Docker image to Prime Intellect registry.

    \b
    Examples:
        prime images push myapp:v1.0.0
        prime images push myapp:latest --context ./app --dockerfile ../docker/Dockerfile.prod
        prime images push myapp:v1 --platform linux/arm64
    """
    try:
        # Parse image reference
        if ":" in image_reference:
            image_name, image_tag = image_reference.rsplit(":", 1)
        else:
            image_name = image_reference
            image_tag = "latest"

        # Validate image name doesn't contain slashes
        if "/" in image_name:
            console.print(
                "[red]Error: Image name cannot contain '/'. "
                "Use simple names like 'myapp:v1.0.0'.[/red]"
            )
            raise typer.Exit(1)

        console.print(
            f"[bold blue]Building and pushing image:[/bold blue] {image_name}:{image_tag}"
        )
        if config.team_id:
            console.print(f"[dim]Team: {config.team_id}[/dim]")
        console.print()

        # Initialize API client
        client = APIClient()

        context_path = Path(context).resolve()
        dockerfile_path = Path(dockerfile).resolve() if dockerfile else context_path / "Dockerfile"

        if not context_path.exists():
            console.print(f"[red]Error: Build context not found at {context_path}[/red]")
            raise typer.Exit(1)

        if not context_path.is_dir():
            console.print(f"[red]Error: Build context must be a directory: {context_path}[/red]")
            raise typer.Exit(1)

        if not dockerfile_path.exists():
            console.print(f"[red]Error: Dockerfile not found at {dockerfile_path}[/red]")
            raise typer.Exit(1)

        if not dockerfile_path.is_file():
            console.print(f"[red]Error: Dockerfile must be a file: {dockerfile_path}[/red]")
            raise typer.Exit(1)

        # Create tar.gz of build context
        console.print("[cyan]Preparing build context...[/cyan]")
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
            tar_path = tmp_file.name

        # Build a .dockerignore matcher so we don't upload ignored paths
        # (e.g. local .venv, node_modules) with the context. BuildKit
        # looks for <Dockerfile>.dockerignore next to the Dockerfile first
        # and falls back to <context>/.dockerignore, so mirror that.
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

            tar_size_mb = Path(tar_path).stat().st_size / (1024 * 1024)
            console.print(f"[green]✓[/green] Build context packaged ({tar_size_mb:.2f} MB)")
            console.print()

            # Initialize build
            console.print("[cyan]Initiating build...[/cyan]")
            try:
                build_payload = {
                    "image_name": image_name,
                    "image_tag": image_tag,
                    "dockerfile_path": PACKAGED_DOCKERFILE_PATH,
                    "platform": platform,
                }
                if config.team_id:
                    build_payload["team_id"] = config.team_id

                build_response = client.request(
                    "POST",
                    "/images/build",
                    json=build_payload,
                )
            except UnauthorizedError:
                console.print(
                    "[red]Error: Not authenticated. Please run 'prime login' first.[/red]"
                )
                raise typer.Exit(1)
            except APIError as e:
                console.print(f"[red]Error: Failed to initiate build: {e}[/red]")
                raise typer.Exit(1)

            build_id = build_response.get("build_id")
            upload_url = build_response.get("upload_url")
            if not build_id or not upload_url:
                console.print(
                    "[red]Error: Invalid response from server "
                    "(missing build_id or upload_url)[/red]"
                )
                raise typer.Exit(1)
            full_image_path = build_response.get("fullImagePath") or f"{image_name}:{image_tag}"

            console.print("[green]✓[/green] Build initiated")
            console.print()

            # Upload build context to GCS
            console.print("[cyan]Uploading build context...[/cyan]")
            try:
                with open(tar_path, "rb") as f:
                    upload_response = httpx.put(
                        upload_url,
                        content=f,
                        headers={"Content-Type": "application/octet-stream"},
                        timeout=600.0,
                    )
                    upload_response.raise_for_status()
            except httpx.HTTPError as e:
                console.print(f"[red]Upload failed: {e}[/red]")
                raise typer.Exit(1)

            console.print("[green]✓[/green] Build context uploaded")
            console.print()

            # Start the build
            console.print("[cyan]Starting build...[/cyan]")
            try:
                client.request(
                    "POST",
                    f"/images/build/{build_id}/start",
                    json={"context_uploaded": True},
                )
            except APIError as e:
                console.print(f"[red]Error: Failed to start build: {e}[/red]")
                raise typer.Exit(1)

            console.print("[green]✓[/green] Build started")
            console.print()

            console.print("[bold green]Build initiated successfully![/bold green]")
            console.print()
            console.print(f"[bold]Build ID:[/bold] {build_id}")
            console.print(f"[bold]Image:[/bold] {full_image_path}")
            console.print()
            console.print("[cyan]Your image is being built.[/cyan]")
            console.print()
            console.print("[bold]Check build status:[/bold]")
            console.print("  prime images list")
            console.print()
            console.print(
                "[dim]The build typically takes a few minutes depending on image complexity.[/dim]"
            )
            console.print(
                "[dim]Once completed, you can use it with: "
                f"prime sandbox create {full_image_path}[/dim]"
            )
            console.print()

        finally:
            # Clean up temporary tar file
            try:
                Path(tar_path).unlink()
            except Exception:
                pass

    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        raise typer.Exit(1)


@app.command("list", epilog=LIST_IMAGES_JSON_HELP)
def list_images(
    output: str = typer.Option("table", "--output", "-o", help="Output format (table or json)"),
    all_images: bool = typer.Option(
        False, "--all", "-a", help="[Deprecated] Show all accessible images (personal + team)"
    ),
):
    """
    List all images you've pushed to Prime Intellect registry.

    Shows personal images by default, or team images when a team is configured.

    \b
    Examples:
        prime images list
        prime images list --output json
    """
    validate_output_format(output, console)

    if all_images and output != "json":
        console.print(
            "[yellow]Warning: --all flag is deprecated and will be removed in a future release. "
            "Images are now scoped to your current context (personal or team).[/yellow]"
        )
        console.print()
    try:
        client = APIClient()

        # Build query params
        params: dict[str, str] = {}
        if config.team_id:
            params["teamId"] = config.team_id

        response = client.request("GET", "/images", params=params if params else None)
        images: list[ImageRow] = response.get("data", [])

        if not images:
            console.print("[yellow]No images or builds found.[/yellow]")
            console.print("Push an image with: [bold]prime images push <name>:<tag>[/bold]")
            return

        if output == "json":
            console.print(json.dumps(response, indent=2))
            return

        # Table output
        is_team_listing: bool = bool(config.team_id)
        title: str
        if is_team_listing:
            title = f"Team Docker Images (team: {config.team_id})"
        else:
            title = "Personal Docker Images"

        grouped: dict[str, list[ImageRow]] = {}
        for img in images:
            owner_scope = img.get("teamId") or img.get("ownerType", "personal")
            key = f"{owner_scope}/{img.get('imageName', '')}:{img.get('imageTag', 'latest')}"
            grouped.setdefault(key, []).append(img)

        ref_max_width: int = _image_ref_column_width(
            console.size.width, is_team_listing=is_team_listing
        )

        table = Table(title=title)
        table.add_column(
            "Image Reference",
            style="cyan",
            no_wrap=True,
            overflow="crop",
            min_width=ref_max_width,
            max_width=ref_max_width,
        )
        table.add_column("Type", justify="center", no_wrap=True)
        if is_team_listing:
            table.add_column("Owner", justify="center")
        # Worst-case label is ``Cancelled / Cancelled`` (21 chars); pin to 21
        # so Rich never wraps the status text across two lines.
        table.add_column("Status", justify="center", no_wrap=True, min_width=21)
        table.add_column("Size", justify="right", no_wrap=True)
        table.add_column("Created", style="dim", no_wrap=True, min_width=16)

        sortable: list[tuple[datetime, list[ImageRow], PartitionMap]] = []
        for _key, artifacts in grouped.items():
            partition = _partition_group(artifacts)
            sortable.append((_group_sort_key(partition), artifacts, partition))
        sortable.sort(key=lambda item: item[0], reverse=True)

        for _ts, artifacts, partition in sortable:
            # Pick a representative row for the Image Reference / Owner
            # columns. Prefer the latest container artifact row so the
            # reference always resolves to something the user can copy-paste,
            # then fall back to the latest VM row, then to any raw artifact.
            container_latest = (partition.get("CONTAINER_IMAGE") or ArtifactPartition()).latest
            vm_latest = (partition.get("VM_SANDBOX") or ArtifactPartition()).latest
            preferred: ImageRow = container_latest or vm_latest or next(iter(artifacts), {})

            image_ref: str = _truncate_ref_left(
                _render_image_reference(preferred, is_team_listing=is_team_listing),
                ref_max_width,
            )
            type_display: str = _render_type_column(partition)
            status_display: str = _render_status_column(partition)
            size_mb: str = _completed_size_mb(partition)
            date_str: str = _display_created(partition)

            row: list[str] = [image_ref, type_display]
            if is_team_listing:
                owner_type = preferred.get(
                    "ownerType", "team" if preferred.get("teamId") else "personal"
                )
                owner_display: str = (
                    "[blue]Team[/blue]" if owner_type == "team" else "[dim]Personal[/dim]"
                )
                row.append(owner_display)
            row.extend([status_display, size_mb, date_str])
            table.add_row(*row)

        console.print()
        console.print(table)
        console.print()
        console.print(f"[dim]Total: {len(grouped)} image(s)[/dim]")
        console.print()

    except UnauthorizedError:
        console.print("[red]Error: Not authenticated. Please run 'prime login' first.[/red]")
        raise typer.Exit(1)
    except APIError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("delete")
def delete_image(
    image_reference: str = typer.Argument(
        ...,
        help="Image reference to delete (e.g., 'myapp:v1.0.0' or 'team-{teamId}/myapp:v1.0.0')",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """
    Delete an image from your registry.

    For team images, you can use the team-prefixed format directly.
    Only the image creator or team admins can delete team images.

    \b
    Examples:
        prime images delete myapp:v1.0.0
        prime images delete myapp:latest --yes
        prime images delete team-abc123/myapp:v1.0.0
    """
    # Store original input for error messages
    original_reference = image_reference

    try:
        # Check for team-prefixed format: team-{teamId}/imagename:tag
        team_id = config.team_id
        if "/" in image_reference:
            namespace, rest = image_reference.split("/", 1)
            if namespace.startswith("team-"):
                # Extract team ID from the reference
                extracted_team_id = namespace[5:]  # Remove "team-" prefix
                if not extracted_team_id:
                    console.print(
                        "[red]Error: Invalid team image reference. "
                        "Expected format: team-{teamId}/imagename:tag[/red]"
                    )
                    raise typer.Exit(1)
                team_id = extracted_team_id
                image_reference = rest
            else:
                # Unrecognized namespace (not team-prefixed)
                console.print(
                    f"[red]Error: Unrecognized image namespace '{namespace}'. "
                    "Use 'imagename:tag' for personal images or "
                    "'team-{{teamId}}/imagename:tag' for team images.[/red]"
                )
                raise typer.Exit(1)

        # Validate image reference has a tag (after team-prefix parsing)
        if ":" not in image_reference:
            console.print(
                "[red]Error: Image reference must include a tag (e.g., myapp:latest)[/red]"
            )
            raise typer.Exit(1)

        image_name, image_tag = image_reference.rsplit(":", 1)

        context = f" (team: {team_id})" if team_id else ""
        if not yes:
            msg = f"Are you sure you want to delete {image_name}:{image_tag}{context}?"
            confirm = typer.confirm(msg)
            if not confirm:
                console.print("[yellow]Cancelled[/yellow]")
                raise typer.Exit(0)

        client = APIClient()

        params = {"teamId": team_id} if team_id else None

        client.request("DELETE", f"/images/{image_name}/{image_tag}", params=params)
        console.print(f"[green]✓[/green] Deleted {image_name}:{image_tag}{context}")

    except UnauthorizedError:
        console.print("[red]Error: Not authenticated. Please run 'prime login' first.[/red]")
        raise typer.Exit(1)
    except APIError as e:
        if "404" in str(e):
            console.print(f"[red]Error: Image {original_reference} not found[/red]")
        elif "403" in str(e):
            console.print(
                "[red]Error: You don't have permission to delete this image. "
                "Only the image creator or team admins can delete team images.[/red]"
            )
        else:
            console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
