"""Commands for managing Docker images in Prime Intellect registry."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import click
import httpx
import typer
from prime_sandboxes import (
    APIClient,
    APIError,
    BulkImageTransferResponse,
    Config,
    ImageClient,
    ImageVisibility,
    UnauthorizedError,
)
from rich.table import Table

from ..utils import (
    PlainTyper,
    confirm,
    confirm_or_skip,
    get_console,
    json_output_help,
    output_data_as_json,
    validate_output_format,
)
from .images_bulk import (
    PACKAGED_DOCKERFILE_PATH,
    package_build_context,
    push_bulk,
)

app = PlainTyper(help="Manage Docker images in Prime Intellect registry", no_args_is_help=True)
console = get_console()

config = Config()


LIST_IMAGES_JSON_HELP = json_output_help(
    "Raw API response is printed unchanged.",
    ".data[] = {displayRef?, fullImagePath?, imageName, imageTag, status, "
    "artifactType, ownerType, visibility, sizeBytes?, createdAt, pushedAt?}",
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


def _render_visibility(value: Any) -> str:
    try:
        visibility = ImageVisibility(str(value or ImageVisibility.PRIVATE.value).upper())
    except ValueError:
        visibility = ImageVisibility.PRIVATE

    if visibility == ImageVisibility.PUBLIC:
        return "[green]Public[/green]"
    return "[dim]Private[/dim]"


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

    The Prime owner prefix (``prime/{ownerSlug}/``, ``prime/{userId}/``, or
    ``prime/team-{teamId}/``) is **always** preserved so the full string is a
    valid reference that can be pasted directly into downstream commands
    (e.g. ``prime sandbox create``), which require a fully-qualified
    ``prime/<owner>/<imageName>:<tag>`` form.

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
        Visibility ~9 chars
        Size     ~10 chars
        Created  ~17 chars ("YYYY-MM-DD HH:MM ")
        Owner    ~10 chars (team listings only)
        borders + padding ~ 3 chars per column

    We subtract these from the terminal width and clamp to a reasonable range
    so the column is neither absurdly wide on large terminals nor unusable on
    tiny ones.
    """
    reserved = 14 + 20 + 9 + 10 + 17 + (10 if is_team_listing else 0)
    num_cols = 6 + (1 if is_team_listing else 0)
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
    image_reference: Optional[str] = typer.Argument(
        None, help="Image reference (e.g., 'myapp:v1.0.0' or 'myapp:latest')"
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
    public: bool = typer.Option(
        False,
        "--public",
        help="Make the image public when the build completes",
    ),
    private: bool = typer.Option(
        False,
        "--private",
        help="Make the image private when the build completes",
    ),
    source_image: Optional[str] = typer.Option(
        None,
        "--source-image",
        help="Copy an existing public image into Prime instead of uploading a build context",
    ),
    platform_image: bool = typer.Option(
        False,
        "--platform-image",
        help="Build an org-less platform image (admins only)",
    ),
):
    """
    Build and push a Docker image to Prime Intellect registry.

    New image tags are private by default. Re-pushing an existing tag keeps
    its current visibility unless --public or --private is provided.

    \b
    Examples:
        prime images push myapp:v1.0.0
        prime images push myapp:latest --context ./app --dockerfile ../docker/Dockerfile.prod
        prime images push myapp:v1 --platform linux/arm64
        prime images push myapp:v1 --public
        prime images push --source-image ubuntu:22.04
        prime images push myubuntu:22.04 --source-image ubuntu:22.04
    """
    try:
        if public and private:
            console.print("[red]Error: --public and --private cannot be used together[/red]")
            raise typer.Exit(1)

        is_transfer = source_image is not None
        if platform_image and private:
            console.print("[red]Error: Platform images must be public[/red]")
            raise typer.Exit(1)
        if platform_image and config.team_id:
            console.print("[red]Error: Platform images cannot be pushed in a team context[/red]")
            raise typer.Exit(1)
        if not is_transfer and image_reference is None:
            console.print(
                "[red]Error: Image reference is required unless --source-image is used[/red]"
            )
            raise typer.Exit(1)

        transfer_sources = [
            source.strip() for source in (source_image or "").split(",") if source.strip()
        ]
        if is_transfer and not transfer_sources:
            console.print(
                "[red]Error: --source-image must include at least one image reference[/red]"
            )
            raise typer.Exit(1)
        if is_transfer and image_reference is not None and len(transfer_sources) > 1:
            console.print(
                "[red]Error: Destination image reference can only be provided for "
                "single-image transfers[/red]"
            )
            raise typer.Exit(1)

        # Parse image reference
        if image_reference is None:
            image_name = None
            image_tag = None
        elif ":" in image_reference:
            image_name, image_tag = image_reference.rsplit(":", 1)
        else:
            image_name = image_reference
            image_tag = "latest"

        # Validate image name doesn't contain slashes
        if image_name is not None and "/" in image_name and not platform_image:
            console.print(
                "[red]Error: Image name cannot contain '/'. "
                "Use simple names like 'myapp:v1.0.0'.[/red]"
            )
            raise typer.Exit(1)

        if is_transfer:
            source_display = ", ".join(transfer_sources)
            destination_display = (
                f"{image_name}:{image_tag}" if image_name and image_tag else "derived"
            )
            if platform_image:
                console.print("[bold blue]Transferring platform image into Prime:[/bold blue]")
            else:
                console.print("[bold blue]Transferring image into Prime:[/bold blue]")
            console.print(f"[bold]Source:[/bold] {source_display}")
            console.print(f"[bold]Destination:[/bold] {destination_display}")
            if platform_image:
                console.print("[bold]Owner:[/bold] Platform")
            if config.team_id:
                console.print(f"[dim]Team: {config.team_id}[/dim]")
            console.print()

            client = ImageClient(APIClient())
            visibility = None
            if public:
                visibility = ImageVisibility.PUBLIC
            elif private:
                visibility = ImageVisibility.PRIVATE
            if platform_image:
                visibility = ImageVisibility.PUBLIC

            try:
                response = client.transfer_image(
                    ",".join(transfer_sources),
                    image_name=image_name,
                    image_tag=image_tag,
                    platform=platform,
                    team_id=config.team_id or None,
                    visibility=visibility,
                    owner_scope="platform" if platform_image else None,
                )
            except UnauthorizedError:
                console.print(
                    "[red]Error: Not authenticated. Please run 'prime login' first.[/red]"
                )
                raise typer.Exit(1)
            except APIError as e:
                console.print(f"[red]Error: Failed to initiate transfer: {e}[/red]")
                raise typer.Exit(1)

            if isinstance(response, BulkImageTransferResponse):
                successful_results = [result for result in response.results if result.build_id]
                build_ids = [result.build_id for result in successful_results if result.build_id]
                failed_results = response.failed
                image_path = (
                    successful_results[0].full_image_path if len(successful_results) == 1 else None
                )
            else:
                build_ids = response.build_ids or [response.build_id]
                failed_results = []
                image_path = response.full_image_path

            if not build_ids:
                console.print("[red]Error: Failed to initiate image transfer[/red]")
                for result in failed_results:
                    console.print(f"[red]- {result.source_image}: {result.error}[/red]")
                raise typer.Exit(1)

            console.print("[green]✓[/green] Transfer queued")
            console.print()
            if len(build_ids) == 1:
                console.print("[bold green]Image transfer queued successfully![/bold green]")
                console.print()
                console.print(f"[bold]Build ID:[/bold] {build_ids[0]}")
                console.print(f"[bold]Image:[/bold] {image_path}")
            else:
                console.print("[bold green]Image transfers queued successfully![/bold green]")
                console.print()
                console.print(f"[bold]Builds:[/bold] {len(build_ids)}")
                console.print(f"[bold]Build IDs:[/bold] {', '.join(build_ids)}")
            if failed_results:
                console.print()
                console.print(
                    f"[yellow]Warning: {len(failed_results)} image transfer(s) failed:[/yellow]"
                )
                for result in failed_results:
                    console.print(f"[yellow]- {result.source_image}: {result.error}[/yellow]")
            if platform_image:
                console.print(f"[bold]Visibility:[/bold] {ImageVisibility.PUBLIC.value}")
            elif public or private:
                requested_visibility = ImageVisibility.PUBLIC if public else ImageVisibility.PRIVATE
                console.print(f"[bold]Visibility:[/bold] {requested_visibility.value}")
            else:
                console.print(
                    "[bold]Visibility:[/bold] PRIVATE for new images "
                    "(existing tags keep their current visibility)"
                )
            console.print()
            console.print("[cyan]Your image transfer is running.[/cyan]")
            console.print()
            console.print("[bold]Check transfer status:[/bold]")
            console.print("  prime images list")
            console.print()
            if failed_results:
                raise typer.Exit(1)
            return

        if platform_image:
            console.print(
                f"[bold blue]Building and pushing platform image:[/bold blue] "
                f"{image_name}:{image_tag}"
            )
        else:
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
        tar_path = package_build_context(context_path, dockerfile_path)

        try:
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
                if platform_image:
                    build_payload["owner_scope"] = "platform"
                    build_payload["visibility"] = ImageVisibility.PUBLIC.value
                elif public:
                    build_payload["visibility"] = ImageVisibility.PUBLIC.value
                elif private:
                    build_payload["visibility"] = ImageVisibility.PRIVATE.value

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
            if platform_image:
                console.print("[bold]Owner:[/bold] Platform")
                console.print(f"[bold]Visibility:[/bold] {ImageVisibility.PUBLIC.value}")
            elif public or private:
                requested_visibility = ImageVisibility.PUBLIC if public else ImageVisibility.PRIVATE
                console.print(f"[bold]Visibility:[/bold] {requested_visibility.value}")
            else:
                console.print(
                    "[bold]Visibility:[/bold] PRIVATE for new images "
                    "(existing tags keep their current visibility)"
                )
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


# Bulk push (JSONL manifest / Harbor task dirs) lives in images_bulk.py.
app.command("push-bulk")(push_bulk)


@app.command("list", epilog=LIST_IMAGES_JSON_HELP)
def list_images(
    output: str = typer.Option("table", "--output", "-o", help="Output format (table or json)"),
    search: Optional[str] = typer.Option(
        None,
        "--search",
        "-q",
        help="Case-insensitive substring match on image name, tag, or reference",
    ),
    all_images: bool = typer.Option(
        False, "--all", "-a", help="[Deprecated] Show all accessible images (personal + team)"
    ),
    platform_image: bool = typer.Option(
        False,
        "--platform-image",
        help="List org-less platform images (admins only)",
    ),
    page: int = typer.Option(1, "--page", "-p", help="Page number"),
    num: int = typer.Option(50, "--num", "-n", help="Items per page (max 250)"),
):
    """
    List all images you've pushed to Prime Intellect registry.

    Shows personal images by default, or team images when a team is configured.

    The --search filter is applied server-side across all your images (not just
    the current page), and pagination reflects the filtered results.

    \b
    Examples:
        prime images list
        prime images list --search myapp
        prime images list -q nvidia
        prime images list --num 100
        prime images list --page 2
        prime images list --output json
        prime images list --platform-image
    """
    validate_output_format(output, console)

    if num < 1 or page < 1:
        console.print("[red]Error:[/red] --num and --page must be at least 1")
        raise typer.Exit(1)
    if num > 250:
        console.print("[red]Error:[/red] --num cannot exceed 250")
        raise typer.Exit(1)
    if platform_image and config.team_id:
        console.print("[red]Error: Platform images cannot be listed in a team context[/red]")
        raise typer.Exit(1)

    if all_images and output != "json":
        console.print(
            "[yellow]Warning: --all flag is deprecated and will be removed in a future release. "
            "Images are now scoped to your current context (personal or team).[/yellow]"
        )
        console.print()
    try:
        client = APIClient()

        offset = (page - 1) * num

        # Build query params
        params: dict[str, str] = {"limit": str(num), "offset": str(offset)}
        if platform_image:
            params["ownerScope"] = "platform"
        elif config.team_id:
            params["teamId"] = config.team_id
        if search:
            params["search"] = search

        response = client.request("GET", "/images", params=params)
        images: list[ImageRow] = response.get("data", [])
        has_total_count: bool = "totalCount" in response
        total_count: int = int(response.get("totalCount", offset + len(images)))

        if output == "json":
            output_data_as_json(response, console)
            return

        push_hint: str = (
            "Push an image with: [bold]prime images push <name>:<tag> --platform-image[/bold]"
            if platform_image
            else "Push an image with: [bold]prime images push <name>:<tag>[/bold]"
        )
        if not images:
            if has_total_count and total_count == 0:
                if search:
                    console.print(f"[yellow]No images match '{search}'.[/yellow]")
                    console.print(
                        "Try a different search term or run without [bold]--search[/bold]."
                    )
                else:
                    console.print("[yellow]No images or builds found.[/yellow]")
                    console.print(push_hint)
            elif has_total_count:
                console.print(
                    f"[yellow]No images on page {page}. Total: {total_count} image(s).[/yellow]"
                )
                console.print("Try [bold]--page 1[/bold] to start from the beginning.")
            elif page > 1:
                console.print(f"[yellow]No images on page {page}.[/yellow]")
                console.print("Try [bold]--page 1[/bold] to start from the beginning.")
            elif search:
                console.print(f"[yellow]No images match '{search}'.[/yellow]")
                console.print("Try a different search term or run without [bold]--search[/bold].")
            else:
                console.print("[yellow]No images or builds found.[/yellow]")
                console.print(push_hint)
            return

        # Table output
        # Platform listings are never team-scoped (rejected above).
        is_team_listing: bool = bool(config.team_id)
        title: str
        if platform_image:
            title = "Platform Docker Images"
        elif is_team_listing:
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
        table.add_column("Visibility", justify="center", no_wrap=True)
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
            visibility_display: str = _render_visibility(preferred.get("visibility"))
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
            row.extend([status_display, visibility_display, size_mb, date_str])
            table.add_row(*row)

        console.print()
        console.print(table)
        console.print()
        shown_groups = len(grouped)
        if has_total_count:
            has_next = offset + shown_groups < total_count
        else:
            has_next = shown_groups >= num
        if has_next or page > 1:
            start = offset + 1
            end = offset + shown_groups
            if has_total_count:
                console.print(
                    f"[dim]Page {page} • showing {start}-{end} of {total_count} image(s)[/dim]"
                )
            else:
                console.print(f"[dim]Page {page} • showing {start}-{end}[/dim]")
            if has_next:
                console.print(f"[dim]Use --page {page + 1} to see more.[/dim]")
        elif has_total_count:
            console.print(f"[dim]Total: {total_count} image(s)[/dim]")
        else:
            console.print(f"[dim]Total: {shown_groups} image(s)[/dim]")
        console.print()

    except UnauthorizedError:
        console.print("[red]Error: Not authenticated. Please run 'prime login' first.[/red]")
        raise typer.Exit(1)
    except APIError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


def _looks_like_registry_host(value: str) -> bool:
    # Docker treats localhost as a registry host even without a dot or port.
    return "." in value or ":" in value or value == "localhost"


def _parse_mutable_image_reference(image_reference: str) -> tuple[str, str, Optional[str]]:
    """Parse refs accepted by mutating image commands.

    Returns ``(image_name, image_tag, team_id)``. ``name:tag`` uses the current
    personal/team context. Owner-prefixed refs are passed through for
    server-side resolution because only the backend can distinguish user slugs
    from team slugs. New slug refs must use ``prime/<owner>/name:tag``; legacy
    ``<userId>/<imageName>:tag`` and ``team-<teamId>/<imageName>:tag`` refs are still
    accepted.
    """
    team_id: Optional[str] = config.team_id
    if "/" in image_reference:
        parts = image_reference.split("/")
        if parts[0] == "prime":
            if len(parts) < 3:
                console.print(
                    "[red]Error: Owner-prefixed image references must use "
                    "prime/<owner>/<imageName>:tag or legacy "
                    "<userId>/<imageName>:tag[/red]"
                )
                raise typer.Exit(1)
            namespace = parts[1]
            repo_and_tag = "/".join(parts[2:])
        elif len(parts) >= 2 and not _looks_like_registry_host(parts[0]):
            namespace = parts[0]
            repo_and_tag = "/".join(parts[1:])
        else:
            console.print(
                "[red]Error: Owner-prefixed image references must use "
                "prime/<owner>/<imageName>:tag or legacy "
                "<userId>/<imageName>:tag[/red]"
            )
            raise typer.Exit(1)
        if not namespace or not repo_and_tag:
            console.print(
                "[red]Error: Owner-prefixed image references must use "
                "prime/<owner>/<imageName>:tag or legacy "
                "<userId>/<imageName>:tag[/red]"
            )
            raise typer.Exit(1)
        if namespace == "team-":
            console.print(
                "[red]Error: Invalid team image reference. "
                "Expected format: prime/team-{teamId}/imagename:tag or "
                "team-{teamId}/imagename:tag[/red]"
            )
            raise typer.Exit(1)
        team_id = None

    if ":" not in image_reference:
        console.print("[red]Error: Image reference must include a tag (e.g., myapp:latest)[/red]")
        raise typer.Exit(1)

    image_name, image_tag = image_reference.rsplit(":", 1)
    return image_name, image_tag, team_id


def _set_image_visibility(image_reference: str, visibility: ImageVisibility) -> None:
    image_name, image_tag, team_id = _parse_mutable_image_reference(image_reference)
    payload: dict[str, str] = {"visibility": visibility.value}
    if team_id:
        payload["teamId"] = team_id

    client = APIClient()
    client.request(
        "PATCH",
        f"/images/{image_name}/{image_tag}/visibility",
        json=payload,
    )
    context = f" (team: {team_id})" if team_id else ""
    console.print(
        f"[green]✓[/green] Updated {image_name}:{image_tag}{context} to {visibility.value}"
    )


BULK_VISIBILITY_BATCH_SIZE = 100


def _split_image_references(values: Optional[List[str]]) -> list[str]:
    """Flatten space- and comma-separated reference arguments, de-duplicated."""
    refs: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        for ref in value.replace(",", " ").split():
            if ref not in seen:
                refs.append(ref)
                seen.add(ref)
    return refs


def _parse_bulk_visibility_ref(ref: str, command: str) -> dict[str, str]:
    """Parse a plain name:tag reference for the bulk visibility endpoint."""
    if "/" in ref:
        console.print(
            f"[red]Error: '{ref}': owner-prefixed references cannot be combined "
            f"with other images; run 'prime images {command} {ref}' on its own[/red]"
        )
        raise typer.Exit(1)
    name, _, tag = ref.rpartition(":")
    if not name or not tag:
        console.print(
            f"[red]Error: '{ref}': image reference must include a tag (e.g., myapp:latest)[/red]"
        )
        raise typer.Exit(1)
    return {"name": name, "tag": tag}


def _visibility_label(visibility: ImageVisibility) -> str:
    return "public" if visibility == ImageVisibility.PUBLIC else "private"


def _display_bulk_visibility_result(
    succeeded: list[str],
    failed: list[dict[str, Any]],
    visibility: ImageVisibility,
) -> None:
    if not succeeded and not failed:
        console.print("[yellow]No images matched.[/yellow]")
        return
    if succeeded:
        console.print(
            f"\n[bold green]Made {len(succeeded)} image(s) "
            f"{_visibility_label(visibility)}:[/bold green]"
        )
        for ref in succeeded:
            console.print(f"  ✓ {ref}")
    if failed:
        console.print(f"\n[bold red]Failed to update {len(failed)} image(s):[/bold red]")
        for failure in failed:
            ref = failure.get("image", "unknown")
            error = failure.get("error", "unknown error")
            console.print(f"  ✗ {ref}: {error}")


def _bulk_visibility_payload(visibility: ImageVisibility) -> dict[str, Any]:
    payload: dict[str, Any] = {"visibility": visibility.value}
    if config.team_id:
        payload["teamId"] = config.team_id
    return payload


def _bulk_set_image_visibility_refs(refs: list[str], visibility: ImageVisibility) -> None:
    """Update explicit references via the bulk endpoint, batched per request."""
    command = "publish" if visibility == ImageVisibility.PUBLIC else "unpublish"
    images = [_parse_bulk_visibility_ref(ref, command) for ref in refs]
    client = APIClient()
    all_succeeded: list[str] = []
    all_failed: list[dict[str, Any]] = []
    batch_error: Optional[str] = None
    batch_size = BULK_VISIBILITY_BATCH_SIZE
    total_batches = (len(images) + batch_size - 1) // batch_size

    with console.status("[bold blue]Updating image visibility...", spinner="dots"):
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            if total_batches > 1:
                console.print(
                    f"[dim]Processing batch {i // batch_size + 1}/"
                    f"{total_batches} ({len(batch)} images)...[/dim]"
                )
            payload = _bulk_visibility_payload(visibility)
            payload["images"] = batch
            try:
                response = client.request("PATCH", "/images/visibility", json=payload)
            except UnauthorizedError:
                raise
            except APIError as e:
                batch_error = str(e)
                break
            all_succeeded.extend(response.get("succeeded", []))
            all_failed.extend(response.get("failed", []))

    if batch_error is not None:
        console.print(f"[red]Error: {batch_error}[/red]")
        if all_succeeded or all_failed:
            _display_bulk_visibility_result(all_succeeded, all_failed, visibility)
            not_processed = len(images) - len(all_succeeded) - len(all_failed)
            console.print(
                f"\n[yellow]{not_processed} image(s) were not processed; "
                "re-run the command to retry them.[/yellow]"
            )
        raise typer.Exit(1)

    _display_bulk_visibility_result(all_succeeded, all_failed, visibility)
    if all_failed:
        raise typer.Exit(1)


def _preview_bulk_visibility_count(client: APIClient, search: str) -> Optional[int]:
    """Count the logical images matching the search, cheaply, via list(limit=1)."""
    params: dict[str, str] = {"limit": "1", "offset": "0", "search": search}
    if config.team_id:
        params["teamId"] = config.team_id
    try:
        response = client.request("GET", "/images", params=params)
    except UnauthorizedError:
        raise
    except APIError:
        return None
    total = response.get("totalCount")
    return int(total) if total is not None else None


def _bulk_set_image_visibility_search(search: str, visibility: ImageVisibility, yes: bool) -> None:
    """Update every image matching the search filter via the bulk endpoint."""
    if "/" in search:
        console.print(
            f"[red]Error: '--search {search}': search matches image name/tag only; "
            "remove the owner prefix (e.g. use 'myapp' instead of 'prime/alice/myapp')[/red]"
        )
        raise typer.Exit(1)
    client = APIClient()
    total = _preview_bulk_visibility_count(client, search)
    if total == 0:
        console.print(f"[yellow]No images match '{search}'.[/yellow]")
        return

    scope = f"team {config.team_id}" if config.team_id else "your personal images"
    count_phrase = f"{total} image(s)" if total is not None else "all matching images"
    if not confirm_or_skip(
        f"Make {count_phrase} matching '{search}' in {scope} {_visibility_label(visibility)}?",
        yes,
    ):
        console.print("Update cancelled")
        return

    payload = _bulk_visibility_payload(visibility)
    payload["search"] = search
    with console.status("[bold blue]Updating image visibility...", spinner="dots"):
        response = client.request("PATCH", "/images/visibility", json=payload)

    failed = response.get("failed", [])
    _display_bulk_visibility_result(response.get("succeeded", []), failed, visibility)
    if failed:
        raise typer.Exit(1)


def _run_visibility_command(
    image_references: Optional[List[str]],
    search: Optional[str],
    yes: bool,
    visibility: ImageVisibility,
) -> None:
    refs = _split_image_references(image_references)
    if refs and search:
        console.print("[red]Error: Provide image references or --search, not both[/red]")
        raise typer.Exit(1)
    if not refs and not search:
        console.print("[red]Error: Provide at least one image reference or --search[/red]")
        raise typer.Exit(1)
    try:
        if search:
            _bulk_set_image_visibility_search(search, visibility, yes)
        elif len(refs) == 1:
            _set_image_visibility(refs[0], visibility)
        else:
            _bulk_set_image_visibility_refs(refs, visibility)
    except UnauthorizedError:
        console.print("[red]Error: Not authenticated. Please run 'prime login' first.[/red]")
        raise typer.Exit(1)
    except APIError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("publish", no_args_is_help=True)
def publish_image(
    image_references: Optional[List[str]] = typer.Argument(
        None,
        help=(
            "Image reference(s) to make public, space or comma-separated. "
            "A single reference may be owner-prefixed "
            "('myapp:v1.0.0', 'prime/<ownerSlug>/myapp:v1.0.0', or "
            "'prime/team-{teamId}/myapp:v1.0.0'; legacy "
            "'<currentUserId>/myapp:v1.0.0' and 'team-{teamId}/myapp:v1.0.0' "
            "also work); multiple references must be plain name:tag in the "
            "current personal or team context"
        ),
    ),
    search: Optional[str] = typer.Option(
        None,
        "--search",
        "-q",
        help=(
            "Publish every image matching this case-insensitive substring of the image name or tag"
        ),
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the confirmation prompt for --search"
    ),
):
    """
    Make one or more images public so other Prime users can run them.

    Pass explicit references, or --search to publish every image matching a
    substring (with a count preview and confirmation). In a team context a
    --search publish only updates images you may change (your own images, or
    all team images if you are a team admin).

    \b
    Examples:
        prime images publish myapp:v1.0.0
        prime images publish myapp:v1.0.0 other:v2.0.0 third:latest
        prime images publish --search swe- --yes
        prime images publish prime/alice/myapp:v1.0.0
        prime images publish prime/team-abc123/myapp:v1.0.0
    """
    _run_visibility_command(image_references, search, yes, ImageVisibility.PUBLIC)


@app.command("unpublish", no_args_is_help=True)
def unpublish_image(
    image_references: Optional[List[str]] = typer.Argument(
        None,
        help=(
            "Image reference(s) to make private, space or comma-separated. "
            "A single reference may be owner-prefixed "
            "('myapp:v1.0.0', 'prime/<ownerSlug>/myapp:v1.0.0', or "
            "'prime/team-{teamId}/myapp:v1.0.0'; legacy "
            "'<currentUserId>/myapp:v1.0.0' and 'team-{teamId}/myapp:v1.0.0' "
            "also work); multiple references must be plain name:tag in the "
            "current personal or team context"
        ),
    ),
    search: Optional[str] = typer.Option(
        None,
        "--search",
        "-q",
        help=(
            "Unpublish every image matching this case-insensitive substring "
            "of the image name or tag"
        ),
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the confirmation prompt for --search"
    ),
):
    """
    Make one or more public images private again.

    Pass explicit references, or --search to unpublish every image matching a
    substring (with a count preview and confirmation). In a team context a
    --search unpublish only updates images you may change (your own images, or
    all team images if you are a team admin).

    \b
    Examples:
        prime images unpublish myapp:v1.0.0
        prime images unpublish myapp:v1.0.0 other:v2.0.0 third:latest
        prime images unpublish --search swe- --yes
        prime images unpublish prime/alice/myapp:v1.0.0
        prime images unpublish prime/team-abc123/myapp:v1.0.0
    """
    _run_visibility_command(image_references, search, yes, ImageVisibility.PRIVATE)


@app.command("delete")
def delete_image(
    image_reference: str = typer.Argument(
        ...,
        help=(
            "Image reference to delete "
            "(e.g., 'myapp:v1.0.0', 'prime/<ownerSlug>/myapp:v1.0.0', "
            "'prime/<currentUserId>/myapp:v1.0.0', or "
            "'prime/team-{teamId}/myapp:v1.0.0'; legacy "
            "'<currentUserId>/myapp:v1.0.0' and "
            "'team-{teamId}/myapp:v1.0.0' also work)"
        ),
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """
    Delete an image from your registry.

    For team images, you can use the prime/team-{teamId}/... format directly.
    Only the image creator or team admins can delete team images.

    \b
    Examples:
        prime images delete myapp:v1.0.0
        prime images delete myapp:latest --yes
        prime images delete prime/alice/myapp:v1.0.0
        prime images delete prime/cmk123/myapp:v1.0.0
        prime images delete prime/team-abc123/myapp:v1.0.0
        prime images delete cmk123/myapp:v1.0.0
        prime images delete team-abc123/myapp:v1.0.0
    """
    # Store original input for error messages
    original_reference = image_reference

    try:
        image_name, image_tag, team_id = _parse_mutable_image_reference(image_reference)

        context = f" (team: {team_id})" if team_id else ""
        if not yes:
            msg = f"Are you sure you want to delete {image_name}:{image_tag}{context}?"
            if not confirm(msg):
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
