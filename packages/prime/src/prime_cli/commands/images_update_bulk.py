"""Bulk logical-image updates from a JSONL manifest (`prime images update-bulk`)."""

import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

import typer
from prime_sandboxes import (
    APIClient,
    APIError,
    ExplicitUpdateImagesRequest,
    ImageClient,
    ImageUpdateItem,
    ImageUpdateResult,
    PersonalImageOwner,
    TeamImageOwner,
    UnauthorizedError,
)
from pydantic import ValidationError
from rich.table import Table

from ..utils import confirm_or_skip, get_console

console = get_console()

UPDATE_BULK_BATCH_SIZE = 100
_MANIFEST_KEYS = {"source", "set"}
FAILURE_TABLE_MAX_ROWS = 20


class BulkUpdateValidationError(Exception):
    """Raised when the manifest fails preflight validation."""

    def __init__(self, problems: List[str]) -> None:
        super().__init__(f"{len(problems)} problem(s)")
        self.problems = problems


def _manifest_line(item: ImageUpdateItem) -> str:
    """Serialize one item back to its manifest JSONL form (API field names)."""
    return json.dumps(item.model_dump(by_alias=True, exclude_none=True), separators=(",", ":"))


def load_update_manifest(manifest_path: Path) -> List[ImageUpdateItem]:
    """Load and fully validate a JSONL manifest of image updates.

    Every line must be a JSON object shaped like one PATCH /images update:
    ``{"source": {...}, "set": {...}}``. All problems are collected before
    raising so the user can fix the manifest in one pass.
    """
    problems: List[str] = []
    items: List[ImageUpdateItem] = []

    try:
        raw_text = manifest_path.read_text()
    except OSError as exc:
        raise BulkUpdateValidationError([f"cannot read manifest: {exc}"]) from exc

    for line_number, raw_line in enumerate(raw_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"line {line_number}: invalid JSON ({exc.msg})")
            continue
        if not isinstance(data, dict):
            problems.append(f"line {line_number}: expected a JSON object")
            continue
        unknown_keys = set(data) - _MANIFEST_KEYS
        if unknown_keys:
            problems.append(
                f"line {line_number}: unknown key(s): {', '.join(sorted(unknown_keys))}"
            )
            continue
        try:
            items.append(ImageUpdateItem.model_validate(data))
        except ValidationError as exc:
            first = exc.errors()[0]
            location = ".".join(str(part) for part in first.get("loc", ()))
            message = first.get("msg", "invalid value")
            problems.append(f"line {line_number}: {location or 'item'}: {message}")

    if not problems and not items:
        problems.append("manifest contains no updates")

    problems.extend(_duplicate_problems(items))

    if problems:
        raise BulkUpdateValidationError(problems)
    return items


def _source_key(item: ImageUpdateItem) -> str:
    source = item.source
    if source.reference is not None:
        return f"ref:{source.reference}"
    owner = source.owner
    if isinstance(owner, TeamImageOwner):
        owner_key = f"team:{owner.team_id}"
    elif isinstance(owner, PersonalImageOwner):
        owner_key = "personal"
    else:
        owner_key = "platform"
    return f"{owner_key}/{source.name}:{source.tag}"


def _destination_key(item: ImageUpdateItem) -> Optional[str]:
    """Best-effort local destination key; None when it cannot move or the
    source is a reference (only the server can resolve those)."""
    patch = item.set
    if patch.name is None and patch.tag is None and patch.owner is None:
        return None
    source = item.source
    if source.reference is not None:
        return None
    owner = patch.owner or source.owner
    if isinstance(owner, TeamImageOwner):
        owner_key = f"team:{owner.team_id}"
    elif isinstance(owner, PersonalImageOwner):
        owner_key = "personal"
    else:
        owner_key = "platform"
    name = patch.name if patch.name is not None else source.name
    tag = patch.tag if patch.tag is not None else source.tag
    return f"{owner_key}/{name}:{tag}"


def _duplicate_problems(items: List[ImageUpdateItem]) -> List[str]:
    problems: List[str] = []
    seen_sources: dict[str, int] = {}
    seen_destinations: dict[str, int] = {}
    for index, item in enumerate(items, start=1):
        source_key = _source_key(item)
        if source_key in seen_sources:
            problems.append(
                f"duplicate source {source_key!r} (updates {seen_sources[source_key]} and {index})"
            )
        else:
            seen_sources[source_key] = index

        destination_key = _destination_key(item)
        if destination_key is not None:
            if destination_key in seen_destinations:
                problems.append(
                    f"duplicate destination {destination_key!r} "
                    f"(updates {seen_destinations[destination_key]} and {index})"
                )
            else:
                seen_destinations[destination_key] = index
    return problems


def _owner_label(owner: Any) -> str:
    if isinstance(owner, TeamImageOwner):
        return f"team {owner.team_id}"
    if isinstance(owner, PersonalImageOwner):
        return "personal"
    return "platform"


def _state_label(state: Any) -> str:
    if state is None:
        return "—"
    return f"{state.name}:{state.tag} [{_owner_label(state.owner)}, {state.visibility.value}]"


def _result_source_label(item: ImageUpdateItem) -> str:
    source = item.source
    if source.reference is not None:
        return source.reference
    return f"{source.name}:{source.tag}"


def _write_failures_manifest(path: Path, failures: List[ImageUpdateItem]) -> None:
    lines = [_manifest_line(item) for item in failures]
    path.write_text("\n".join(lines) + "\n")


def _print_outcomes(
    outcomes: List[Tuple[ImageUpdateItem, ImageUpdateResult]], *, dry_run: bool
) -> None:
    verb = "Would update" if dry_run else "Updated"
    for item, result in outcomes:
        if result.success:
            console.print(
                f"  [green]✓[/green] {verb} {_state_label(result.before)} "
                f"→ {_state_label(result.after)}"
            )
        else:
            message = result.error.message if result.error else "unknown error"
            code = f" ({result.error.code})" if result.error else ""
            console.print(f"  [red]✗[/red] {_result_source_label(item)}: {message}{code}")


def update_bulk(
    manifest: Path = typer.Option(
        ...,
        "--manifest",
        "-m",
        help=(
            "JSONL manifest; each line is one update: "
            '{"source": {...}, "set": {...}} using PATCH /images field names'
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Server-side validation and before/after preview without applying",
    ),
    failures_out: Path = typer.Option(
        Path("update-bulk-failures.jsonl"),
        "--failures-out",
        help="Where to write failed updates as a re-runnable manifest",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the confirmation prompt for owner moves and platform promotions",
    ),
):
    """
    Apply many logical-image updates from a JSONL manifest.

    Every line is validated before anything is sent. Updates are applied in
    batches of 100; each update is independent, so one failure does not stop
    the rest. Failed updates are written to a re-runnable failures manifest.

    \b
    Example manifest lines:
        {"source":{"owner":{"type":"personal"},"name":"old-a","tag":"v1"},"set":{"name":"new-a"}}
        {"source":{"reference":"prime/research/old-b:v2"},"set":{"owner":{"type":"platform"}}}
        {"source":{"owner":{"type":"personal"},"name":"c","tag":"v3"},"set":{"visibility":"PUBLIC"}}
    """
    try:
        items = load_update_manifest(manifest)
    except BulkUpdateValidationError as exc:
        console.print(
            f"[red]Error: cannot start bulk update ({len(exc.problems)} problem(s)):[/red]"
        )
        for problem in exc.problems:
            console.print(f"  [red]- {problem}[/red]")
        raise typer.Exit(1)

    owner_moves = sum(1 for item in items if item.set.owner is not None)
    console.print(
        f"[bold blue]Bulk image update:[/bold blue] {len(items)} update(s)"
        + (f", {owner_moves} owner move(s)" if owner_moves else "")
    )

    if dry_run:
        console.print("[dim]Dry run: validating against the server, no writes.[/dim]")
    elif owner_moves and not confirm_or_skip(
        f"Apply {len(items)} update(s) including {owner_moves} owner move(s)?", yes
    ):
        console.print("Update cancelled")
        return

    client = ImageClient(APIClient())
    outcomes: List[Tuple[ImageUpdateItem, ImageUpdateResult]] = []
    failed_items: List[ImageUpdateItem] = []
    unsent_items: List[ImageUpdateItem] = []
    transport_error: Optional[str] = None

    for offset in range(0, len(items), UPDATE_BULK_BATCH_SIZE):
        batch = items[offset : offset + UPDATE_BULK_BATCH_SIZE]
        try:
            response = client.update_images(
                ExplicitUpdateImagesRequest(dry_run=dry_run, updates=batch)
            )
        except UnauthorizedError:
            console.print("[red]Error: Not authenticated. Please run 'prime login' first.[/red]")
            raise typer.Exit(1)
        except APIError as exc:
            transport_error = str(exc)
            unsent_items.extend(items[offset:])
            break
        for item, result in zip(batch, response.results):
            outcomes.append((item, result))
            if not result.success:
                failed_items.append(item)

    _print_outcomes(outcomes, dry_run=dry_run)

    succeeded_count = sum(1 for _, result in outcomes if result.success)
    console.print()
    console.print(
        f"[bold]{'Would update' if dry_run else 'Updated'}:[/bold] {succeeded_count}  "
        f"[bold]Failed:[/bold] {len(failed_items)}  "
        f"[bold]Not sent:[/bold] {len(unsent_items)}"
    )

    if transport_error is not None:
        console.print(f"[red]Error: {transport_error}[/red]")

    retry_items = failed_items + unsent_items
    if retry_items and not dry_run:
        _write_failures_manifest(failures_out, retry_items)
        console.print(
            f"[yellow]Wrote {len(retry_items)} unapplied update(s) to "
            f"{failures_out}; re-run with: prime images update-bulk "
            f"--manifest {failures_out}[/yellow]"
        )

    if failed_items and len(failed_items) > FAILURE_TABLE_MAX_ROWS:
        table = Table(title="First failures")
        table.add_column("Source", style="cyan")
        table.add_column("Error", style="red")
        for item, result in outcomes:
            if result.success:
                continue
            table.add_row(
                _result_source_label(item),
                result.error.message if result.error else "unknown error",
            )
            if table.row_count >= FAILURE_TABLE_MAX_ROWS:
                break
        console.print(table)

    if retry_items or transport_error is not None:
        raise typer.Exit(1)
