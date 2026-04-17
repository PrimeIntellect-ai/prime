"""Tests for `prime images list` rendering helpers and end-to-end output."""

from __future__ import annotations

from typing import Any

from prime_cli.commands import images as images_cmd
from prime_cli.commands.images import (
    ArtifactPartition,
    ImageRow,
    _completed_size_mb,
    _display_created,
    _group_sort_key,
    _image_ref_column_width,
    _partition_group,
    _render_image_reference,
    _render_status_column,
    _render_status_slot,
    _render_type_column,
    _truncate_ref_left,
)
from prime_cli.main import app
from typer.testing import CliRunner

runner: CliRunner = CliRunner()

TEST_ENV: dict[str, str] = {
    "COLUMNS": "200",
    "LINES": "50",
    "NO_COLOR": "1",
    "PRIME_DISABLE_VERSION_CHECK": "1",
}


USER_ID: str = "cmkrcib4x00004kjyxq48nltd"
TEAM_ID: str = "team-abc123"


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _art(
    artifact_type: str,
    status: str,
    *,
    image_name: str = "nvidia-basic-dev",
    image_tag: str = "latest",
    pushed_at: str | None = None,
    completed_at: str | None = None,
    started_at: str | None = None,
    created_at: str = "2026-04-01T10:00:00",
    size_bytes: int | None = None,
    team_id: str | None = None,
    owner_type: str | None = None,
    display_ref: str | None = None,
) -> ImageRow:
    ref: str = (
        display_ref
        if display_ref is not None
        else (f"{('team-' + team_id) if team_id else USER_ID}/{image_name}:{image_tag}")
    )
    row: dict[str, Any] = {
        "id": f"id-{artifact_type}-{status}-{created_at}",
        "artifactType": artifact_type,
        "imageName": image_name,
        "imageTag": image_tag,
        "status": status,
        "fullImagePath": f"{USER_ID}/{image_name}:{image_tag}",
        "errorMessage": None,
        "sizeBytes": size_bytes,
        "createdAt": created_at,
        "startedAt": started_at,
        "completedAt": completed_at,
        "pushedAt": pushed_at,
        "teamId": team_id,
        "ownerType": owner_type or ("team" if team_id else "personal"),
        "displayRef": ref,
    }
    return row


# ---------------------------------------------------------------------------
# _partition_group — only the newest row per artifact type survives
# ---------------------------------------------------------------------------


def test_partition_keeps_newest_completed_per_type():
    arts = [
        _art("CONTAINER_IMAGE", "COMPLETED", pushed_at="2026-04-16T22:24:07"),
        _art("VM_SANDBOX", "COMPLETED", pushed_at="2026-04-16T22:24:07"),
    ]
    part = _partition_group(arts)
    assert part["CONTAINER_IMAGE"].latest is not None
    assert part["CONTAINER_IMAGE"].latest["status"] == "COMPLETED"
    assert part["VM_SANDBOX"].latest is not None
    assert part["VM_SANDBOX"].latest["status"] == "COMPLETED"


def test_partition_newer_completed_beats_older_failed():
    arts = [
        _art("CONTAINER_IMAGE", "FAILED", completed_at="2026-04-16T21:00:00"),
        _art("CONTAINER_IMAGE", "COMPLETED", pushed_at="2026-04-16T22:24:07"),
        _art("CONTAINER_IMAGE", "FAILED", completed_at="2026-04-16T20:55:00"),
    ]
    part = _partition_group(arts)
    assert part["CONTAINER_IMAGE"].latest["status"] == "COMPLETED"


def test_partition_fresh_completed_wins_over_stale_building_zombie():
    # The real-world case observed in production: a stale BUILDING row from
    # days ago (never reaped) alongside a fresh COMPLETED push. The fresh
    # COMPLETED row has a later pushedAt, so it wins — no misleading
    # "(rebuilding)" signal.
    arts = [
        _art(
            "VM_SANDBOX",
            "BUILDING",
            started_at="2026-04-09 20:52:01",
            created_at="2026-04-09T20:52:00",
        ),
        _art(
            "VM_SANDBOX",
            "COMPLETED",
            pushed_at="2026-04-17T18:21:28",
            completed_at="2026-04-17T18:21:28",
            created_at="2026-04-17T18:19:01",
        ),
    ]
    part = _partition_group(arts)
    assert part["VM_SANDBOX"].latest["status"] == "COMPLETED"


def test_partition_active_build_wins_over_older_completed():
    # If a genuinely-new build started *after* the last completed push, it's
    # the truth and should be reflected in the status.
    arts = [
        _art("CONTAINER_IMAGE", "COMPLETED", pushed_at="2026-04-15T20:57:52"),
        _art(
            "CONTAINER_IMAGE",
            "BUILDING",
            started_at="2026-04-17T10:00:05",
            created_at="2026-04-17T10:00:00",
        ),
    ]
    part = _partition_group(arts)
    assert part["CONTAINER_IMAGE"].latest["status"] == "BUILDING"


def test_partition_surfaces_failure_when_only_failed_rows_exist():
    arts = [
        _art("CONTAINER_IMAGE", "FAILED", completed_at="2026-04-16T21:00:00"),
        _art("VM_SANDBOX", "FAILED", completed_at="2026-04-16T21:00:00"),
    ]
    part = _partition_group(arts)
    assert part["CONTAINER_IMAGE"].latest["status"] == "FAILED"
    assert part["VM_SANDBOX"].latest["status"] == "FAILED"


def test_partition_keeps_unknown_status_row_as_latest():
    arts = [_art("CONTAINER_IMAGE", "QUEUED", created_at="2026-04-17T12:00:00")]
    part = _partition_group(arts)
    assert part["CONTAINER_IMAGE"].latest is not None
    assert part["CONTAINER_IMAGE"].latest["status"] == "QUEUED"


def test_partition_coerces_null_artifact_type_to_container_image():
    arts = [
        {
            "id": "x",
            "artifactType": None,
            "imageName": "myimage",
            "imageTag": "latest",
            "status": "COMPLETED",
            "pushedAt": "2026-04-17T12:00:00",
            "createdAt": "2026-04-17T12:00:00",
            "displayRef": f"{USER_ID}/myimage:latest",
        }
    ]
    part = _partition_group(arts)
    assert "CONTAINER_IMAGE" in part
    assert None not in part
    assert part["CONTAINER_IMAGE"].latest is not None


def test_partition_coerces_non_string_artifact_type():
    arts = [
        {
            "id": "x",
            "artifactType": 42,  # defensive: server could send a number
            "imageName": "myimage",
            "imageTag": "latest",
            "status": "COMPLETED",
            "pushedAt": "2026-04-17T12:00:00",
            "createdAt": "2026-04-17T12:00:00",
            "displayRef": f"{USER_ID}/myimage:latest",
        }
    ]
    part = _partition_group(arts)
    assert "CONTAINER_IMAGE" in part
    # ``sorted(partition)`` must not raise on mixed key types.
    _render_status_column(part)


# ---------------------------------------------------------------------------
# _render_status_slot — renders the raw status of the latest row
# ---------------------------------------------------------------------------


def test_render_status_slot_ready():
    part = ArtifactPartition(latest={"status": "COMPLETED"})
    assert _render_status_slot(part) == "[green]Ready[/green]"


def test_render_status_slot_building():
    part = ArtifactPartition(latest={"status": "BUILDING"})
    assert _render_status_slot(part) == "[yellow]Building[/yellow]"


def test_render_status_slot_uploading():
    part = ArtifactPartition(latest={"status": "UPLOADING"})
    assert _render_status_slot(part) == "[yellow]Uploading[/yellow]"


def test_render_status_slot_pending():
    part = ArtifactPartition(latest={"status": "PENDING"})
    assert _render_status_slot(part) == "[blue]Pending[/blue]"


def test_render_status_slot_failed():
    part = ArtifactPartition(latest={"status": "FAILED"})
    assert _render_status_slot(part) == "[red]Failed[/red]"


def test_render_status_slot_cancelled():
    part = ArtifactPartition(latest={"status": "CANCELLED"})
    assert _render_status_slot(part) == "[dim]Cancelled[/dim]"


def test_render_status_slot_none_part_returns_dash():
    assert _render_status_slot(None) == "[dim]—[/dim]"


def test_render_status_slot_empty_part_returns_dash():
    assert _render_status_slot(ArtifactPartition()) == "[dim]—[/dim]"


def test_render_status_slot_unknown_status_renders_dim_title_case():
    # A status outside the recognised enum set (e.g. the backend adds
    # ``QUEUED`` later) is surfaced as a dim title-cased label.
    part = ArtifactPartition(latest={"status": "QUEUED"})
    assert _render_status_slot(part) == "[dim]Queued[/dim]"


# ---------------------------------------------------------------------------
# _render_type_column
# ---------------------------------------------------------------------------


def test_render_type_column_container_and_vm():
    arts = [
        _art("CONTAINER_IMAGE", "COMPLETED", pushed_at="2026-04-16T22:24:07"),
        _art("VM_SANDBOX", "COMPLETED", pushed_at="2026-04-16T22:24:07"),
    ]
    text = _render_type_column(_partition_group(arts))
    assert "Container" in text and "VM" in text
    assert " / " in text
    assert text.index("Container") < text.index("VM")


def test_render_type_column_container_only_legacy():
    arts = [_art("CONTAINER_IMAGE", "COMPLETED", pushed_at="2026-04-16T22:24:07")]
    text = _render_type_column(_partition_group(arts))
    assert "Container" in text
    assert "VM" not in text
    assert " / " not in text


# ---------------------------------------------------------------------------
# _render_status_column (positional slots)
# ---------------------------------------------------------------------------


def test_render_status_column_healthy_slots():
    arts = [
        _art("CONTAINER_IMAGE", "COMPLETED", pushed_at="2026-04-16T22:24:07"),
        _art("VM_SANDBOX", "COMPLETED", pushed_at="2026-04-16T22:24:07"),
    ]
    text = _render_status_column(_partition_group(arts))
    assert text == "[green]Ready[/green] / [green]Ready[/green]"


def test_render_status_column_partial_failure_aligned():
    arts = [
        _art("CONTAINER_IMAGE", "COMPLETED", pushed_at="2026-04-16T22:24:07"),
        _art("VM_SANDBOX", "FAILED", completed_at="2026-04-16T22:00:00"),
    ]
    text = _render_status_column(_partition_group(arts))
    assert text == "[green]Ready[/green] / [red]Failed[/red]"


def test_render_status_column_container_only_legacy():
    arts = [_art("CONTAINER_IMAGE", "COMPLETED", pushed_at="2026-04-16T22:24:07")]
    text = _render_status_column(_partition_group(arts))
    assert text == "[green]Ready[/green]"
    assert " / " not in text


def test_render_status_column_stale_zombie_hidden_by_fresh_completed():
    # Mirrors the real nvidia-basic-dev:latest payload: a week-old stuck
    # BUILDING VM row + a fresh successful build. The fresh COMPLETED row has
    # a newer pushedAt, so only "Ready" is shown — no false "rebuilding".
    arts = [
        _art("CONTAINER_IMAGE", "COMPLETED", pushed_at="2026-04-17T18:21:28"),
        _art("VM_SANDBOX", "COMPLETED", pushed_at="2026-04-17T18:21:28"),
        # Old stuck BUILDING row that should be ignored because it's older.
        _art(
            "VM_SANDBOX",
            "BUILDING",
            started_at="2026-04-09T20:52:01",
            created_at="2026-04-09T20:52:00",
        ),
        # Recent failed attempts before the successful build.
        _art("CONTAINER_IMAGE", "FAILED", completed_at="2026-04-16T21:00:00"),
        _art("VM_SANDBOX", "FAILED", completed_at="2026-04-16T21:00:00"),
    ]
    text = _render_status_column(_partition_group(arts))
    assert text == "[green]Ready[/green] / [green]Ready[/green]"
    assert "rebuilding" not in text
    assert "Failed" not in text
    assert "Building" not in text


def test_render_status_column_active_build_on_top_of_older_completed():
    # Container has been rebuilt recently (BUILDING started *after* the last
    # completed push); status should reflect the new build rather than the
    # stale Ready.
    arts = [
        _art("CONTAINER_IMAGE", "COMPLETED", pushed_at="2026-04-15T20:57:52"),
        _art(
            "CONTAINER_IMAGE",
            "BUILDING",
            started_at="2026-04-17T10:00:05",
            created_at="2026-04-17T10:00:00",
        ),
        _art("VM_SANDBOX", "COMPLETED", pushed_at="2026-04-15T20:57:52"),
        _art(
            "VM_SANDBOX",
            "BUILDING",
            started_at="2026-04-17T10:00:05",
            created_at="2026-04-17T10:00:00",
        ),
    ]
    text = _render_status_column(_partition_group(arts))
    assert text == "[yellow]Building[/yellow] / [yellow]Building[/yellow]"


def test_render_status_column_surfaces_unknown_status_alongside_known():
    arts = [
        _art("CONTAINER_IMAGE", "COMPLETED", pushed_at="2026-04-16T22:24:07"),
        _art("VM_SANDBOX", "QUEUED", created_at="2026-04-16T22:00:00"),
    ]
    text = _render_status_column(_partition_group(arts))
    assert text == "[green]Ready[/green] / [dim]Queued[/dim]"


# ---------------------------------------------------------------------------
# _render_image_reference
# ---------------------------------------------------------------------------


def test_render_image_reference_always_shows_user_prefix_for_personal():
    img = _art("CONTAINER_IMAGE", "COMPLETED", image_name="myapp", image_tag="v1")
    assert _render_image_reference(img, is_team_listing=False) == f"{USER_ID}/myapp:v1"


def test_render_image_reference_keeps_team_prefix_for_team_listing():
    img = _art(
        "CONTAINER_IMAGE",
        "COMPLETED",
        team_id=TEAM_ID,
        image_name="myapp",
        image_tag="v1",
    )
    assert _render_image_reference(img, is_team_listing=True) == f"team-{TEAM_ID}/myapp:v1"


def test_render_image_reference_falls_back_without_display_ref():
    img = {"imageName": "legacy", "imageTag": "v2"}
    assert _render_image_reference(img, is_team_listing=False) == "legacy:v2"


# ---------------------------------------------------------------------------
# _truncate_ref_left
# ---------------------------------------------------------------------------


def test_truncate_ref_left_returns_unchanged_when_within_width():
    assert _truncate_ref_left("short/name:tag", 40) == "short/name:tag"


def test_truncate_ref_left_clips_head_and_preserves_name_tag():
    ref = f"{USER_ID}/nvidia-basic-dev:latest"
    out = _truncate_ref_left(ref, 30)
    assert out.endswith("nvidia-basic-dev:latest")
    assert out.startswith("…")
    assert len(out) == 30


def test_truncate_ref_left_handles_zero_or_negative_budget():
    ref = f"{USER_ID}/x:y"
    assert _truncate_ref_left(ref, 0) == ref
    assert _truncate_ref_left(ref, -5) == ref


def test_truncate_ref_left_clamps_tiny_budget_to_two():
    out = _truncate_ref_left("abcdefghij", 1)
    assert out == "…j"


# ---------------------------------------------------------------------------
# _image_ref_column_width
# ---------------------------------------------------------------------------


def test_image_ref_column_width_wide_terminal_capped_at_80():
    assert _image_ref_column_width(300, is_team_listing=False) == 80


def test_image_ref_column_width_narrow_terminal_clamped_to_floor():
    assert _image_ref_column_width(60, is_team_listing=False) == 30


def test_image_ref_column_width_scales_with_available_space():
    w_personal = _image_ref_column_width(140, is_team_listing=False)
    w_team = _image_ref_column_width(140, is_team_listing=True)
    assert w_team < w_personal
    assert 20 <= w_team <= 80
    assert 20 <= w_personal <= 80


# ---------------------------------------------------------------------------
# _completed_size_mb
# ---------------------------------------------------------------------------


def test_completed_size_sums_only_latest_completed_rows():
    arts = [
        _art(
            "CONTAINER_IMAGE",
            "COMPLETED",
            pushed_at="2026-04-16T22:00:00",
            size_bytes=1024 * 1024 * 100,
        ),
        _art(
            "VM_SANDBOX", "COMPLETED", pushed_at="2026-04-16T22:00:00", size_bytes=1024 * 1024 * 200
        ),
    ]
    part = _partition_group(arts)
    assert _completed_size_mb(part) == "300.0 MB"


def test_completed_size_ignores_older_completed_if_newer_is_not_completed():
    # Size reflects the *current* picture: if the latest CONTAINER_IMAGE row
    # is BUILDING (in flight), its size is unknown, so its contribution is 0
    # even though an older COMPLETED row has a size.
    arts = [
        _art(
            "CONTAINER_IMAGE",
            "COMPLETED",
            pushed_at="2026-04-15T20:00:00",
            size_bytes=1024 * 1024 * 100,
        ),
        _art(
            "CONTAINER_IMAGE",
            "BUILDING",
            started_at="2026-04-17T10:00:05",
            created_at="2026-04-17T10:00:00",
        ),
    ]
    part = _partition_group(arts)
    assert _completed_size_mb(part) == "[dim]—[/dim]"


def test_completed_size_returns_dash_when_no_completed():
    arts = [
        _art("CONTAINER_IMAGE", "BUILDING", started_at="2026-04-17T09:00:00"),
        _art("VM_SANDBOX", "PENDING", created_at="2026-04-17T08:00:00"),
    ]
    part = _partition_group(arts)
    assert _completed_size_mb(part) == "[dim]—[/dim]"


# ---------------------------------------------------------------------------
# _display_created
# ---------------------------------------------------------------------------


def test_display_created_uses_latest_pushed_at_when_completed_exists():
    arts = [
        _art("CONTAINER_IMAGE", "COMPLETED", pushed_at="2026-04-15T20:57:52"),
        _art("VM_SANDBOX", "COMPLETED", pushed_at="2026-04-16T22:24:07"),
    ]
    part = _partition_group(arts)
    assert _display_created(part) == "2026-04-16 22:24"


def test_display_created_falls_back_to_started_at_for_active_builds():
    arts = [
        _art(
            "CONTAINER_IMAGE",
            "BUILDING",
            started_at="2026-04-17T09:00:00",
            created_at="2026-04-17T08:59:00",
        ),
        _art("VM_SANDBOX", "PENDING", created_at="2026-04-17T08:00:00"),
    ]
    part = _partition_group(arts)
    assert _display_created(part) == "2026-04-17 09:00"


def test_display_created_falls_back_to_completed_at_for_failures():
    arts = [
        _art("CONTAINER_IMAGE", "FAILED", completed_at="2026-04-16T12:00:00"),
        _art("VM_SANDBOX", "FAILED", completed_at="2026-04-16T13:00:00"),
    ]
    part = _partition_group(arts)
    assert _display_created(part) == "2026-04-16 13:00"


def test_display_created_skips_truthy_but_unparseable_pushed_at():
    # Regression for the original bugbot finding: if pushedAt is a malformed
    # but truthy string, ``_latest`` skips it and chooses the next parseable
    # key. The tuple-returning _latest reuses *its* parsed timestamp so the
    # display never silently empties.
    arts = [
        _art(
            "CONTAINER_IMAGE",
            "COMPLETED",
            pushed_at="definitely-not-a-date",
            completed_at="2026-04-16T22:24:07",
        )
    ]
    part = _partition_group(arts)
    assert _display_created(part) == "2026-04-16 22:24"


def test_group_sort_key_matches_display_created():
    arts = [
        _art(
            "CONTAINER_IMAGE",
            "COMPLETED",
            pushed_at="definitely-not-a-date",
            completed_at="2026-04-16T22:24:07",
        )
    ]
    part = _partition_group(arts)
    display = _display_created(part)
    sort_key = _group_sort_key(part)
    assert sort_key.strftime("%Y-%m-%d %H:%M") == display


# ---------------------------------------------------------------------------
# End-to-end: run the CLI with a mocked API client
# ---------------------------------------------------------------------------


def _build_dummy_client(payload: list[ImageRow]) -> type:
    class DummyAPIClient:
        def request(
            self,
            method: str,
            path: str,
            json: Any = None,
            params: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            assert method == "GET"
            assert path == "/images"
            return {"data": payload}

    return DummyAPIClient


class _StubConfig:
    def __init__(self, team_id: str | None = None) -> None:
        self.team_id: str | None = team_id


def _patch_config(monkeypatch: Any, team_id: str | None = None) -> None:
    """Replace the module-level ``config`` with a stub."""
    monkeypatch.setattr(images_cmd, "config", _StubConfig(team_id=team_id))


def test_list_cli_shows_type_and_positional_status(monkeypatch):
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    _patch_config(monkeypatch, team_id=None)

    payload = [
        _art(
            "CONTAINER_IMAGE",
            "COMPLETED",
            pushed_at="2026-04-16T22:24:07",
            size_bytes=100 * 1024 * 1024,
        ),
        _art(
            "VM_SANDBOX",
            "COMPLETED",
            pushed_at="2026-04-16T22:24:07",
            size_bytes=200 * 1024 * 1024,
        ),
        _art("CONTAINER_IMAGE", "FAILED", completed_at="2026-04-16T20:00:00"),
        _art("VM_SANDBOX", "FAILED", completed_at="2026-04-16T20:00:00"),
    ]

    monkeypatch.setattr("prime_cli.commands.images.APIClient", _build_dummy_client(payload))

    result = runner.invoke(app, ["images", "list"], env=TEST_ENV)
    assert result.exit_code == 0, result.output

    assert "Container / VM" in result.output
    assert "Ready / Ready" in result.output
    assert "Container:" not in result.output
    assert "VM:" not in result.output
    # Stale failures are older than the completed rows → hidden.
    assert "Failed" not in result.output
    assert f"{USER_ID}/nvidia-basic-dev:latest" in result.output
    assert "300.0 MB" in result.output


def test_list_cli_ignores_stale_building_zombie_when_completed_is_newer(monkeypatch):
    # Production-observed case: a week-old BUILDING VM row that was never
    # reaped sits alongside today's successful build. The CLI must use the
    # newer COMPLETED row, not the zombie.
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    _patch_config(monkeypatch, team_id=None)

    payload = [
        _art(
            "CONTAINER_IMAGE",
            "COMPLETED",
            pushed_at="2026-04-17T18:21:28",
            size_bytes=992_290_762,
        ),
        _art(
            "VM_SANDBOX",
            "COMPLETED",
            pushed_at="2026-04-17T18:21:28",
            size_bytes=2_585_571_832,
        ),
        _art(
            "VM_SANDBOX",
            "BUILDING",
            started_at="2026-04-09T20:52:01",
            created_at="2026-04-09T20:52:00",
        ),
    ]

    monkeypatch.setattr("prime_cli.commands.images.APIClient", _build_dummy_client(payload))

    result = runner.invoke(app, ["images", "list"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    assert "Ready / Ready" in result.output
    assert "rebuilding" not in result.output
    assert "Building" not in result.output


def test_list_cli_shows_building_when_build_is_newer_than_last_completed(monkeypatch):
    # Real rebuild in progress: BUILDING row has a later startedAt than the
    # previous successful build's pushedAt, so Building is the current truth.
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    _patch_config(monkeypatch, team_id=None)

    payload = [
        _art(
            "CONTAINER_IMAGE",
            "COMPLETED",
            pushed_at="2026-04-15T20:57:52",
            size_bytes=50 * 1024 * 1024,
        ),
        _art(
            "VM_SANDBOX",
            "COMPLETED",
            pushed_at="2026-04-15T20:57:52",
            size_bytes=50 * 1024 * 1024,
        ),
        _art(
            "CONTAINER_IMAGE",
            "BUILDING",
            started_at="2026-04-17T10:00:05",
            created_at="2026-04-17T10:00:00",
        ),
        _art(
            "VM_SANDBOX",
            "BUILDING",
            started_at="2026-04-17T10:00:05",
            created_at="2026-04-17T10:00:00",
        ),
    ]

    monkeypatch.setattr("prime_cli.commands.images.APIClient", _build_dummy_client(payload))

    result = runner.invoke(app, ["images", "list"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    assert "Building / Building" in result.output
    # Created reflects the newer startedAt of the active build rather than
    # the older pushedAt of the previous release.
    assert "2026-04-17 10:00" in result.output


def test_list_cli_team_listing_keeps_owner_column_and_prefix(monkeypatch):
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    _patch_config(monkeypatch, team_id=TEAM_ID)

    payload = [
        _art(
            "CONTAINER_IMAGE",
            "COMPLETED",
            pushed_at="2026-04-16T22:24:07",
            size_bytes=10 * 1024 * 1024,
            team_id=TEAM_ID,
            image_name="teamapp",
        ),
        _art(
            "VM_SANDBOX",
            "COMPLETED",
            pushed_at="2026-04-16T22:24:07",
            size_bytes=20 * 1024 * 1024,
            team_id=TEAM_ID,
            image_name="teamapp",
        ),
    ]

    monkeypatch.setattr("prime_cli.commands.images.APIClient", _build_dummy_client(payload))

    result = runner.invoke(app, ["images", "list"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    assert "Owner" in result.output
    assert f"team-{TEAM_ID}/teamapp:latest" in result.output


def test_list_cli_container_only_legacy_image(monkeypatch):
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    _patch_config(monkeypatch, team_id=None)

    payload = [
        _art(
            "CONTAINER_IMAGE",
            "COMPLETED",
            pushed_at="2026-04-16T22:24:07",
            size_bytes=10 * 1024 * 1024,
            image_name="legacy",
        ),
    ]

    monkeypatch.setattr("prime_cli.commands.images.APIClient", _build_dummy_client(payload))

    result = runner.invoke(app, ["images", "list"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    assert "Container" in result.output
    assert "Ready" in result.output
    assert "VM" not in result.output


def test_list_cli_first_time_build_shows_building(monkeypatch):
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    _patch_config(monkeypatch, team_id=None)

    payload = [
        _art(
            "CONTAINER_IMAGE",
            "BUILDING",
            started_at="2026-04-17T09:00:05",
            created_at="2026-04-17T09:00:00",
        ),
        _art(
            "VM_SANDBOX",
            "BUILDING",
            started_at="2026-04-17T09:00:05",
            created_at="2026-04-17T09:00:00",
        ),
    ]

    monkeypatch.setattr("prime_cli.commands.images.APIClient", _build_dummy_client(payload))

    result = runner.invoke(app, ["images", "list"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    assert "Container / VM" in result.output
    assert "Building / Building" in result.output
    assert "Container:" not in result.output
    assert "—" in result.output


def test_list_cli_truncates_owner_prefix_on_narrow_terminal(monkeypatch):
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    _patch_config(monkeypatch, team_id=None)

    payload = [
        _art(
            "CONTAINER_IMAGE",
            "COMPLETED",
            pushed_at="2026-04-16T22:24:07",
            size_bytes=1024 * 1024,
            image_name="nvidia-basic-dev",
        ),
    ]

    monkeypatch.setattr("prime_cli.commands.images.APIClient", _build_dummy_client(payload))

    narrow_env = dict(TEST_ENV, COLUMNS="90")
    result = runner.invoke(app, ["images", "list"], env=narrow_env)
    assert result.exit_code == 0, result.output
    assert "nvidia-basic-dev:latest" in result.output
    assert "…" in result.output
    assert f"{USER_ID}/nvidia-basic-dev" not in result.output


def test_list_cli_newest_group_first(monkeypatch):
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    _patch_config(monkeypatch, team_id=None)

    older = _art(
        "CONTAINER_IMAGE",
        "COMPLETED",
        pushed_at="2026-03-24T20:11:46",
        image_name="old-image",
        size_bytes=1024 * 1024,
    )
    newer = _art(
        "CONTAINER_IMAGE",
        "COMPLETED",
        pushed_at="2026-04-16T22:24:07",
        image_name="new-image",
        size_bytes=1024 * 1024,
    )
    payload = [older, newer]

    monkeypatch.setattr("prime_cli.commands.images.APIClient", _build_dummy_client(payload))

    result = runner.invoke(app, ["images", "list"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    new_idx = result.output.find("new-image")
    old_idx = result.output.find("old-image")
    assert 0 <= new_idx < old_idx
