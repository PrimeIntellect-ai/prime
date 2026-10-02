"""Tests for `prime images list` rendering helpers and end-to-end output."""

from __future__ import annotations

import json
from typing import Any, Callable

import pytest
from prime_cli.commands import images as images_cmd
from prime_cli.commands.images import (
    ImageRow,
    _completed_size_mb,
    _display_created,
    _image_ref_column_width,
    _latest,
    _render_image_reference,
    _render_status,
    _row_timestamp,
    _truncate_ref_left,
)
from prime_cli.main import app
from prime_sandboxes import (
    ImageArtifactType,
    ImageBuildStatus,
    ImageListItem,
    ImageListResponse,
    ImageOwnerType,
)
from rich.text import Text
from typer.testing import CliRunner

USER_ID = "cmkrcib4x00004kjyxq48nltd"
TEAM_ID = "team-abc123"

TEST_ENV: dict[str, str] = {
    "LINES": "50",
    "NO_COLOR": "1",
    "PRIME_DISABLE_VERSION_CHECK": "1",
}


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _vm(
    status: ImageBuildStatus = ImageBuildStatus.COMPLETED,
    *,
    image: str = "nvidia-basic-dev:latest",
    pushed_at: str | None = None,
    completed_at: str | None = None,
    started_at: str | None = None,
    created_at: str = "2026-04-01T10:00:00",
    size_bytes: int | None = None,
    team_id: str | None = None,
) -> ImageRow:
    """Build a typed image-list item carrying the fields the CLI reads."""
    name, _, tag = image.partition(":")
    tag = tag or "latest"
    scope = f"team-{team_id}" if team_id else USER_ID
    return ImageListItem.model_validate(
        {
            "id": f"{name}:{tag}",
            "artifactType": ImageArtifactType.VM_SANDBOX,
            "imageName": name,
            "imageTag": tag,
            "status": status,
            "sizeBytes": size_bytes,
            "createdAt": created_at,
            "startedAt": started_at,
            "completedAt": completed_at,
            "pushedAt": pushed_at,
            "teamId": team_id,
            "ownerType": ImageOwnerType.TEAM if team_id else ImageOwnerType.PERSONAL,
            "displayRef": f"prime/{scope}/{name}:{tag}",
        }
    )


# ---------------------------------------------------------------------------
# _latest — only the newest row per group survives
# ---------------------------------------------------------------------------


def test_latest_keeps_newest_row_per_group():
    latest = _latest(
        [
            _vm(pushed_at="2026-04-16T22:24:07"),
            _vm(status=ImageBuildStatus.FAILED, completed_at="2026-04-16T21:00:00"),
            _vm(status=ImageBuildStatus.FAILED, completed_at="2026-04-16T20:55:00"),
        ]
    )
    assert latest.status == ImageBuildStatus.COMPLETED


def test_latest_fresh_completed_wins_over_stale_building_zombie():
    # Real-world case: a stale BUILDING row from 8 days ago (never reaped)
    # alongside a fresh COMPLETED push. The fresher COMPLETED row wins.
    latest = _latest(
        [
            _vm(
                status=ImageBuildStatus.BUILDING,
                started_at="2026-04-09T20:52:01",
                created_at="2026-04-09T20:52:00",
            ),
            _vm(pushed_at="2026-04-17T18:21:28", completed_at="2026-04-17T18:21:28"),
        ]
    )
    assert latest.status == ImageBuildStatus.COMPLETED


def test_latest_active_build_wins_over_older_completed():
    latest = _latest(
        [
            _vm(pushed_at="2026-04-15T20:57:52"),
            _vm(
                status=ImageBuildStatus.BUILDING,
                started_at="2026-04-17T10:00:05",
                created_at="2026-04-17T10:00:00",
            ),
        ]
    )
    assert latest.status == ImageBuildStatus.BUILDING


def test_latest_surfaces_failure_when_only_failed_rows_exist():
    latest = _latest(
        [
            _vm(status=ImageBuildStatus.FAILED, completed_at="2026-04-16T21:00:00"),
        ]
    )
    assert latest.status == ImageBuildStatus.FAILED


# ---------------------------------------------------------------------------
# _render_status — renders the raw status of the latest row
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status,expected_plain,expected_style",
    [
        (ImageBuildStatus.COMPLETED, "Ready", "green"),
        (ImageBuildStatus.BUILDING, "Building", "yellow"),
        (ImageBuildStatus.UPLOADING, "Uploading", "yellow"),
        (ImageBuildStatus.PENDING, "Pending", "blue"),
        (ImageBuildStatus.FAILED, "Failed", "red"),
        (ImageBuildStatus.CANCELLED, "Cancelled", "dim"),
    ],
)
def test_render_status_known(status, expected_plain, expected_style):
    assert _render_status(_vm(status=status)) == Text(expected_plain, style=expected_style)


# ---------------------------------------------------------------------------
# _render_image_reference
# ---------------------------------------------------------------------------


def test_render_image_reference_always_shows_user_prefix_for_personal():
    assert (
        _render_image_reference(_vm(image="myapp:v1"), is_team_listing=False)
        == f"prime/{USER_ID}/myapp:v1"
    )


def test_render_image_reference_keeps_team_prefix_for_team_listing():
    assert (
        _render_image_reference(_vm(image="myapp:v1", team_id=TEAM_ID), is_team_listing=True)
        == f"prime/team-{TEAM_ID}/myapp:v1"
    )


def test_render_image_reference_falls_back_without_display_ref():
    image = _vm(image="fallback-image:v2").model_copy(
        update={"display_ref": None, "full_image_path": None}
    )
    assert _render_image_reference(image, is_team_listing=False) == "fallback-image:v2"


# ---------------------------------------------------------------------------
# _truncate_ref_left
# ---------------------------------------------------------------------------


def test_truncate_ref_left_returns_unchanged_when_within_width():
    assert _truncate_ref_left("short/name:tag", 40) == "short/name:tag"


def test_truncate_ref_left_clips_head_and_preserves_name_tag():
    ref = f"prime/{USER_ID}/nvidia-basic-dev:latest"
    out = _truncate_ref_left(ref, 30)
    assert out.endswith("nvidia-basic-dev:latest")
    assert out.startswith("…")
    assert len(out) == 30


@pytest.mark.parametrize("budget", [0, -5])
def test_truncate_ref_left_handles_zero_or_negative_budget(budget):
    ref = f"prime/{USER_ID}/x:y"
    assert _truncate_ref_left(ref, budget) == ref


def test_truncate_ref_left_clamps_tiny_budget_to_two():
    assert _truncate_ref_left("abcdefghij", 1) == "…j"


# ---------------------------------------------------------------------------
# _image_ref_column_width
# ---------------------------------------------------------------------------


def test_image_ref_column_width_wide_terminal_capped_at_80():
    assert _image_ref_column_width(300, is_team_listing=False) == 80


def test_image_ref_column_width_narrow_terminal_clamped_to_floor():
    assert _image_ref_column_width(60, is_team_listing=False) == 30


def test_image_ref_column_width_team_shrinks_ref_budget():
    w_personal = _image_ref_column_width(140, is_team_listing=False)
    w_team = _image_ref_column_width(140, is_team_listing=True)
    assert 20 <= w_team < w_personal <= 80


# ---------------------------------------------------------------------------
# _completed_size_mb
# ---------------------------------------------------------------------------


def test_completed_size_shows_latest_completed_row():
    row = _vm(pushed_at="2026-04-16T22:00:00", size_bytes=200 * 1024 * 1024)
    assert _completed_size_mb(row) == "200.0 MB"


def test_completed_size_ignores_older_completed_if_newer_is_not_completed():
    # If the latest row is BUILDING it contributes nothing — even if an
    # older COMPLETED row has a size. Size reflects the *current* picture.
    row = _vm(
        status=ImageBuildStatus.BUILDING,
        started_at="2026-04-17T10:00:05",
        created_at="2026-04-17T10:00:00",
        size_bytes=100 * 1024 * 1024,
    )
    assert _completed_size_mb(row) == Text("—", style="dim")


def test_completed_size_returns_dash_when_size_missing():
    row = _vm(pushed_at="2026-04-16T22:00:00")
    assert _completed_size_mb(row) == Text("—", style="dim")


# ---------------------------------------------------------------------------
# _display_created / _row_timestamp
# ---------------------------------------------------------------------------


def test_display_created_uses_pushed_at_when_completed():
    row = _vm(pushed_at="2026-04-16T22:24:07")
    assert _display_created(row) == "2026-04-16 22:24"


def test_display_created_falls_back_to_started_at_for_active_builds():
    row = _vm(
        status=ImageBuildStatus.BUILDING,
        started_at="2026-04-17T09:00:00",
        created_at="2026-04-17T08:59:00",
    )
    assert _display_created(row) == "2026-04-17 09:00"


def test_display_created_falls_back_to_completed_at_for_failures():
    row = _vm(status=ImageBuildStatus.FAILED, completed_at="2026-04-16T12:00:00")
    assert _display_created(row) == "2026-04-16 12:00"


def test_row_timestamp_matches_display_created():
    row = _vm(pushed_at="2026-04-16T22:24:07")
    assert _row_timestamp(row).strftime("%Y-%m-%d %H:%M") == _display_created(row)


# ---------------------------------------------------------------------------
# End-to-end: run the CLI with a mocked API client
# ---------------------------------------------------------------------------


class _StubConfig:
    def __init__(self, team_id: str | None = None) -> None:
        self.team_id = team_id


def _image_list_response(
    payload: list[ImageRow],
    *,
    offset: int = 0,
    limit: int = 50,
    total_count: int | None = None,
    include_total_count: bool = True,
) -> ImageListResponse:
    logical_images = {
        (row.owner_type, row.team_id, row.image_name, row.image_tag) for row in payload
    }
    pagination = {}
    if include_total_count:
        pagination["total_count"] = len(logical_images) if total_count is None else total_count
    return ImageListResponse(data=payload, offset=offset, limit=limit, **pagination)


@pytest.fixture
def run_images_list(monkeypatch) -> Callable[..., Any]:
    """Invoke ``prime images list`` against a typed ``ImageClient.list`` stub."""
    runner = CliRunner()

    def _run(
        payload: list[ImageRow],
        *,
        team_id: str | None = None,
        env: dict[str, str] | None = None,
        plain: bool = False,
    ):
        class DummyImageClient:
            def __init__(self, _api_client) -> None:
                pass

            def list(
                self,
                *,
                team_id=None,
                search=None,
                platform=False,
                offset=0,
                limit=50,
            ):
                return _image_list_response(payload, offset=offset, limit=limit)

        monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
        monkeypatch.setattr(images_cmd, "config", _StubConfig(team_id=team_id))
        monkeypatch.setattr(images_cmd, "ImageClient", DummyImageClient)
        command_env = env or TEST_ENV
        monkeypatch.setattr(images_cmd.console, "_width", int(command_env.get("COLUMNS", "200")))
        monkeypatch.setattr(images_cmd.console, "_height", int(command_env.get("LINES", "50")))
        args = ["images", "list"] + (["--plain"] if plain else [])
        return runner.invoke(app, args, env=command_env)

    return _run


def test_list_cli_shows_latest_status(run_images_list):
    result = run_images_list(
        [
            _vm(pushed_at="2026-04-16T22:24:07", size_bytes=200 * 1024 * 1024),
            _vm(status=ImageBuildStatus.FAILED, completed_at="2026-04-16T20:00:00"),
        ]
    )
    assert result.exit_code == 0, result.output
    assert "Ready" in result.output
    assert "Failed" not in result.output  # stale, hidden by newer COMPLETED
    assert f"prime/{USER_ID}/nvidia-basic-dev:latest" in result.output
    assert "200.0 MB" in result.output


def test_list_cli_ignores_stale_building_zombie_when_completed_is_newer(run_images_list):
    result = run_images_list(
        [
            _vm(pushed_at="2026-04-17T18:21:28", size_bytes=2_585_571_832),
            # The week-old stuck BUILDING row (as seen in production DB).
            _vm(
                status=ImageBuildStatus.BUILDING,
                started_at="2026-04-09T20:52:01",
                created_at="2026-04-09T20:52:00",
            ),
        ]
    )
    assert result.exit_code == 0, result.output
    assert "Ready" in result.output
    assert "Building" not in result.output


def test_list_cli_shows_building_when_build_is_newer_than_last_completed(run_images_list):
    result = run_images_list(
        [
            _vm(pushed_at="2026-04-15T20:57:52", size_bytes=50 * 1024 * 1024),
            _vm(
                status=ImageBuildStatus.BUILDING,
                started_at="2026-04-17T10:00:05",
                created_at="2026-04-17T10:00:00",
            ),
        ]
    )
    assert result.exit_code == 0, result.output
    assert "Building" in result.output
    assert "2026-04-17 10:00" in result.output


def test_list_cli_plain_output_has_no_markup_leaks(run_images_list):
    result = run_images_list(
        [
            _vm(
                image="plainapp:latest",
                team_id=TEAM_ID,
                pushed_at="2026-04-16T22:24:07",
                size_bytes=10 * 1024 * 1024,
            ),
        ],
        team_id=TEAM_ID,
        plain=True,
        env=dict(TEST_ENV, COLUMNS="200"),
    )
    assert result.exit_code == 0, result.output
    assert "Ready" in result.output
    assert "Private" in result.output
    assert "Team" in result.output
    for tag in ("[cyan]", "[green]", "[dim]", "[blue]"):
        assert tag not in result.output


def test_list_cli_team_listing_keeps_owner_column_and_prefix(run_images_list):
    result = run_images_list(
        [
            _vm(
                image="teamapp:latest",
                team_id=TEAM_ID,
                pushed_at="2026-04-16T22:24:07",
                size_bytes=20 * 1024 * 1024,
            ),
        ],
        team_id=TEAM_ID,
    )
    assert result.exit_code == 0, result.output
    assert "Owner" in result.output
    assert f"prime/team-{TEAM_ID}/teamapp:latest" in result.output


def test_list_cli_first_time_build_shows_building(run_images_list):
    result = run_images_list(
        [
            _vm(
                status=ImageBuildStatus.BUILDING,
                started_at="2026-04-17T09:00:05",
                created_at="2026-04-17T09:00:00",
            ),
        ]
    )
    assert result.exit_code == 0, result.output
    assert "Building" in result.output
    assert "—" in result.output


def test_list_cli_truncates_owner_prefix_on_narrow_terminal(run_images_list):
    result = run_images_list(
        [_vm(pushed_at="2026-04-16T22:24:07", size_bytes=1024 * 1024)],
        env=dict(TEST_ENV, COLUMNS="90"),
    )
    assert result.exit_code == 0, result.output
    assert "nvidia-basic-dev:latest" in result.output
    assert "…" in result.output
    assert f"prime/{USER_ID}/nvidia-basic-dev" not in result.output


def _run_list_capturing_params(
    monkeypatch,
    args: list[str],
    *,
    payload: list[ImageRow] | None = None,
    team_id: str | None = None,
    total_count: int | None = None,
    include_total_count: bool = True,
):
    """Invoke the CLI and capture arguments forwarded to ``ImageClient.list``."""
    captured: dict[str, Any] = {}

    class DummyImageClient:
        def __init__(self, _api_client) -> None:
            pass

        def list(
            self,
            *,
            team_id=None,
            search=None,
            platform=False,
            offset=0,
            limit=50,
        ):
            captured["params"] = {
                "team_id": team_id,
                "search": search,
                "platform": platform,
                "offset": offset,
                "limit": limit,
            }
            return _image_list_response(
                payload or [],
                offset=offset,
                limit=limit,
                total_count=total_count,
                include_total_count=include_total_count,
            )

    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    monkeypatch.setattr(images_cmd, "config", _StubConfig(team_id=team_id))
    monkeypatch.setattr(images_cmd, "ImageClient", DummyImageClient)
    result = CliRunner().invoke(app, ["images", "list", *args], env=TEST_ENV)
    return result, captured


def test_list_forwards_search_param(monkeypatch):
    result, captured = _run_list_capturing_params(
        monkeypatch,
        ["--search", "myapp"],
        payload=[_vm(image="myapp:latest", pushed_at="2026-04-16T22:24:07")],
    )
    assert result.exit_code == 0, result.output
    assert captured["params"].get("search") == "myapp"


def test_list_search_short_flag_q_forwards_param(monkeypatch):
    result, captured = _run_list_capturing_params(
        monkeypatch,
        ["-q", "nvidia"],
        payload=[_vm(pushed_at="2026-04-16T22:24:07")],
    )
    assert result.exit_code == 0, result.output
    assert captured["params"].get("search") == "nvidia"


def test_list_without_search_omits_param(monkeypatch):
    result, captured = _run_list_capturing_params(
        monkeypatch,
        [],
        payload=[_vm(pushed_at="2026-04-16T22:24:07")],
    )
    assert result.exit_code == 0, result.output
    assert captured["params"].get("search") is None


def test_list_empty_search_result_shows_search_aware_message(monkeypatch):
    result, captured = _run_list_capturing_params(
        monkeypatch,
        ["--search", "doesnotexist"],
        payload=[],
    )
    assert result.exit_code == 0, result.output
    assert captured["params"].get("search") == "doesnotexist"
    assert "No images match 'doesnotexist'" in result.output


def test_list_search_out_of_range_page_shows_page_guidance(monkeypatch):
    result, captured = _run_list_capturing_params(
        monkeypatch,
        ["--search", "myapp", "--page", "2"],
        payload=[],
        total_count=1,
    )
    assert result.exit_code == 0, result.output
    assert captured["params"]["offset"] == 50
    assert "No images on page" in result.output
    assert "No images match" not in result.output


def test_list_search_out_of_range_page_without_total_count_shows_page_guidance(monkeypatch):
    result, _ = _run_list_capturing_params(
        monkeypatch,
        ["--search", "myapp", "--page", "2"],
        payload=[],
        include_total_count=False,
    )

    assert result.exit_code == 0, result.output
    assert "No images on page" in result.output
    assert "No images match" not in result.output


def test_list_search_first_page_without_total_count_shows_no_matches(monkeypatch):
    result, _ = _run_list_capturing_params(
        monkeypatch,
        ["--search", "myapp"],
        payload=[],
        include_total_count=False,
    )

    assert result.exit_code == 0, result.output
    assert "No images match 'myapp'" in result.output


def test_list_out_of_range_page_without_total_count_no_search(monkeypatch):
    result, _ = _run_list_capturing_params(
        monkeypatch,
        ["--page", "3"],
        payload=[],
        include_total_count=False,
    )

    assert result.exit_code == 0, result.output
    assert "No images on page" in result.output


def test_list_cli_newest_group_first(run_images_list):
    result = run_images_list(
        [
            _vm(image="old-image:latest", pushed_at="2026-03-24T20:11:46", size_bytes=1024 * 1024),
            _vm(image="new-image:latest", pushed_at="2026-04-16T22:24:07", size_bytes=1024 * 1024),
        ]
    )
    assert result.exit_code == 0, result.output
    new_idx = result.output.find("new-image")
    old_idx = result.output.find("old-image")
    assert 0 <= new_idx < old_idx


# ---------------------------------------------------------------------------
# --platform-image — org-less platform image listing (admins only)
# ---------------------------------------------------------------------------


def _platform_row(**kw: Any) -> ImageRow:
    row = _vm(**kw)
    return row.model_copy(
        update={
            "owner_type": ImageOwnerType.PLATFORM,
            "display_ref": f"{row.image_name}:{row.image_tag}",
        }
    )


def test_list_platform_image_forwards_owner_scope(monkeypatch):
    result, captured = _run_list_capturing_params(
        monkeypatch,
        ["--platform-image"],
        payload=[_platform_row(image="ubuntu:22.04", pushed_at="2026-04-16T22:24:07")],
    )
    assert result.exit_code == 0, result.output
    assert captured["params"].get("platform") is True
    assert captured["params"].get("team_id") is None
    assert "Platform Images" in result.output


def test_list_platform_image_ignores_team_context(monkeypatch):
    result, captured = _run_list_capturing_params(
        monkeypatch,
        ["--platform-image"],
        payload=[_platform_row(image="ubuntu:22.04", pushed_at="2026-04-16T22:24:07")],
        team_id=TEAM_ID,
    )
    assert result.exit_code == 0, result.output
    assert captured["params"].get("platform") is True
    assert captured["params"].get("team_id") is None
    assert "Team context ignored" in result.output
    assert "Platform Images" in result.output
    # Platform listings never render the team-scoped Owner column.
    assert "Owner" not in result.output


def test_list_platform_image_team_context_json_output_stays_clean(monkeypatch):
    result, captured = _run_list_capturing_params(
        monkeypatch,
        ["--platform-image", "--json"],
        payload=[_platform_row(image="ubuntu:22.04", pushed_at="2026-04-16T22:24:07")],
        team_id=TEAM_ID,
    )
    assert result.exit_code == 0, result.output
    assert captured["params"].get("platform") is True
    assert captured["params"].get("team_id") is None
    assert "Team context ignored" not in result.output
    payload = json.loads(result.output)
    assert payload["totalCount"] == 1
    assert "total_count" not in payload
    assert payload["data"][0]["artifactType"] == "VM_SANDBOX"


def test_list_json_omits_total_count_when_api_omits_it(monkeypatch):
    result, _ = _run_list_capturing_params(
        monkeypatch,
        ["--json"],
        payload=[_vm(pushed_at="2026-04-16T22:24:07")],
        include_total_count=False,
    )

    assert result.exit_code == 0, result.output
    assert "totalCount" not in json.loads(result.output)


def test_list_platform_image_empty_shows_platform_push_hint(monkeypatch):
    result, _ = _run_list_capturing_params(
        monkeypatch,
        ["--platform-image"],
        payload=[],
    )
    assert result.exit_code == 0, result.output
    assert "--platform-image" in result.output


def test_list_without_platform_flag_omits_owner_scope(monkeypatch):
    result, captured = _run_list_capturing_params(
        monkeypatch,
        [],
        payload=[_vm(pushed_at="2026-04-16T22:24:07")],
    )
    assert result.exit_code == 0, result.output
    assert captured["params"].get("platform") is False
