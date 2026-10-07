"""Capacity warnings for `prime train --volume` (ENG-6561)."""

import io
from types import SimpleNamespace

from prime_cli.api.training import Volume, VolumeUsage
from prime_cli.client import APIError, NotFoundError
from prime_cli.commands.volume_capacity import (
    DOCS_URL,
    parse_size,
    warn_checkpoint_retention,
    warn_volume_usage,
)
from rich.console import Console

GIB = 2**30
VOL = Volume(name="ckpts", size="200Gi", status="RUNNING", clusterId="c", pvcName="p")


def _usage_output(result) -> str:
    def get_volume_usage(name, team_id=None, timeout=None):
        if isinstance(result, Exception):
            raise result
        return VolumeUsage.model_validate(result)

    out = Console(file=io.StringIO(), width=200, height=50)
    warn_volume_usage(SimpleNamespace(get_volume_usage=get_volume_usage), VOL, None, out)
    return out.file.getvalue()


def test_parse_size() -> None:
    assert parse_size("200Gi") == 200 * GIB
    assert parse_size("2Ti") == 2 * 2**40
    assert parse_size("500G") == 500 * 10**9
    assert parse_size(None) is None and parse_size("lots") is None


def test_usage_over_threshold_prints_loud_warning() -> None:
    text = _usage_output(
        {"name": "ckpts", "size": "200Gi", "usedBytes": 190 * GIB, "observedAt": None}
    )
    assert "volume 'ckpts' is 95% full" in text
    assert "190.0 GiB used of 200.0 GiB" in text and "10.0 GiB free" in text
    assert "keep_last + 1" in text and DOCS_URL in text


def test_usage_below_threshold_is_silent() -> None:
    assert _usage_output({"size": "200Gi", "usedBytes": 100 * GIB}) == ""


def test_usage_unavailable_prints_one_line_with_quota_and_link() -> None:
    text = _usage_output(NotFoundError("HTTP 404"))
    assert text.count("\n") == 1
    assert "quota 200Gi" in text and "usage unavailable" in text and DOCS_URL in text

    text = _usage_output({"size": "200Gi", "usedBytes": None, "unavailableReason": "not mounted"})
    assert "usage unavailable (not mounted)" in text


def test_usage_errors_never_raise() -> None:
    assert "usage unavailable" in _usage_output(APIError("Request timed out"))
    assert "usage unavailable" in _usage_output({"usedBytes": "lots"})
    text = _usage_output({"usedBytes": 199 * GIB, "observedAt": "yesterday-ish"})
    assert "measurement time unknown" in text


def _retention_output(cfg: dict) -> str:
    out = Console(file=io.StringIO(), width=300, height=50)
    warn_checkpoint_retention(cfg, "ckpts", out)
    return out.file.getvalue()


def test_checkpoints_without_keep_last_get_an_advisory() -> None:
    for cfg in (
        {"ckpt": {"interval": 50}},
        {"trainer": {"ckpt": {"interval": 50}}},
        {"ckpt": {"interval": 50}, "trainer": {"ckpt": {}}},
    ):
        text = _retention_output(cfg)
        assert "`keep_last` is not set" in text and DOCS_URL in text
    assert "keep_interval = 100" in _retention_output(
        {"ckpt": {"interval": 50, "keep_last": 2, "keep_interval": 100}}
    )


def test_bounded_or_disabled_checkpoints_are_silent() -> None:
    for cfg in (
        {},
        {"ckpt": "None"},
        {"ckpt": {"interval": 50}, "trainer": {"ckpt": "None"}},
        {"ckpt": {"interval": 50, "keep_last": 2}},
        {"ckpt": {"interval": 50}, "trainer": {"ckpt": {"keep_last": 2}}},
        # No interval: only the final checkpoint is saved.
        {"ckpt": {}},
        {"trainer": {"ckpt": {"keep_interval": 100}}},
    ):
        assert _retention_output(cfg) == "", cfg
