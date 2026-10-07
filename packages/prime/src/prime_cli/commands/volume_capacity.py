"""Capacity warnings for `prime train --volume` (ENG-6561).

A named volume's size is a hard quota: a checkpoint save that does not fit
fails with "Disk quota exceeded" and the trainer crash-loops. These checks
only print; they never block or fail a launch.
"""

import re
from typing import Any, Dict, Optional

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from ..api.training import HostedTrainingClient, Volume, VolumeUsage
from ..client import APIError
from ..utils.time_utils import format_time_ago

DOCS_URL = "https://docs.primeintellect.ai/hosted-training/volume-capacity"
WARN_FRACTION = 0.8
USAGE_TIMEOUT_SECONDS = 5
GIB = 2**30
_UNITS = {
    **{"": 1, "k": 10**3, "M": 10**6, "G": 10**9, "T": 10**12, "P": 10**15},
    **{"Ki": 2**10, "Mi": 2**20, "Gi": 2**30, "Ti": 2**40, "Pi": 2**50},
}


def parse_size(size: Optional[str]) -> Optional[int]:
    """Bytes in a Kubernetes quantity like "200Gi" or "2Ti"; None if unparseable."""
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([A-Za-z]*)", (size or "").strip())
    if not match or match[2] not in _UNITS:
        return None
    return int(float(match[1]) * _UNITS[match[2]])


def warn_volume_usage(
    client: HostedTrainingClient, volume: Volume, team_id: Optional[str], out: Console
) -> None:
    """Warn loudly when `volume` is at least WARN_FRACTION full. Platforms
    without the usage endpoint (404) or any failure get one short line."""
    name = escape(volume.name)
    try:
        usage = client.get_volume_usage(volume.name, team_id=team_id, timeout=USAGE_TIMEOUT_SECONDS)
    except (APIError, ValueError):
        usage = VolumeUsage()
    size = usage.size or volume.size
    quota = parse_size(size)
    used = usage.used_bytes
    if used is None or not quota:
        reason = f" ({escape(usage.unavailable_reason)})" if usage.unavailable_reason else ""
        out.print(
            f"Volume '{name}' (quota {escape(size or 'unknown')}): usage unavailable{reason}. "
            f"Leave room for checkpoints: {DOCS_URL}",
            highlight=False,
        )
        return
    if used < WARN_FRACTION * quota:
        return

    try:
        age = f"measured {format_time_ago(usage.observed_at)}" if usage.observed_at else None
    except ValueError:
        age = None
    free = max(quota - used, 0)
    body = (
        f"[bold]{used / GIB:.1f} GiB used of {quota / GIB:.1f} GiB ({escape(size or '')}), "
        f"{free / GIB:.1f} GiB free[/bold] ({age or 'measurement time unknown'}).\n\n"
        "The size is a hard quota. A checkpoint save that does not fit fails with "
        '"Disk quota exceeded" and the run crash-loops.\n'
        "A save needs room for keep_last + 1 checkpoints: the new one is written "
        "before old ones are pruned. RL rollout traces grow for the whole run.\n\n"
        f"Free space with `prime volumes ssh {name} --read-write`, or grow it with "
        f"`prime volumes resize {name} --size <bigger>`.\n"
        f"Docs: {DOCS_URL}"
    )
    out.print(
        Panel(
            body,
            title=f"[bold]Warning: volume '{name}' is {used / quota:.0%} full[/bold]",
            border_style="bold yellow",
        ),
        highlight=False,
    )


def warn_checkpoint_retention(cfg: Dict[str, Any], volume: str, out: Console) -> None:
    """Warn when trainer checkpoints are saved mid-run (`[ckpt]` or
    `[trainer.ckpt]` with an `interval`) without a bound on how many stay on
    the volume."""
    trainer = cfg.get("trainer")
    sub = trainer.get("ckpt") if isinstance(trainer, dict) else None
    tables = [t for t in (cfg.get("ckpt"), sub) if isinstance(t, dict)]
    if sub == "None" or not tables:
        return
    ckpt = {k: v for t in tables for k, v in t.items()}
    # Without `interval` the trainer saves only once, at the end of training.
    if ckpt.get("interval") is None:
        return
    name = escape(volume)
    if ckpt.get("keep_last") is None:
        out.print(
            "[yellow]Warning:[/yellow] trainer checkpoints are on but `keep_last` is not "
            f"set, so every checkpoint stays on volume '{name}' until it is full. Set "
            "`keep_last` under \\[ckpt]; a save needs room for keep_last + 1 checkpoints. "
            f"See {DOCS_URL}",
            highlight=False,
        )
    if ckpt.get("keep_interval") is not None:
        out.print(
            f"[yellow]Warning:[/yellow] `keep_interval = {escape(str(ckpt['keep_interval']))}` "
            f"keeps every matching checkpoint on volume '{name}' for good; they are never "
            f"pruned. See {DOCS_URL}",
            highlight=False,
        )
