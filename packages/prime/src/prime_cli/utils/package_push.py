"""The parts of a package push that `prime env push` and `prime pipelines push` share:
finding who the package belongs to, building its wheel and source archive, and
naming it the way the Hub does."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import zipfile
from email.parser import Parser as EmailParser
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from packaging.utils import InvalidWheelFilename, canonicalize_name, parse_wheel_filename

from ..client import APIClient, APIError
from .hub import HubAPIError
from .package_archive import write_source_archive
from .plain import get_console

# The platform's `TEAM_ID_PATTERN` (a cuid); anything else configured is a slug.
_TEAM_ID = re.compile(r"^c[a-z0-9]{24}$")
_USERNAME = re.compile(r"^[a-z0-9-]{3,30}$")


# What to tell someone whose push the Hub refused, by error code.
PUSH_ERROR_TIPS = {
    "version_exists": "Each version is published once. Bump the version in pyproject.toml.",
    "release_superseded": "Another push of this version finished first. Push again.",
    "name_taken": "Another owner already publishes a verifiers v1 package by this name.",
    "name_tombstoned": "A deleted package had this name, so it can't be used again.",
    "name_reserved": "This name is reserved. Choose another in pyproject.toml.",
    "package_exists": "You already have a package by this name.",
    "invalid_components": "Legacy (v0) packages can't declare components.",
    "forbidden": "Check --owner/--team: you need write access to push there.",
}


def print_push_error(console: Any, error: HubAPIError, *, bump_hint: str = "") -> None:
    """Say why the Hub refused a push, and what to do about it.

    `bump_hint` follows the tip for a version that's already published, naming
    the command's own way to bump it.
    """
    console.print(f"[red]Push failed: {error.message} ({error.code})[/red]")
    tip = PUSH_ERROR_TIPS.get(error.code)
    if tip and error.code == "version_exists" and bump_hint:
        tip = f"{tip} {bump_hint}"
    if tip:
        console.print(f"[yellow]{tip}[/yellow]")


class PushError(Exception):
    """A push can't go on; the message says why, for the person pushing."""


def hub_names(name: str) -> List[str]:
    """The names to look a package up by, the normalized one last.

    The Hub creates packages under PEP 503-normalized names (`math-env`). It
    matches names without regard to case, but not `_` against `-`, and packages
    pushed before names were normalized kept their own spelling, so `math_env`
    is tried too. A new package takes the last name.
    """
    canonical = canonicalize_name(name)
    others = [n for n in (name, canonical.replace("-", "_")) if n != canonical]
    return [*dict.fromkeys(others), canonical]


def check_name_matches_distribution(name: str, distribution: str) -> None:
    """The Hub checks that a wheel's filename names the package it's pushed to."""
    if canonicalize_name(name) != canonicalize_name(distribution):
        raise PushError(
            f"The package is '{distribution}' in pyproject.toml, so it can't be pushed "
            f"as '{name}'. Rename the project in pyproject.toml instead."
        )


def _teams(api: APIClient) -> List[Dict[str, Any]]:
    teams: List[Dict[str, Any]] = []
    offset = 0
    while True:
        response = api.get("/user/teams", params={"offset": offset, "limit": 100})
        batch = response.get("data", []) if isinstance(response, dict) else []
        teams.extend(batch)
        total = response.get("total_count", len(teams)) if isinstance(response, dict) else 0
        if not batch or len(teams) >= total:
            return teams
        offset += len(batch)


def _prompt_for_username(api: APIClient) -> str:
    """Ask for a username until one is free, and set it."""
    console = get_console()
    print_ = console.print
    print_(
        "[yellow]Your user profile is missing a username.[/yellow] You must choose one to publish."
    )
    print_("[dim]Note: This username can only be chosen once and will be public.[/dim]")
    while True:
        try:
            chosen = typer.prompt("Enter your desired username").strip().lower()
        except typer.Abort:
            raise PushError("Cancelled") from None
        if not _USERNAME.match(chosen):
            print_(
                "[red]Invalid username.[/red] "
                "Use 3-30 chars with lowercase letters, numbers, and '-' only."
            )
            continue
        try:
            api.patch("/user/slug", json={"slug": chosen})
        except APIError as e:
            message = str(e)
            if "409" in message or "already taken" in message.lower():
                print_("[red]That username is already taken.[/red] Please choose another.")
                continue
            raise PushError(f"Failed to set username: {e}") from e
        print_(f"[green]✓ Username set to {chosen}[/green]")
        return chosen


def resolve_owner(
    api: APIClient,
    *,
    owner: Optional[str],
    team: Optional[str],
    configured_team: Optional[str],
) -> str:
    """The slug to push under: `--owner`, else `--team`, else the configured team,
    else your own username (asking you to choose one if you have none)."""
    if owner:
        return owner.strip()
    if team:
        return team.strip()
    if configured_team and configured_team.strip():
        configured_team = configured_team.strip()
        if not _TEAM_ID.match(configured_team):
            return configured_team
        for item in _teams(api):
            if item.get("teamId") == configured_team and item.get("slug"):
                return str(item["slug"])
        raise PushError(
            "Your configured team isn't one of your teams. Pass --team or --owner, "
            "or change it with 'prime config set-team'."
        )
    response = api.get("/user/whoami")
    data = response.get("data") if isinstance(response, dict) else None
    slug = data.get("slug") if isinstance(data, dict) else None
    return str(slug) if slug else _prompt_for_username(api)


def build_wheel(package_dir: Path) -> Path:
    """Build the package's wheel into a clean `dist/` and return it."""
    dist_dir = package_dir / "dist"
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    if shutil.which("uv"):
        command = ["uv", "build", "--wheel", "--out-dir", "dist"]
    else:
        command = [sys.executable, "-m", "build", "--wheel", "--outdir", "dist", "."]
    try:
        subprocess.run(command, cwd=package_dir, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise PushError(f"Build failed:\n{e.stderr}") from e
    except FileNotFoundError as e:
        raise PushError("Build tool not found. Please install 'uv' or 'build'.") from e
    wheels = sorted(dist_dir.glob("*.whl"))
    if not wheels:
        raise PushError("No wheel file found after build")
    return wheels[0]


def wheel_distribution(wheel_path: Path) -> str:
    try:
        name, _version, _build, _tags = parse_wheel_filename(wheel_path.name)
    except InvalidWheelFilename as e:
        raise PushError(f"'{wheel_path.name}' isn't a valid wheel filename") from e
    return str(name)


def requires_dist_of(wheel_path: Path) -> List[str]:
    """The wheel's `Requires-Dist` entries, URL dependencies included."""
    try:
        with zipfile.ZipFile(wheel_path, "r") as wheel:
            metadata_files = [n for n in wheel.namelist() if n.endswith(".dist-info/METADATA")]
            if not metadata_files:
                return []
            content = wheel.read(metadata_files[0]).decode("utf-8")
    except (zipfile.BadZipFile, KeyError, UnicodeDecodeError):
        return []
    return EmailParser().parsestr(content).get_all("Requires-Dist") or []


def write_source(package_dir: Path, out_dir: Path, distribution: str, version: str) -> Path:
    """The source archive, named as an sdist so the Hub accepts it."""
    filename = f"{canonicalize_name(distribution).replace('-', '_')}-{version}.tar.gz"
    return write_source_archive(package_dir, out_dir / filename)
