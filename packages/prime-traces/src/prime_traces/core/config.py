"""Lightweight configuration for the Prime Traces SDK.

Mirrors the other prime SDK packages: reads config.json (project-local or
~/.prime) plus environment variables, env taking precedence.
"""

import json
import os
from pathlib import Path
from typing import Optional

CONFIG_DIR_ENV_VAR = "PRIME_CONFIG_DIR"
CONFIG_DIR_NAME = ".prime"
CONFIG_FILE_NAME = "config.json"


def global_config_dir() -> Path:
    """The per-user config directory, ~/.prime."""
    return Path.home() / CONFIG_DIR_NAME


def _owned_by_current_user(path: Path) -> bool:
    """True when `path` (following symlinks) is owned by the running user.

    Project-local configs are picked up by walking parent directories, so on a
    shared machine anyone who can write to an ancestor could plant one that
    points the SDK at their account. Refusing files we don't own closes that.
    Windows has no uid model; the check is skipped there.
    """
    getuid = getattr(os, "getuid", None)
    if getuid is None:
        return True
    try:
        return path.stat().st_uid == getuid()
    except OSError:
        return False


def find_local_config_dir(start: Optional[Path] = None) -> Optional[Path]:
    """Find the nearest project-local config directory, or None.

    Walks from `start` (default: the working directory) up through its parents
    looking for `.prime/config.json`. The walk stops at the home directory:
    `~/.prime` is the global config, and anything above home is not the user's
    project. Only files owned by the current user count.
    """
    try:
        current = (start or Path.cwd()).resolve()
        home = Path.home().resolve()
    except OSError:
        return None
    for directory in (current, *current.parents):
        if directory == home:
            break
        config_file = directory / CONFIG_DIR_NAME / CONFIG_FILE_NAME
        if config_file.is_file() and _owned_by_current_user(config_file):
            return directory / CONFIG_DIR_NAME
    return None


def resolve_config_dir() -> tuple[Path, str]:
    """Choose the config directory: PRIME_CONFIG_DIR > project-local > ~/.prime.

    Returns the directory and its source: "env", "local", or "global".
    """
    explicit = os.getenv(CONFIG_DIR_ENV_VAR)
    if explicit and explicit.strip():
        return Path(explicit.strip()).expanduser(), "env"
    local = find_local_config_dir()
    if local is not None:
        return local, "local"
    return global_config_dir(), "global"


class Config:
    """Minimal configuration class for SDK packages.

    Reads from a project-local or ~/.prime config.json and environment variables.
    """

    DEFAULT_BASE_URL: str = "https://api.primeintellect.ai"

    def __init__(self, config_dir: Optional[Path | str] = None) -> None:
        """Load config from `config_dir`, or from the resolved location when omitted.

        Resolution order: `PRIME_CONFIG_DIR`, then the nearest ancestor of the
        working directory holding `.prime/config.json` (see
        `find_local_config_dir`), then `~/.prime`. A project-local config is a
        complete replacement for the global one, never merged with it.
        """
        if config_dir is not None:
            self.config_dir = Path(config_dir).expanduser()
            self.config_source = "explicit"
        else:
            self.config_dir, self.config_source = resolve_config_dir()
        self.config_file = self.config_dir / CONFIG_FILE_NAME
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from file"""
        config_data: object = {}
        if self.config_file.exists():
            try:
                config_data = json.loads(self.config_file.read_text())
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                config_data = {}
        # Valid JSON that is not an object (a list, a bare string) must degrade
        # the same way invalid JSON does: every accessor assumes a dict, and
        # the client constructs a Config even when all its parameters were
        # passed explicitly — a crash here would take those callers down too.
        self.config = config_data if isinstance(config_data, dict) else {}

    @staticmethod
    def _strip_api_v1(url: str) -> str:
        return url.rstrip("/").removesuffix("/api/v1")

    @property
    def api_key(self) -> str:
        """Get API key with precedence: env > file > empty."""
        return os.getenv("PRIME_API_KEY") or self.config.get("api_key", "")

    @property
    def team_id(self) -> Optional[str]:
        """Get team ID with precedence: env > file > None."""
        team_id = os.getenv("PRIME_TEAM_ID")
        if team_id is not None:
            return team_id
        return self.config.get("team_id") or None

    @property
    def base_url(self) -> str:
        """Get platform API base URL with precedence: env > file > default."""
        env_val = os.getenv("PRIME_API_BASE_URL") or os.getenv("PRIME_BASE_URL")
        if env_val:
            return self._strip_api_v1(env_val)
        return self._strip_api_v1(self.config.get("base_url", self.DEFAULT_BASE_URL))

    @property
    def traces_url(self) -> str:
        """Base URL of the Prime Traces service.

        Prime Traces is a separately deployed service, so it gets its own
        override: precedence is PRIME_TRACES_URL > config "traces_url" >
        the platform base URL. The fallback assumes the service is
        path-routed under the platform domain; whether production uses that
        or a dedicated domain is still an open deployment decision, and this
        property is the single place that absorbs it.

        For local development against the service's compose stack:
        PRIME_TRACES_URL=http://localhost:8083
        """
        env_val = os.getenv("PRIME_TRACES_URL")
        if env_val:
            return self._strip_api_v1(env_val)
        file_val = self.config.get("traces_url")
        if file_val:
            return self._strip_api_v1(file_val)
        return self.base_url
