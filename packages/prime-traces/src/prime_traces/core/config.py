"""Lightweight configuration for the Prime Traces SDK.

Mirrors the other prime SDK packages: reads config.json (project-local or
~/.prime) plus environment variables, env taking precedence.
"""

import hashlib
import json
import os
import warnings
from pathlib import Path
from typing import Iterator, Optional

CONFIG_DIR_ENV_VAR = "PRIME_CONFIG_DIR"
CONFIG_DIR_NAME = ".prime"
CONFIG_FILE_NAME = "config.json"
TRUSTED_CONFIGS_FILE_NAME = "trusted_configs.json"


def global_config_dir() -> Path:
    """The per-user config directory, ~/.prime."""
    return Path.home() / CONFIG_DIR_NAME


def trusted_configs_file() -> Path:
    """The per-user registry of project-local configs approved via `prime config trust`."""
    return global_config_dir() / TRUSTED_CONFIGS_FILE_NAME


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


def _file_digest(path: Path) -> Optional[str]:
    try:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def load_trusted_configs() -> dict[str, str]:
    """Resolved config.json path -> content digest, for every approved local config."""
    try:
        data = json.loads(trusted_configs_file().read_text())
    except (OSError, ValueError):
        return {}
    trusted = data.get("trusted") if isinstance(data, dict) else None
    if not isinstance(trusted, dict):
        return {}
    return {str(path): str(digest) for path, digest in trusted.items()}


def is_trusted_local_config(config_file: Path) -> bool:
    """Whether the user approved this file *with its current content*.

    Ownership alone is not enough: a `.prime/config.json` committed to a
    repository the user clones is owned by the user, and could redirect an
    environment-provided PRIME_API_KEY to an attacker's base_url. So a
    discovered local config is only honored after `prime config trust`, and
    the approval is tied to a content digest — if the file changes it has to
    be trusted again. The SDK only reads the registry; the CLI maintains it.
    """
    if config_file.is_symlink() or config_file.parent.is_symlink():
        # Project files reached through a symlink are never trusted (the CLI
        # applies the same rule, and refuses to write through them).
        return False
    if not _owned_by_current_user(config_file):
        # A trusted path swapped for someone else's file (same bytes or not)
        # is not the file the user approved.
        return False
    digest = _file_digest(config_file)
    if digest is None:
        return False
    return load_trusted_configs().get(str(config_file.resolve())) == digest


def _candidate_local_config_files(start: Optional[Path] = None) -> Iterator[Path]:
    """Owned `.prime/config.json` files from `start` upward, nearest first.

    The walk stops at the home directory: `~/.prime` is the global config, and
    anything above home is not the user's project.
    """
    try:
        current = (start or Path.cwd()).resolve()
        home = Path.home().resolve()
    except OSError:
        return
    for directory in (current, *current.parents):
        if directory == home:
            break
        config_file = directory / CONFIG_DIR_NAME / CONFIG_FILE_NAME
        if config_file.is_file() and _owned_by_current_user(config_file):
            yield config_file


def discover_local_config(start: Optional[Path] = None) -> tuple[Optional[Path], list[Path]]:
    """The nearest *trusted* project-local config dir, plus untrusted files passed over.

    Untrusted candidates are skipped rather than stopping the search, so a
    file planted in a nested directory cannot mask the user's own trusted one
    further up.
    """
    untrusted: list[Path] = []
    for config_file in _candidate_local_config_files(start):
        if is_trusted_local_config(config_file):
            return config_file.parent, untrusted
        untrusted.append(config_file)
    return None, untrusted


def find_local_config_dir(start: Optional[Path] = None) -> Optional[Path]:
    """Find the nearest trusted project-local config directory, or None."""
    return discover_local_config(start)[0]


def resolve_config_dir() -> tuple[Path, str, list[Path]]:
    """Choose the config directory: PRIME_CONFIG_DIR > trusted project-local > ~/.prime.

    Returns the directory, its source ("env", "local", or "global"), and any
    untrusted project-local config files that were ignored on the way.
    """
    explicit = os.getenv(CONFIG_DIR_ENV_VAR)
    if explicit and explicit.strip():
        return Path(explicit.strip()).expanduser(), "env", []
    local, untrusted = discover_local_config()
    if local is not None:
        return local, "local", untrusted
    return global_config_dir(), "global", untrusted


UNTRUSTED_CONFIG_WARNING = "Ignoring untrusted project config"
_warned_untrusted: set[Path] = set()


def _warn_untrusted(untrusted: list[Path]) -> None:
    """Warn once per file per process that a local config was ignored."""
    for config_file in untrusted:
        if config_file in _warned_untrusted:
            continue
        _warned_untrusted.add(config_file)
        warnings.warn(
            f"{UNTRUSTED_CONFIG_WARNING} {config_file}; run "
            f"'prime config trust {config_file.parent.parent}' to use it, or set "
            f"{CONFIG_DIR_ENV_VAR}.",
            RuntimeWarning,
            stacklevel=2,
        )


class Config:
    """Minimal configuration class for SDK packages.

    Reads from a project-local or ~/.prime config.json and environment variables.
    """

    DEFAULT_BASE_URL: str = "https://api.primeintellect.ai"

    def __init__(self, config_dir: Optional[Path | str] = None) -> None:
        """Load config from `config_dir`, or from the resolved location when omitted.

        Resolution order: `PRIME_CONFIG_DIR`, then the nearest ancestor of the
        working directory holding a *trusted* `.prime/config.json` (see
        `discover_local_config`), then `~/.prime`. A project-local config is a
        complete replacement for the global one, never merged with it.
        """
        self.untrusted_local_configs: list[Path] = []
        if config_dir is not None:
            self.config_dir = Path(config_dir).expanduser()
            self.config_source = "explicit"
        else:
            self.config_dir, self.config_source, self.untrusted_local_configs = resolve_config_dir()
        self.config_file = self.config_dir / CONFIG_FILE_NAME
        self._load_config()
        _warn_untrusted(self.untrusted_local_configs)

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
