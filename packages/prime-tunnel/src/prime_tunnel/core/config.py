import json
import os
import re
from pathlib import Path
from typing import Optional

LOCAL_CONTEXT_FILE = Path(".prime") / "context.json"
_CONTEXT_NAME = re.compile(r"[a-zA-Z0-9_-]+")


def find_local_context_file() -> Optional[Path]:
    """Nearest ``.prime/context.json`` at or above the working directory.

    Written by ``prime switch --local`` / ``prime config use --local``. The walk
    stops at the home directory, whose ``.prime`` is the global config directory,
    and skips symlinks and files owned by another user. Mirrors the Prime CLI's
    resolution.
    """
    try:
        current = Path.cwd().resolve()
        home = Path.home().resolve()
    except (OSError, RuntimeError):
        return None
    getuid = getattr(os, "getuid", None)
    for directory in (current, *current.parents):
        if directory == home:
            return None
        candidate = directory / LOCAL_CONTEXT_FILE
        try:
            # A symlinked pin could point writes (e.g. `prime switch`) at another
            # file such as ~/.prime/config.json, so only plain files count.
            if candidate.parent.is_symlink() or candidate.is_symlink():
                continue
            if candidate.is_file() and (getuid is None or candidate.stat().st_uid == getuid()):
                return candidate
        except OSError:
            continue
    return None


def _read_local_context(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"Cannot read {path}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"Invalid {path}: expected a JSON object")
    context = data.get("context")
    if context is not None and (
        not isinstance(context, str) or _CONTEXT_NAME.fullmatch(context) is None
    ):
        raise ValueError(f"Invalid {path}: bad context name {context!r}")
    for key in ("team_id", "team_name", "team_role"):
        if data.get(key) is not None and not isinstance(data[key], str):
            raise ValueError(f"Invalid {path}: {key} must be a string or null")
    return data


class Config:
    """Minimal configuration class for Prime Tunnel SDK.

    Reads ~/.prime/config.json, the context selected by PRIME_CONTEXT or the
    nearest .prime/context.json, and environment variables (highest precedence).
    This is a simplified version that doesn't write configs.
    """

    DEFAULT_BASE_URL: str = "https://api.primeintellect.ai"

    def __init__(self) -> None:
        self.config_dir = Path.home() / ".prime"
        self.config_file = self.config_dir / "config.json"
        self.environments_dir = self.config_dir / "environments"
        self.local_context_file: Optional[Path] = None
        self._context_error: Optional[ValueError] = None
        self._load_config()
        try:
            self._load_context()
        except ValueError as e:
            if self.local_context_file is None:
                raise  # an explicit PRIME_CONTEXT fails immediately
            # A broken directory pin is raised on first read instead: a caller
            # passing every value explicitly never reads the config, so it must
            # not be taken down, while any value the pin would supply fails loudly.
            self._context_error = e

    @property
    def config(self) -> dict:
        """Resolved config values; raises if the selected context is broken."""
        if self._context_error is not None:
            raise self._context_error
        return self._config

    @config.setter
    def config(self, value: dict) -> None:
        self._config = value

    def _load_config(self) -> None:
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                config_data = json.loads(self.config_file.read_text())
                self.config = config_data if isinstance(config_data, dict) else {}
            except (json.JSONDecodeError, IOError):
                self.config = {}
        else:
            self.config = {}

    def _load_context(self) -> None:
        """Overlay the context selected for this process or directory.

        PRIME_CONTEXT (set by ``prime --context``) wins. Otherwise the nearest
        ``.prime/context.json`` may select a saved context and/or pin a team
        (``team_id`` present; null means the personal account).
        """
        context = os.getenv("PRIME_CONTEXT")
        local: dict = {}
        source = "PRIME_CONTEXT"
        if not context:
            self.local_context_file = find_local_context_file()
            if self.local_context_file is None:
                return
            local = _read_local_context(self.local_context_file)
            context = local.get("context")
            source = str(self.local_context_file)
        if context:
            self._apply_context(context, source)
        if "team_id" in local:
            team_id = local.get("team_id") or None
            self.config.update(
                {
                    "team_id": team_id,
                    "team_name": local.get("team_name") if team_id else None,
                    "team_role": local.get("team_role") if team_id else None,
                }
            )

    def _apply_context(self, context: str, source: str) -> None:
        if context.casefold() == "production":
            self.config.update(
                {
                    "base_url": self.DEFAULT_BASE_URL,
                    "frontend_url": None,
                    "inference_url": None,
                    "traces_url": None,
                    "team_id": None,
                    "team_name": None,
                    "team_role": None,
                }
            )
            return

        if _CONTEXT_NAME.fullmatch(context) is None:
            raise ValueError(f"Invalid context name: {context!r}")

        environment_file = self.environments_dir / f"{context}.json"
        if not environment_file.exists():
            raise ValueError(
                f"Context file not found: {environment_file} (context '{context}' from {source})"
            )
        try:
            environment_config = json.loads(environment_file.read_text())
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to load context '{context}': {e}") from e
        if not isinstance(environment_config, dict):
            raise ValueError(f"Invalid context '{context}': expected a JSON object")
        self.config.update(environment_config)

    @staticmethod
    def _strip_api_v1(url: str) -> str:
        return url.rstrip("/").removesuffix("/api/v1")

    @property
    def api_key(self) -> str:
        """Get API key with precedence: env > file > empty."""
        return os.getenv("PRIME_API_KEY") or self.config.get("api_key", "")

    @property
    def team_id(self) -> Optional[str]:
        """Get team ID with precedence: env > file > None.

        An explicitly empty PRIME_TEAM_ID means personal scope: it must not
        fall back to the file's team, and must never leak onto the wire as
        ``teamId: ""``.
        """
        team_id = os.getenv("PRIME_TEAM_ID")
        if team_id is not None:
            return team_id or None
        return self.config.get("team_id") or None

    @property
    def user_id(self) -> Optional[str]:
        """Get user ID with precedence: env > file > None."""
        user_id = os.getenv("PRIME_USER_ID")
        if user_id is not None:
            return user_id
        return self.config.get("user_id") or None

    @property
    def base_url(self) -> str:
        """Get API base URL with precedence: env > file > default."""
        env_val = os.getenv("PRIME_API_BASE_URL") or os.getenv("PRIME_BASE_URL")
        if env_val:
            return self._strip_api_v1(env_val)
        return self._strip_api_v1(self.config.get("base_url", self.DEFAULT_BASE_URL))

    @property
    def bin_dir(self) -> Path:
        """Directory for binary files (frpc)."""
        path = self.config_dir / "bin"
        path.mkdir(parents=True, exist_ok=True)
        return path
