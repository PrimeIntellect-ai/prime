import json
import os
import re
from pathlib import Path
from typing import Optional

from prime_traces.core.config import Config as _TracesSdkConfig
from pydantic import BaseModel, ConfigDict

LOCAL_CONTEXT_FILE = Path(".prime") / "context.json"
_CONTEXT_NAME = re.compile(r"[a-zA-Z0-9_-]+")
_TEAM_KEYS = ("team_id", "team_name", "team_role")
# Keys a saved environment (context) file holds; writes to any other key always
# go to the global config file.
_ENVIRONMENT_KEYS = frozenset(
    {
        "api_key",
        "team_id",
        "team_name",
        "team_role",
        "user_id",
        "user_name",
        "base_url",
        "frontend_url",
        "inference_url",
        "traces_url",
    }
)


def find_local_context_file(start: Optional[Path] = None) -> Optional[Path]:
    """Return the nearest ``.prime/context.json`` at or above ``start`` (default: cwd).

    The walk stops at the home directory, whose ``.prime`` is the global config
    directory, and skips symlinks and files owned by another user.
    """
    try:
        current = (start or Path.cwd()).resolve()
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


def read_local_context(path: Path) -> dict:
    """Parse and validate a directory context file.

    It may select a saved ``context`` and/or pin a team (``team_id`` present,
    ``null`` meaning the personal account). It never holds credentials or URLs.
    """
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
    for key in _TEAM_KEYS:
        if data.get(key) is not None and not isinstance(data[key], str):
            raise ValueError(f"Invalid {path}: {key} must be a string or null")
    return data


def write_local_context(path: Path, data: dict) -> None:
    """Write a directory context file, removing it (and an empty ``.prime``) when empty."""
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError(f"Refusing to write {path}: it or its directory is a symlink")
    if not data:
        path.unlink(missing_ok=True)
        try:
            path.parent.rmdir()
        except OSError:
            pass
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


class ConfigModel(BaseModel):
    api_key: str = ""
    team_id: str | None = None
    team_name: str | None = None
    team_role: str | None = None
    user_id: str | None = None
    user_name: str | None = None
    base_url: str = "https://api.primeintellect.ai"
    frontend_url: str = "https://app.primeintellect.ai"
    inference_url: str = "https://api.pinference.ai/api/v1"
    traces_url: str | None = None
    ssh_key_path: str = str(Path.home() / ".ssh" / "id_rsa")
    current_environment: str = "production"
    share_resources_with_team: bool = False

    model_config = ConfigDict(populate_by_name=True)


class Config:
    DEFAULT_BASE_URL: str = "https://api.primeintellect.ai"
    DEFAULT_FRONTEND_URL: str = "https://app.primeintellect.ai"
    DEFAULT_INFERENCE_URL: str = "https://api.pinference.ai/api/v1"
    DEFAULT_TRACES_URL: str = _TracesSdkConfig.DEFAULT_TRACES_URL
    DEFAULT_SSH_KEY_PATH: str = str(Path.home() / ".ssh" / "id_rsa")

    def __init__(self, use_context: bool = True) -> None:
        """Load the global config, then any context selected for this invocation.

        Precedence (highest first): per-field env vars (PRIME_API_KEY, ...), a
        temporary ``--context``/PRIME_CONTEXT, the nearest directory context file
        (``.prime/context.json``), then ``~/.prime/config.json``. With
        ``use_context=False`` only the global config is loaded, so every write
        goes to it.
        """
        self.config_dir = Path.home() / ".prime"
        self.config_file = self.config_dir / "config.json"
        self.environments_dir = self.config_dir / "environments"
        # Directory context in effect, if any (never set alongside PRIME_CONTEXT).
        self.local_context_file: Optional[Path] = None
        self.local_context: dict = {}
        # Saved environment a directory context selects: credential writes go
        # to its file instead of the global config.
        self._profile_name: Optional[str] = None
        self._profile_overlay: dict = {}
        self._team_overlay: dict = {}
        self._ensure_config_dir()
        self._load_config()

        if not use_context:
            return

        # Check for PRIME_CONTEXT env var to temporarily override config
        context = os.getenv("PRIME_CONTEXT")
        if context:
            self.load_environment(context, persist=False)
            return

        local_file = find_local_context_file()
        if local_file is None:
            return
        local = read_local_context(local_file)
        self.local_context_file = local_file
        self.local_context = local
        pinned = local.get("context")
        if pinned:
            try:
                loaded = self.load_environment(pinned, persist=False)
            except (TypeError, AttributeError) as e:
                # e.g. a saved context with "frontend_url": null
                raise ValueError(
                    f"Invalid context '{pinned}' selected by {local_file}: {e}. "
                    f"Fix {self.environments_dir / pinned}.json or run 'prime config unpin'."
                ) from e
            if not loaded:
                raise ValueError(
                    f"Unknown context '{pinned}' selected by {local_file}. "
                    "Save it with 'prime config save', or run 'prime config unpin'."
                )
            if pinned.casefold() != "production":
                self._profile_name = pinned
        if "team_id" in local:
            team_id = local.get("team_id") or None
            self._team_overlay = {
                "team_id": team_id,
                "team_name": local.get("team_name") if team_id else None,
                "team_role": local.get("team_role") if team_id else None,
            }
            self._refresh()

    @property
    def context_override(self) -> str | None:
        """Return the temporary context selected for this invocation, if any."""
        context = os.getenv("PRIME_CONTEXT")
        return context if context else None

    @staticmethod
    def _strip_api_v1(url: str) -> str:
        # make base_url consistent even if user passed a /api/v1 variant
        return url.rstrip("/").removesuffix("/api/v1")

    def _ensure_config_dir(self) -> None:
        """Create config directory if it doesn't exist"""
        self.config_dir.mkdir(exist_ok=True)
        self.environments_dir.mkdir(exist_ok=True)
        if not self.config_file.exists():
            self._save_config(
                ConfigModel(
                    api_key="",
                    team_id=None,
                    user_id=None,
                    base_url=self.DEFAULT_BASE_URL,
                    frontend_url=self.DEFAULT_FRONTEND_URL,
                    inference_url=self.DEFAULT_INFERENCE_URL,
                    ssh_key_path=self.DEFAULT_SSH_KEY_PATH,
                    current_environment="production",
                ).model_dump()
            )

    def _load_config(self) -> None:
        """Load configuration from file"""
        if self.config_file.exists():
            config_data = json.loads(self.config_file.read_text())
            self._stored = ConfigModel(**config_data).model_dump()
        else:
            self._stored = {}
        self._refresh()

    def _refresh(self) -> None:
        """Recompute the effective config: global file, then context, then team pin."""
        self.config = {**self._stored, **self._profile_overlay, **self._team_overlay}

    def _save_config(self, config: dict) -> None:
        """Save configuration to file"""
        self.config_file.write_text(json.dumps(config, indent=2))
        self._stored = dict(config)
        self._refresh()

    def _update(self, **fields: object) -> None:
        """Persist fields to the selected context's file, or else the global config."""
        if self._profile_name is not None and _ENVIRONMENT_KEYS.issuperset(fields):
            env_file = self._environment_file(self._profile_name)
            env_config = json.loads(env_file.read_text())
            if not isinstance(env_config, dict):
                raise ValueError(f"Invalid configuration in {env_file}")
            env_config.update(fields)
            env_file.write_text(json.dumps(env_config, indent=2))
            self._profile_overlay.update(fields)
            self._refresh()
            return
        self._save_config({**self._stored, **fields})

    def _environment_file(self, name: str) -> Path:
        return self.environments_dir / f"{self._sanitize_environment_name(name)}.json"

    @property
    def writes_context(self) -> Optional[str]:
        """Saved context that credential writes go to, when a directory selects one."""
        return self._profile_name

    @property
    def team_pinned(self) -> bool:
        """Whether the directory context file pins the team."""
        return bool(self._team_overlay)

    @property
    def api_key(self) -> str:
        """Get API key with precedence: env > file > empty."""
        return os.getenv("PRIME_API_KEY") or self.config.get("api_key", "")

    def set_api_key(self, value: str) -> None:
        """Set API key in config file"""
        self._update(api_key=value)

    @property
    def team_id(self) -> Optional[str]:
        """Get team ID with precedence: env > file > None."""
        env_val = os.getenv("PRIME_TEAM_ID")
        if env_val is not None and env_val.strip():
            return env_val
        return self.config.get("team_id") or None

    @property
    def team_id_from_env(self) -> bool:
        """Check if team ID is set via environment variable."""
        env_val = os.getenv("PRIME_TEAM_ID")
        return bool(env_val and env_val.strip())

    @property
    def team_name(self) -> Optional[str]:
        """Get team name from config file (only valid if team_id not from env)."""
        return self.config.get("team_name") or None

    @property
    def team_role(self) -> Optional[str]:
        """Get team role from config file (only valid if team_id not from env)."""
        return self.config.get("team_role") or None

    def set_team(
        self, value: str | None, team_name: str | None = None, team_role: str | None = None
    ) -> None:
        """Set team ID, name, and role in config file."""
        self._update(
            team_id=value or None,
            team_name=team_name if value else None,
            team_role=team_role if value else None,
        )

    @property
    def user_id(self) -> Optional[str]:
        """Get user ID with precedence: env > file > None."""
        user_id = os.getenv("PRIME_USER_ID")
        if user_id is not None:
            return user_id
        return self.config.get("user_id") or None

    @property
    def user_id_from_env(self) -> bool:
        """Check if user ID is set via environment variable."""
        return os.getenv("PRIME_USER_ID") is not None

    @property
    def user_name(self) -> Optional[str]:
        """Get user name from config file (only valid if user_id not from env)."""
        return self.config.get("user_name") or None

    def set_user_id(self, value: str | None, user_name: str | None = None) -> None:
        """Set user ID and display name in config file."""
        self._update(user_id=value if value else None, user_name=user_name if value else None)

    @property
    def base_url(self) -> str:
        """Get API base URL with precedence: env > file > default."""
        env_val = os.getenv("PRIME_API_BASE_URL") or os.getenv("PRIME_BASE_URL")
        if env_val:
            return self._strip_api_v1(env_val)
        return self._strip_api_v1(self.config.get("base_url", self.DEFAULT_BASE_URL))

    def set_base_url(self, value: str) -> None:
        """Set API base URL in config file"""
        self._update(base_url=self._strip_api_v1(value))

    @property
    def frontend_url(self) -> str:
        """Get frontend URL with precedence: env > file > default."""
        env_val = os.getenv("PRIME_FRONTEND_URL")
        if env_val:
            return env_val.rstrip("/")
        return (self.config.get("frontend_url", self.DEFAULT_FRONTEND_URL)).rstrip("/")

    def set_frontend_url(self, value: str) -> None:
        """Set frontend URL in config file"""
        self._update(frontend_url=value.rstrip("/"))

    @property
    def inference_url(self) -> str:
        """Get inference URL with precedence: env > file > default."""
        env_val = os.getenv("PRIME_INFERENCE_URL")
        if env_val:
            return env_val.rstrip("/")
        return self.config.get("inference_url", self.DEFAULT_INFERENCE_URL).rstrip("/")

    def set_inference_url(self, value: str) -> None:
        """Set inference URL in config file"""
        self._update(inference_url=value.rstrip("/"))

    def _configured_traces_url(self) -> str | None:
        """The explicitly configured traces URL (env > file), or None when unset."""
        env_val = os.getenv("PRIME_TRACES_URL")
        if env_val:
            return self._strip_api_v1(env_val)
        return self._stored_traces_url()

    def _stored_traces_url(self) -> str | None:
        """The traces URL stored in the config file, excluding env overrides."""
        file_val = self.config.get("traces_url")
        if file_val:
            return self._strip_api_v1(str(file_val))
        return None

    @property
    def traces_url(self) -> str:
        """Get Prime Traces service URL with precedence: env > file > DEFAULT_TRACES_URL.

        base_url is never consulted: a platform override says nothing about
        where the traces service lives, and the platform API 404s /api/v1/traces.
        """
        return self._configured_traces_url() or self.DEFAULT_TRACES_URL

    def set_traces_url(self, value: str) -> None:
        """Set Prime Traces service URL in config file; empty clears the override."""
        self._update(traces_url=self._strip_api_v1(value) if value else None)

    def set_traces_url_for_active_environment(self, value: str) -> None:
        """Persist only the traces URL for the command's selected environment."""
        traces_url = self._strip_api_v1(value) if value else None
        if self._profile_name is not None:
            # A directory context selects this environment: only its file changes.
            self._update(traces_url=traces_url)
            return
        selected_environment = self.current_environment
        context_override = os.getenv("PRIME_CONTEXT")

        root_config = json.loads(self.config_file.read_text())
        if not isinstance(root_config, dict):
            raise ValueError(f"Invalid configuration in {self.config_file}")
        root_environment = str(root_config.get("current_environment", "production"))

        if (
            context_override
            and selected_environment == "production"
            and root_environment.casefold() != "production"
        ):
            raise ValueError(
                "Cannot persist production traces settings while another environment is active; "
                "run 'prime config use production' first"
            )

        update_root = (
            not context_override or root_environment.casefold() == selected_environment.casefold()
        )

        if update_root:
            root_config["traces_url"] = traces_url
            self.config_file.write_text(json.dumps(root_config, indent=2))
            self._stored["traces_url"] = traces_url

        if selected_environment != "production":
            sanitized = self._sanitize_environment_name(selected_environment)
            env_file = self.environments_dir / f"{sanitized}.json"
            if env_file.exists():
                env_config = json.loads(env_file.read_text())
                if not isinstance(env_config, dict):
                    raise ValueError(f"Invalid configuration in {env_file}")
                env_config["traces_url"] = traces_url
                env_file.write_text(json.dumps(env_config, indent=2))

        if self._profile_overlay:
            self._profile_overlay["traces_url"] = traces_url
        self._refresh()

    @property
    def ssh_key_path(self) -> str:
        """Get SSH private key path with precedence: env > file > default."""
        env_val = os.getenv("PRIME_SSH_KEY_PATH")
        if env_val:
            return str(Path(env_val).expanduser())
        return self.config.get("ssh_key_path", self.DEFAULT_SSH_KEY_PATH)

    def set_ssh_key_path(self, value: str) -> None:
        """Set SSH private key path in config file"""
        self._update(ssh_key_path=str(Path(value).expanduser().resolve()))

    @property
    def share_resources_with_team(self) -> bool:
        """Get share_resources_with_team setting from config file."""
        val = self.config.get("share_resources_with_team", False)
        if isinstance(val, str):
            return val.lower() == "true"
        return bool(val)

    def set_share_resources_with_team(self, value: bool) -> None:
        """Set share_resources_with_team in config file"""
        self._update(share_resources_with_team=value)

    @property
    def current_environment(self) -> str:
        """Get current environment name"""
        current_env: str = self.config.get("current_environment", "production")
        return current_env

    def set_current_environment(self, value: str) -> None:
        """Set current environment name"""
        self._update(current_environment=value)

    def _sanitize_environment_name(self, name: str) -> str:
        """Sanitize environment name to prevent path traversal"""
        # Only allow alphanumeric characters, hyphens, and underscores
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "", name)
        if not sanitized or sanitized != name:
            raise ValueError(
                f"Invalid environment name: {name!r}. "
                "Only alphanumeric characters, hyphens, and underscores are allowed."
            )
        return sanitized

    def view(self) -> dict:
        """Get all config values"""
        return {
            "api_key": self.api_key,
            "team_id": self.team_id,
            "team_name": self.team_name,
            "team_role": self.team_role,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "base_url": self.base_url,
            "frontend_url": self.frontend_url,
            "inference_url": self.inference_url,
            "traces_url": self.traces_url,
            "ssh_key_path": self.ssh_key_path,
            "current_environment": self.current_environment,
            "share_resources_with_team": self.share_resources_with_team,
        }

    def save_environment(self, name: str) -> None:
        """Save current configuration as a named environment"""
        if name.lower() == "production":
            raise ValueError("Cannot save custom environment with reserved name 'production'")

        sanitized_name = self._sanitize_environment_name(name)
        env_file = self.environments_dir / f"{sanitized_name}.json"
        env_config = {
            "api_key": self.api_key,
            "team_id": self.team_id,
            "team_name": None if self.team_id_from_env else self.team_name,
            "team_role": None if self.team_id_from_env else self.team_role,
            "user_id": self.user_id,
            "user_name": None if self.user_id_from_env else self.user_name,
            "base_url": self.base_url,
            "frontend_url": self.frontend_url,
            "inference_url": self.inference_url,
            "traces_url": self._configured_traces_url(),
        }
        env_file.write_text(json.dumps(env_config, indent=2))

    def delete_environment(self, name: str) -> None:
        """Delete a saved environment configuration."""
        if name.lower() == "production":
            raise ValueError("Cannot delete built-in environment 'production'")

        sanitized_name = self._sanitize_environment_name(name)
        active = {
            self.current_environment.casefold(),
            str(self._stored.get("current_environment", "production")).casefold(),
        }
        if sanitized_name.casefold() in active:
            raise ValueError(
                f"Cannot delete currently active environment '{name}'. "
                "Use 'prime config use production' or another saved environment first."
            )

        env_file = self.environments_dir / f"{sanitized_name}.json"
        if not env_file.exists():
            raise ValueError(f"Unknown environment: {name}")

        env_file.unlink()

    def load_environment(self, name: str, persist: bool = True) -> bool:
        """Load a named environment configuration.

        Args:
            name: The environment name to load
            persist: If True, save changes to disk. If False, only update in-memory config.

        Returns:
            True if the environment was loaded successfully, False otherwise.
        """
        if persist and self._profile_name is not None:
            # Persistent setters would write into the pinned context's file.
            raise ValueError(
                f"Cannot switch environments while {self.local_context_file} selects "
                f"context '{self._profile_name}'; use Config(use_context=False) to "
                "change the global environment."
            )
        if name.lower() == "production":
            # Built-in production environment
            if persist:
                self.set_base_url(self.DEFAULT_BASE_URL)
                self.set_frontend_url(self.DEFAULT_FRONTEND_URL)
                self.set_inference_url(self.DEFAULT_INFERENCE_URL)
                self.set_traces_url("")  # No override: follow base_url
                self.set_team(None)  # Production defaults to personal account
                self.set_current_environment("production")
            else:
                self._profile_overlay = {
                    "base_url": self.DEFAULT_BASE_URL,
                    "frontend_url": self.DEFAULT_FRONTEND_URL,
                    "inference_url": self.DEFAULT_INFERENCE_URL,
                    "traces_url": None,
                    "team_id": None,
                    "team_name": None,
                    "team_role": None,
                    "current_environment": "production",
                }
                self._refresh()
            return True

        try:
            sanitized_name = self._sanitize_environment_name(name)
            env_file = self.environments_dir / f"{sanitized_name}.json"
            if env_file.exists():
                try:
                    env_config = json.loads(env_file.read_text())
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in environment file {sanitized_name}.json: {e}")
                except (OSError, UnicodeDecodeError) as e:
                    raise ValueError(
                        f"Cannot read environment file {sanitized_name}.json: {e}"
                    ) from e

                if not isinstance(env_config, dict):
                    raise ValueError(
                        f"Invalid environment file {sanitized_name}.json: expected a JSON object"
                    )

                if persist:
                    if "api_key" in env_config:
                        self.set_api_key(env_config["api_key"])
                    # Set team_id, team_name, and team_role from environment
                    self.set_team(
                        env_config.get("team_id", None),
                        team_name=env_config.get("team_name", None),
                        team_role=env_config.get("team_role", None),
                    )
                    # Set user_id and user_name from environment
                    self.set_user_id(
                        env_config.get("user_id", None),
                        user_name=env_config.get("user_name", None),
                    )
                    self.set_base_url(env_config.get("base_url", self.DEFAULT_BASE_URL))
                    self.set_frontend_url(env_config.get("frontend_url", self.DEFAULT_FRONTEND_URL))
                    self.set_inference_url(
                        env_config.get("inference_url", self.DEFAULT_INFERENCE_URL)
                    )
                    self.set_traces_url(env_config.get("traces_url") or "")
                    self.set_current_environment(name)
                else:
                    # In-memory only - don't persist to disk
                    overlay: dict = {}
                    if "api_key" in env_config:
                        overlay["api_key"] = env_config["api_key"]
                    overlay["team_id"] = env_config.get("team_id", None)
                    overlay["team_name"] = env_config.get("team_name", None)
                    overlay["team_role"] = env_config.get("team_role", None)
                    overlay["user_id"] = env_config.get("user_id", None)
                    overlay["user_name"] = env_config.get("user_name", None)
                    # Normalize URLs the same way set_* methods do
                    base_url = env_config.get("base_url", self.DEFAULT_BASE_URL)
                    overlay["base_url"] = self._strip_api_v1(base_url)
                    frontend_url = env_config.get("frontend_url", self.DEFAULT_FRONTEND_URL)
                    overlay["frontend_url"] = frontend_url.rstrip("/")
                    inference_url = env_config.get("inference_url", self.DEFAULT_INFERENCE_URL)
                    overlay["inference_url"] = inference_url.rstrip("/")
                    traces_url = env_config.get("traces_url")
                    overlay["traces_url"] = self._strip_api_v1(traces_url) if traces_url else None
                    overlay["current_environment"] = name
                    self._profile_overlay = overlay
                    self._refresh()
                return True
        except ValueError:
            # Re-raise sanitization errors
            raise
        return False

    def update_current_environment_file(self) -> None:
        """Mirror the global config into the saved file of its current environment.

        Uses the stored values only: env var overrides, a temporary context and a
        directory context never leak onto disk. A no-op while a directory context
        selects an environment, since writes already went to that file.
        """
        if self._profile_name is not None:
            return
        stored = self._stored
        current = str(stored.get("current_environment", "production"))
        if current == "production":
            # Only update custom environments, not the built-in production
            return
        try:
            env_file = self._environment_file(current)
        except ValueError:
            # Skip updating if environment name is invalid
            return
        if env_file.exists():
            traces_url = stored.get("traces_url")
            env_config = {
                "api_key": stored.get("api_key", ""),
                "team_id": stored.get("team_id"),
                "team_name": stored.get("team_name"),
                "team_role": stored.get("team_role"),
                "user_id": stored.get("user_id"),
                "user_name": stored.get("user_name"),
                "base_url": self._strip_api_v1(stored.get("base_url", self.DEFAULT_BASE_URL)),
                "frontend_url": stored.get("frontend_url", self.DEFAULT_FRONTEND_URL).rstrip("/"),
                "inference_url": stored.get("inference_url", self.DEFAULT_INFERENCE_URL).rstrip(
                    "/"
                ),
                "traces_url": self._strip_api_v1(traces_url) if traces_url else None,
            }
            env_file.write_text(json.dumps(env_config, indent=2))

    def list_environments(self) -> list[str]:
        """List all saved environment names"""
        environments = ["production"]  # Built-in environment
        if self.environments_dir.exists():
            for env_file in self.environments_dir.glob("*.json"):
                env_name = env_file.stem
                # Skip any files that would conflict with built-in environments
                if env_name.lower() != "production":
                    environments.append(env_name)
        return environments
