import hashlib
import json
import os
import re
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Iterator, Optional

from pydantic import BaseModel, ConfigDict

CONFIG_DIR_ENV_VAR = "PRIME_CONFIG_DIR"
CONFIG_DIR_NAME = ".prime"
CONFIG_FILE_NAME = "config.json"
ENVIRONMENTS_DIR_NAME = "environments"
TRUSTED_CONFIGS_FILE_NAME = "trusted_configs.json"


def global_config_dir() -> Path:
    """The per-user config directory, ~/.prime."""
    return Path.home() / CONFIG_DIR_NAME


def is_global_config_dir(path: Path) -> bool:
    """Whether `path` is ~/.prime, comparing physical paths.

    `Path.cwd()` is always the physical path while `Path.home()` may go through
    a symlink (a symlinked home, macOS /var -> /private/var), so an unresolved
    comparison would call the global config "project-local".
    """
    try:
        return path.resolve() == global_config_dir().resolve()
    except OSError:
        return False


def trusted_configs_file() -> Path:
    """The per-user registry of project-local configs the user has approved."""
    return global_config_dir() / TRUSTED_CONFIGS_FILE_NAME


def _owned_by_current_user(path: Path) -> bool:
    """True when `path` (following symlinks) is owned by the running user.

    Project-local configs are picked up by walking parent directories, so on a
    shared machine anyone who can write to an ancestor could plant one that
    points the CLI at their account. Refusing files we don't own closes that.
    Windows has no uid model; the check is skipped there.
    """
    getuid = getattr(os, "getuid", None)
    if getuid is None:
        return True
    try:
        return path.stat().st_uid == getuid()
    except OSError:
        return False


def _project_config_dir_of(path: Path) -> Path:
    """The `.prime` directory a config.json or environments/<name>.json belongs to."""
    if path.parent.name == ENVIRONMENTS_DIR_NAME:
        return path.parent.parent
    return path.parent


def involves_symlink(path: Path) -> bool:
    """Whether `path`, its parent directory, or its `.prime` directory is a symlink.

    Project-local files are never trusted or written through symlinks: a
    cloned repository can ship `.prime`, `.prime/environments`, or the file
    itself as a link that redirects a credential write outside the ignored
    tree (or onto an existing file). Checking only the leaf is not enough.
    """
    config_dir = _project_config_dir_of(path)
    candidates = {path, path.parent, config_dir}
    return any(candidate.is_symlink() for candidate in candidates)


def is_owned_by_current_user(path: Path) -> bool:
    """Public form of the ownership check used by discovery and the --local guard."""
    return _owned_by_current_user(path)


def _file_digest(path: Path) -> str | None:
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


def _save_trusted_configs(trusted: dict[str, str]) -> None:
    global_config_dir().mkdir(parents=True, exist_ok=True)
    _write_private_json(
        trusted_configs_file(), {"version": 1, "trusted": trusted}, allow_symlink=True
    )


def is_trusted_local_file(path: Path) -> bool:
    """Whether the user approved this file *with its current content*.

    Ownership alone is not enough: a `.prime/config.json` committed to a
    repository the user clones is owned by the user, and could redirect an
    environment-provided PRIME_API_KEY to an attacker's base_url. So a
    discovered local config is only honored after `prime config trust`, and
    the approval is tied to a content digest — if the file changes (e.g. via
    `git pull`) it has to be trusted again. Files the CLI writes itself
    (`prime login --local`, `prime config set-api-key --local`) are trusted
    as part of the write.

    The same rule covers the named environment files next to a local config
    (`.prime/environments/<name>.json`): `PRIME_CONTEXT` / `prime config use`
    load URLs from them, so they are part of the same trust boundary.
    """
    if involves_symlink(path):
        # Never trusted, so never discovered or loaded — and therefore never
        # written to, which is the property that matters (see involves_symlink).
        return False
    if not _owned_by_current_user(path):
        # A trusted path swapped for someone else's file (same bytes or not)
        # is not the file the user approved.
        return False
    digest = _file_digest(path)
    if digest is None:
        return False
    return load_trusted_configs().get(str(path.resolve())) == digest


is_trusted_local_config = is_trusted_local_file


def _local_environment_files(config_file: Path) -> list[Path]:
    """The named environment files that belong to a local config."""
    environments_dir = config_file.parent / ENVIRONMENTS_DIR_NAME
    if not environments_dir.is_dir() or environments_dir.is_symlink():
        return []
    # Symlinked environment files are never written to and never trusted.
    return sorted(
        p for p in environments_dir.glob("*.json") if p.is_file() and not involves_symlink(p)
    )


def trust_local_file(path: Path) -> Path:
    """Approve a single file with its current content; returns its resolved path."""
    if involves_symlink(path):
        raise ValueError(
            f"Refusing to trust {path}: it, its directory, or its .prime directory is a symlink"
        )
    path = path.resolve()
    digest = _file_digest(path)
    if digest is None:
        raise ValueError(f"Cannot read {path}")
    if not _owned_by_current_user(path):
        raise ValueError(f"Refusing to trust {path}: not owned by the current user")
    trusted = load_trusted_configs()
    trusted[str(path)] = digest
    _save_trusted_configs(trusted)
    return path


def trust_local_config(config_file: Path) -> Path:
    """Approve `config_file` and its environment files; returns the resolved config path."""
    resolved = trust_local_file(config_file)
    for env_file in _local_environment_files(resolved):
        if _owned_by_current_user(env_file):
            trust_local_file(env_file)
    return resolved


def forget_trusted_file(path: Path) -> None:
    """Drop a single file's registry entry (e.g. after deleting it)."""
    trusted = load_trusted_configs()
    if trusted.pop(str(path.resolve()), None) is not None:
        _save_trusted_configs(trusted)


def untrust_local_config(config_file: Path) -> bool:
    """Withdraw approval for the config and its environment files.

    Returns whether the config file was trusted before.
    """
    resolved = config_file.resolve()
    trusted = load_trusted_configs()
    removed = trusted.pop(str(resolved), None) is not None
    env_prefix = str(resolved.parent / "environments") + os.sep
    stale = [key for key in trusted if key.startswith(env_prefix)]
    for key in stale:
        del trusted[key]
    if removed or stale:
        _save_trusted_configs(trusted)
    return removed


def _candidate_local_config_files(start: Path | None = None) -> Iterator[Path]:
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


def discover_local_config(start: Path | None = None) -> tuple[Path | None, list[Path]]:
    """The nearest *trusted* project-local config dir, plus untrusted files passed over.

    Untrusted candidates are skipped rather than stopping the search, so a
    file planted in a nested directory cannot mask the user's own trusted one
    further up.
    """
    untrusted: list[Path] = []
    for config_file in _candidate_local_config_files(start):
        if is_trusted_local_file(config_file):
            return config_file.parent, untrusted
        untrusted.append(config_file)
    return None, untrusted


def find_local_config_dir(start: Path | None = None) -> Path | None:
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

# The SDKs bundled into the CLI resolve the same config and each raise a
# RuntimeWarning for an ignored local file; the CLI prints one line instead.
# This module loads before any command module constructs an SDK client.
warnings.filterwarnings("ignore", message=UNTRUSTED_CONFIG_WARNING)


def _warn_untrusted(untrusted: list[Path], kind: str = "project config") -> None:
    """Tell the user (once per file per process) that a local file was ignored."""
    for path in untrusted:
        if path in _warned_untrusted:
            continue
        _warned_untrusted.add(path)
        project = path.parent.parent if kind == "project config" else path.parent.parent.parent
        sys.stderr.write(
            f"Ignoring untrusted {kind} {path}; run 'prime config trust {project}' to use it.\n"
        )


def _refuse_unsafe_write(path: Path, *, allow_symlink: bool) -> None:
    """Refuse to write into a file the current user doesn't own, or through a symlink.

    On a shared machine someone with write access to the directory could swap
    the file for their own; O_TRUNC would then hand them the secret. A symlink
    is worse: a cloned repository can ship `.prime/config.json -> ../leak.json`
    (dangling, so the path looks absent) and the write would create a
    credential file outside the ignored `.prime/` directory — or clobber
    whatever an existing link points at. Project-local config dirs therefore
    never write through symlinks; the global `~/.prime` may, for users who
    keep dotfiles symlinked, as long as link and target are theirs and the
    target exists. A brand-new path is fine. Windows has no uid model; the
    ownership part is skipped there.
    """
    if not allow_symlink and involves_symlink(path):
        raise PermissionError(
            f"Refusing to write {path}: it, its directory, or its .prime directory is a symlink"
        )
    try:
        stats = [path.lstat()]
    except FileNotFoundError:
        return
    if path.is_symlink():
        try:
            stats.append(path.stat())
        except FileNotFoundError:
            raise PermissionError(f"Refusing to write through dangling symlink {path}") from None
    getuid = getattr(os, "getuid", None)
    if getuid is not None and any(st.st_uid != getuid() for st in stats):
        raise PermissionError(f"Refusing to write {path}: it is not owned by the current user")


_warned_skipped_writes: set[Path] = set()


def _warn_skipped_environment_write(env_file: Path) -> None:
    """Tell the user (once per file per process) an environment file was left alone."""
    if env_file in _warned_skipped_writes:
        return
    _warned_skipped_writes.add(env_file)
    sys.stderr.write(
        f"Not updating untrusted environment file {env_file}; review it and run "
        f"'prime config trust {env_file.parent.parent.parent}' first.\n"
    )


def _write_private_json(path: Path, data: dict, *, allow_symlink: bool = False) -> None:
    """Write JSON readable only by the owner, atomically; these files hold API keys.

    The content goes to a fresh 0600 temporary file in the same directory
    which is then renamed over the target, so:
    - a legacy permissive file (say 0644) never holds the new secret, not
      even between write and chmod — it is replaced, not rewritten;
    - readers (the SDKs load the trust registry on every `Config()`) see
      either the old or the new content, never an empty or partial file.
    A user-owned symlink in the global config dir is honored by replacing
    its target; project-local symlinks were refused before we get here.
    """
    _refuse_unsafe_write(path, allow_symlink=allow_symlink)
    target = path.resolve() if path.is_symlink() else path
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(data, indent=2))
        os.chmod(tmp_name, 0o600)  # mkstemp already uses 0600; be explicit
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


class ConfigModel(BaseModel):
    api_key: str = ""
    team_id: str | None = None
    team_name: str | None = None
    team_role: str | None = None
    user_id: str | None = None
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
    DEFAULT_SSH_KEY_PATH: str = str(Path.home() / ".ssh" / "id_rsa")

    def __init__(self, config_dir: Path | str | None = None, *, create: bool = True) -> None:
        """Load config from `config_dir`, or from the resolved location when omitted.

        Resolution order: `PRIME_CONFIG_DIR`, then the nearest ancestor of the
        working directory holding a *trusted* `.prime/config.json` (see
        `discover_local_config`), then `~/.prime`. A project-local config is a
        complete replacement for the global one, never merged with it.

        With `create=False` nothing is written until the first `set_*` call, so
        a flow that may still fail (e.g. login) does not leave a default file
        behind that would shadow the global config.
        """
        self.untrusted_local_configs: list[Path] = []
        if config_dir is not None:
            self.config_dir = Path(config_dir).expanduser()
            self.config_source = "explicit"
        else:
            self.config_dir, self.config_source, self.untrusted_local_configs = resolve_config_dir()
        self.config_file = self.config_dir / CONFIG_FILE_NAME
        self.environments_dir = self.config_dir / "environments"
        if create:
            self._ensure_config_dir()
        self._load_config()
        _warn_untrusted(self.untrusted_local_configs)

        # Check for PRIME_CONTEXT env var to temporarily override config
        context = os.getenv("PRIME_CONTEXT")
        if context:
            self.load_environment(context, persist=False)

    @classmethod
    def local(cls, directory: Path | str | None = None, *, create: bool = True) -> "Config":
        """A Config stored in `<directory>/.prime` (default: the working directory).

        Writing to it (or constructing with `create=True`) creates the file and
        trusts it, so subsequent `Config()` calls from anywhere under
        `directory` discover it.
        """
        root = Path(directory) if directory is not None else Path.cwd()
        return cls(config_dir=root / CONFIG_DIR_NAME, create=create)

    def adopt_urls(self, other: "Config") -> None:
        """Copy the stored service URLs (not credentials) from another config.

        Used when creating a project-local config so it keeps pointing at the
        same deployment the user already had configured (e.g. a dev base_url)
        instead of silently resetting to production. In-memory only; persisted
        by the next `set_*` call.
        """
        for key in ("base_url", "frontend_url", "inference_url", "traces_url"):
            value = other.config.get(key)
            if value:
                self.config[key] = value

    @property
    def is_global(self) -> bool:
        """Whether this config is the per-user ~/.prime one (symlink-insensitive)."""
        return is_global_config_dir(self.config_dir)

    @staticmethod
    def _strip_api_v1(url: str) -> str:
        # make base_url consistent even if user passed a /api/v1 variant
        return url.rstrip("/").removesuffix("/api/v1")

    def _ensure_dirs(self) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.environments_dir.mkdir(exist_ok=True)

    def _ensure_config_dir(self) -> None:
        """Create config directory and a default config file if they don't exist"""
        self._ensure_dirs()
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
            self.config = ConfigModel(**config_data).model_dump()
        else:
            self.config = {}

    def _write_json(self, path: Path, data: dict) -> None:
        """Owner-only JSON write; symlinks are only followed for the global config."""
        _write_private_json(path, data, allow_symlink=self.is_global)

    def _save_config(self, config: dict) -> None:
        """Save configuration to file"""
        self._ensure_dirs()
        self._write_json(self.config_file, config)
        self.config = config
        self._record_trust()

    def _record_trust(self, path: Path | None = None) -> None:
        """Re-approve a project-local file after the CLI itself changed it.

        Trust is bound to the file's content digest, so every write by the CLI
        has to refresh it. Only discovered/explicit project configs need this;
        the global config and an explicit PRIME_CONFIG_DIR are trusted as such.
        Applies to config.json and to the environment files next to it.
        """
        if self.config_source in ("local", "explicit") and not self.is_global:
            trust_local_file(path or self.config_file)

    def _environment_file_is_trusted(self, env_file: Path) -> bool:
        """Whether an environment file may be loaded from this config.

        Project configs need the check whether discovered or reached via
        `Config.local()`: config.json is vetted by digest (or refused up
        front), but a repository update can rewrite an environment file while
        leaving config.json untouched. Mirrors `_record_trust`. The global
        config and an explicit PRIME_CONFIG_DIR are trusted as such.
        """
        if self.is_global or self.config_source not in ("local", "explicit"):
            return True
        return is_trusted_local_file(env_file)

    @property
    def api_key(self) -> str:
        """Get API key with precedence: env > file > empty."""
        return os.getenv("PRIME_API_KEY") or self.config.get("api_key", "")

    def set_api_key(self, value: str) -> None:
        """Set API key in config file"""
        self.config["api_key"] = value
        self._save_config(self.config)

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
        self.config["team_id"] = value or None
        self.config["team_name"] = team_name if value else None
        self.config["team_role"] = team_role if value else None
        self._save_config(self.config)

    @property
    def user_id(self) -> Optional[str]:
        """Get user ID with precedence: env > file > None."""
        user_id = os.getenv("PRIME_USER_ID")
        if user_id is not None:
            return user_id
        return self.config.get("user_id") or None

    def set_user_id(self, value: str | None) -> None:
        """Set user ID in config file"""
        self.config["user_id"] = value if value else None
        self._save_config(self.config)

    @property
    def base_url(self) -> str:
        """Get API base URL with precedence: env > file > default."""
        env_val = os.getenv("PRIME_API_BASE_URL") or os.getenv("PRIME_BASE_URL")
        if env_val:
            return self._strip_api_v1(env_val)
        return self._strip_api_v1(self.config.get("base_url", self.DEFAULT_BASE_URL))

    def set_base_url(self, value: str) -> None:
        """Set API base URL in config file"""
        value = value.rstrip("/")
        if value.endswith("/api/v1"):
            value = value[:-7]
        self.config["base_url"] = value
        self._save_config(self.config)

    @property
    def frontend_url(self) -> str:
        """Get frontend URL with precedence: env > file > default."""
        env_val = os.getenv("PRIME_FRONTEND_URL")
        if env_val:
            return env_val.rstrip("/")
        return (self.config.get("frontend_url", self.DEFAULT_FRONTEND_URL)).rstrip("/")

    def set_frontend_url(self, value: str) -> None:
        """Set frontend URL in config file"""
        value = value.rstrip("/")
        self.config["frontend_url"] = value
        self._save_config(self.config)

    @property
    def inference_url(self) -> str:
        """Get inference URL with precedence: env > file > default."""
        env_val = os.getenv("PRIME_INFERENCE_URL")
        if env_val:
            return env_val.rstrip("/")
        return self.config.get("inference_url", self.DEFAULT_INFERENCE_URL).rstrip("/")

    def set_inference_url(self, value: str) -> None:
        """Set inference URL in config file"""
        value = value.rstrip("/")
        self.config["inference_url"] = value
        self._save_config(self.config)

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
        """Get Prime Traces service URL with precedence: env > file > base_url."""
        return self._configured_traces_url() or self.base_url

    def set_traces_url(self, value: str) -> None:
        """Set Prime Traces service URL in config file; empty clears the override."""
        self.config["traces_url"] = self._strip_api_v1(value) if value else None
        self._save_config(self.config)

    def set_traces_url_for_active_environment(self, value: str) -> None:
        """Persist only the traces URL for the command's selected environment."""
        traces_url = self._strip_api_v1(value) if value else None
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

        # Validate everything before the first write so a refusal leaves no
        # half-applied state (root updated, environment file not).
        env_file: Path | None = None
        if selected_environment != "production":
            sanitized = self._sanitize_environment_name(selected_environment)
            env_file = self.environments_dir / f"{sanitized}.json"
            if not env_file.exists():
                env_file = None
            elif not self._environment_file_is_trusted(env_file):
                # Read-modify-write would launder a rewritten file into trust.
                raise ValueError(
                    f"Environment file {env_file} is not trusted; review it and run "
                    f"'prime config trust {self.config_dir.parent}' first"
                )

        if update_root:
            root_config["traces_url"] = traces_url
            self._write_json(self.config_file, root_config)
            self._record_trust()

        if env_file is not None:
            env_config = json.loads(env_file.read_text())
            if not isinstance(env_config, dict):
                raise ValueError(f"Invalid configuration in {env_file}")
            env_config["traces_url"] = traces_url
            self._write_json(env_file, env_config)
            self._record_trust(env_file)

        self.config["traces_url"] = traces_url

    @property
    def ssh_key_path(self) -> str:
        """Get SSH private key path with precedence: env > file > default."""
        env_val = os.getenv("PRIME_SSH_KEY_PATH")
        if env_val:
            return str(Path(env_val).expanduser())
        return self.config.get("ssh_key_path", self.DEFAULT_SSH_KEY_PATH)

    def set_ssh_key_path(self, value: str) -> None:
        """Set SSH private key path in config file"""
        self.config["ssh_key_path"] = str(Path(value).expanduser().resolve())
        self._save_config(self.config)

    @property
    def share_resources_with_team(self) -> bool:
        """Get share_resources_with_team setting from config file."""
        val = self.config.get("share_resources_with_team", False)
        if isinstance(val, str):
            return val.lower() == "true"
        return bool(val)

    def set_share_resources_with_team(self, value: bool) -> None:
        """Set share_resources_with_team in config file"""
        self.config["share_resources_with_team"] = value
        self._save_config(self.config)

    @property
    def current_environment(self) -> str:
        """Get current environment name"""
        current_env: str = self.config.get("current_environment", "production")
        return current_env

    def set_current_environment(self, value: str) -> None:
        """Set current environment name"""
        self.config["current_environment"] = value
        self._save_config(self.config)

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
        self._ensure_dirs()
        env_file = self.environments_dir / f"{sanitized_name}.json"
        if env_file.exists() and not self._environment_file_is_trusted(env_file):
            # Overwriting would put the API key into a file the repository
            # controls (it may be tracked) and then record it as trusted.
            raise ValueError(
                f"Environment file {env_file} exists but is not trusted; review it and run "
                f"'prime config trust {self.config_dir.parent}' first, or delete it"
            )
        env_config = {
            "api_key": self.api_key,
            "team_id": self.team_id,
            "team_name": None if self.team_id_from_env else self.team_name,
            "team_role": None if self.team_id_from_env else self.team_role,
            "user_id": self.user_id,
            "base_url": self.base_url,
            "frontend_url": self.frontend_url,
            "inference_url": self.inference_url,
            "traces_url": self._configured_traces_url(),
        }
        self._write_json(env_file, env_config)
        self._record_trust(env_file)

    def delete_environment(self, name: str) -> None:
        """Delete a saved environment configuration."""
        if name.lower() == "production":
            raise ValueError("Cannot delete built-in environment 'production'")

        sanitized_name = self._sanitize_environment_name(name)
        if self.current_environment.casefold() == sanitized_name.casefold():
            raise ValueError(
                f"Cannot delete currently active environment '{name}'. "
                "Use 'prime config use production' or another saved environment first."
            )

        env_file = self.environments_dir / f"{sanitized_name}.json"
        if not self.is_global and involves_symlink(env_file):
            # unlink() through a symlinked environments/ dir would delete
            # whatever external file the link points at.
            raise ValueError(
                f"Refusing to delete {env_file}: it, its directory, or its .prime directory "
                "is a symlink"
            )
        if not env_file.exists():
            raise ValueError(f"Unknown environment: {name}")

        env_file.unlink()
        if not self.is_global:
            forget_trusted_file(env_file)

    def load_environment(self, name: str, persist: bool = True) -> bool:
        """Load a named environment configuration.

        Args:
            name: The environment name to load
            persist: If True, save changes to disk. If False, only update in-memory config.

        Returns:
            True if the environment was loaded successfully, False otherwise.
        """
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
                self.config["base_url"] = self.DEFAULT_BASE_URL
                self.config["frontend_url"] = self.DEFAULT_FRONTEND_URL
                self.config["inference_url"] = self.DEFAULT_INFERENCE_URL
                self.config["traces_url"] = None
                self.config["team_id"] = None
                self.config["team_name"] = None
                self.config["team_role"] = None
                self.config["current_environment"] = "production"
            return True

        try:
            sanitized_name = self._sanitize_environment_name(name)
            env_file = self.environments_dir / f"{sanitized_name}.json"
            if env_file.exists():
                if not self._environment_file_is_trusted(env_file):
                    if persist:
                        raise ValueError(
                            f"Environment file {env_file} is not trusted; review it and run "
                            f"'prime config trust {self.config_dir.parent}' to use it"
                        )
                    # PRIME_CONTEXT: don't take the whole command down, just skip it.
                    _warn_untrusted([env_file], kind="environment file")
                    return False
                try:
                    env_config = json.loads(env_file.read_text())
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in environment file {sanitized_name}.json: {e}")

                if persist:
                    if "api_key" in env_config:
                        self.set_api_key(env_config["api_key"])
                    # Set team_id, team_name, and team_role from environment
                    self.set_team(
                        env_config.get("team_id", None),
                        team_name=env_config.get("team_name", None),
                        team_role=env_config.get("team_role", None),
                    )
                    # Set user_id from environment
                    self.set_user_id(env_config.get("user_id", None))
                    self.set_base_url(env_config.get("base_url", self.DEFAULT_BASE_URL))
                    self.set_frontend_url(env_config.get("frontend_url", self.DEFAULT_FRONTEND_URL))
                    self.set_inference_url(
                        env_config.get("inference_url", self.DEFAULT_INFERENCE_URL)
                    )
                    self.set_traces_url(env_config.get("traces_url") or "")
                    self.set_current_environment(name)
                else:
                    # In-memory only - don't persist to disk
                    if "api_key" in env_config:
                        self.config["api_key"] = env_config["api_key"]
                    self.config["team_id"] = env_config.get("team_id", None)
                    self.config["team_name"] = env_config.get("team_name", None)
                    self.config["team_role"] = env_config.get("team_role", None)
                    self.config["user_id"] = env_config.get("user_id", None)
                    # Normalize URLs the same way set_* methods do
                    base_url = env_config.get("base_url", self.DEFAULT_BASE_URL)
                    self.config["base_url"] = self._strip_api_v1(base_url)
                    frontend_url = env_config.get("frontend_url", self.DEFAULT_FRONTEND_URL)
                    self.config["frontend_url"] = frontend_url.rstrip("/")
                    inference_url = env_config.get("inference_url", self.DEFAULT_INFERENCE_URL)
                    self.config["inference_url"] = inference_url.rstrip("/")
                    traces_url = env_config.get("traces_url")
                    self.config["traces_url"] = (
                        self._strip_api_v1(traces_url) if traces_url else None
                    )
                    self.config["current_environment"] = name
                return True
        except ValueError:
            # Re-raise sanitization errors
            raise
        return False

    def update_current_environment_file(self, *, from_env: bool = True) -> None:
        """Update the current environment's saved file with current config.

        With `from_env=False` the raw stored values are written instead of the
        env-precedence properties, so PRIME_* shell variables can't leak back
        onto disk (logout relies on this). Either way the write goes through
        the private writer and refreshes the file's trust entry.
        """
        if self.current_environment != "production":
            # Only update custom environments, not the built-in production
            try:
                sanitized_name = self._sanitize_environment_name(self.current_environment)
                env_file = self.environments_dir / f"{sanitized_name}.json"
                if env_file.exists():
                    if not self._environment_file_is_trusted(env_file):
                        # Implicit side effect of login/set-*: don't fail the
                        # command, but never copy the key into a file the
                        # repository rewrote (it may be tracked).
                        _warn_skipped_environment_write(env_file)
                        return
                    if from_env:
                        env_config = {
                            "api_key": self.api_key,
                            "team_id": self.team_id,
                            "team_name": None if self.team_id_from_env else self.team_name,
                            "team_role": None if self.team_id_from_env else self.team_role,
                            "user_id": self.user_id,
                            "base_url": self.base_url,
                            "frontend_url": self.frontend_url,
                            "inference_url": self.inference_url,
                            "traces_url": self._stored_traces_url(),
                        }
                    else:
                        raw = self.config
                        env_config = {
                            "api_key": raw.get("api_key", ""),
                            "team_id": raw.get("team_id"),
                            "team_name": raw.get("team_name"),
                            "team_role": raw.get("team_role"),
                            "user_id": raw.get("user_id"),
                            "base_url": raw.get("base_url", self.DEFAULT_BASE_URL),
                            "frontend_url": raw.get("frontend_url", self.DEFAULT_FRONTEND_URL),
                            "inference_url": raw.get("inference_url", self.DEFAULT_INFERENCE_URL),
                            "traces_url": raw.get("traces_url"),
                        }
                    self._write_json(env_file, env_config)
                    self._record_trust(env_file)
            except ValueError:
                # Skip updating if environment name is invalid
                pass

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
