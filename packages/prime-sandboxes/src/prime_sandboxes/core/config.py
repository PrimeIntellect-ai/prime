"""Lightweight configuration for SDK packages."""

import json
import os
import re
from pathlib import Path
from typing import Optional


class Config:
    """Minimal configuration class for SDK packages.

    Config() reads from ~/.prime/config.json and environment variables.
    from_values() uses only the supplied values. Neither writes config files.
    """

    DEFAULT_BASE_URL: str = "https://api.primeintellect.ai"

    def __init__(self) -> None:
        self._explicit = False
        self.config_dir: Optional[Path] = Path.home() / ".prime"
        self.config_file: Optional[Path] = self.config_dir / "config.json"
        self.environments_dir: Optional[Path] = self.config_dir / "environments"
        self._load_config()
        self._load_context()

    @classmethod
    def from_values(
        cls,
        *,
        api_key: str,
        base_url: str,
        team_id: Optional[str] = None,
    ) -> "Config":
        """Resolve explicit values without reading the environment or local profiles.

        ``team_id=None`` means personal scope. This config has no filesystem
        directory; sandbox clients must use ``persist_auth_cache=False``.
        """
        if not api_key.strip():
            raise ValueError("An explicit API key is required")
        if not base_url.strip():
            raise ValueError("An explicit API base URL is required")
        config = cls.__new__(cls)
        config._explicit = True
        config.config_dir = None
        config.config_file = None
        config.environments_dir = None
        config.config = {
            "api_key": api_key,
            "base_url": cls._strip_api_v1(base_url),
            "team_id": team_id or None,
        }
        return config

    def _load_config(self) -> None:
        """Load configuration from file"""
        if self.config_file is not None and self.config_file.exists():
            try:
                config_data = json.loads(self.config_file.read_text())
                self.config = config_data
            except (json.JSONDecodeError, IOError):
                self.config = {}
        else:
            self.config = {}

    def _load_context(self) -> None:
        """Overlay the profile selected by the Prime CLI for this process."""
        context = os.getenv("PRIME_CONTEXT")
        if not context:
            return

        if context.casefold() == "production":
            self.config.update(
                {
                    "base_url": self.DEFAULT_BASE_URL,
                    "team_id": None,
                    "team_name": None,
                    "team_role": None,
                }
            )
            return

        if re.fullmatch(r"[a-zA-Z0-9_-]+", context) is None:
            raise ValueError(f"Invalid context name: {context!r}")

        assert self.environments_dir is not None
        environment_file = self.environments_dir / f"{context}.json"
        if not environment_file.exists():
            raise ValueError(f"Context file not found: {environment_file}")
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
        if self._explicit:
            return self.config["api_key"]
        return os.getenv("PRIME_API_KEY") or self.config.get("api_key", "")

    @property
    def team_id(self) -> Optional[str]:
        """Get team ID with precedence: env > file > None.

        An explicitly empty PRIME_TEAM_ID means personal scope: it must not
        fall back to the file's team, and must never leak onto the wire as
        ``teamId: ""``.
        """
        if self._explicit:
            return self.config["team_id"]
        team_id = os.getenv("PRIME_TEAM_ID")
        if team_id is not None:
            return team_id or None
        return self.config.get("team_id") or None

    @property
    def user_id(self) -> Optional[str]:
        """Get user ID with precedence: env > file > None."""
        if self._explicit:
            return None
        user_id = os.getenv("PRIME_USER_ID")
        if user_id is not None:
            return user_id
        return self.config.get("user_id") or None

    @property
    def base_url(self) -> str:
        """Get API base URL with precedence: env > file > default."""
        if self._explicit:
            return self.config["base_url"]
        env_val = os.getenv("PRIME_API_BASE_URL") or os.getenv("PRIME_BASE_URL")
        if env_val:
            return self._strip_api_v1(env_val)
        return self._strip_api_v1(self.config.get("base_url", self.DEFAULT_BASE_URL))
