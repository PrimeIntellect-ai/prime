"""Configuration: ``~/.prime/config.json`` plus environment variables, env
taking precedence. Same shape as the other prime SDKs, plus ``frontend_url``."""

import json
import os
from pathlib import Path
from typing import Optional


class Config:
    """Minimal configuration class for SDK packages.

    Reads from ~/.prime/config.json and environment variables.
    """

    DEFAULT_BASE_URL: str = "https://api.primeintellect.ai"
    DEFAULT_FRONTEND_URL: str = "https://app.primeintellect.ai"

    def __init__(self) -> None:
        self.config_dir = Path.home() / ".prime"
        self.config_file = self.config_dir / "config.json"
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from file."""
        config_data: object = {}
        if self.config_file.exists():
            try:
                config_data = json.loads(self.config_file.read_text())
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                config_data = {}
        # Valid JSON that is not an object (a list, a bare string) must degrade
        # the same way invalid JSON does: every accessor below assumes a dict.
        self.config = config_data if isinstance(config_data, dict) else {}

    @staticmethod
    def _strip_api_v1(url: str) -> str:
        return url.rstrip("/").removesuffix("/api/v1")

    @property
    def api_key(self) -> str:
        """API key with precedence: env > file > empty."""
        return os.getenv("PRIME_API_KEY") or self.config.get("api_key", "")

    @property
    def team_id(self) -> Optional[str]:
        """Team ID with precedence: env > file > None."""
        team_id = os.getenv("PRIME_TEAM_ID")
        if team_id is not None:
            return team_id
        return self.config.get("team_id") or None

    @property
    def base_url(self) -> str:
        """Platform API base URL with precedence: env > file > default."""
        env_val = os.getenv("PRIME_API_BASE_URL") or os.getenv("PRIME_BASE_URL")
        if env_val:
            return self._strip_api_v1(env_val)
        return self._strip_api_v1(self.config.get("base_url", self.DEFAULT_BASE_URL))

    @property
    def frontend_url(self) -> str:
        """Dashboard base URL; fallback when a create response omits ``viewer_url``."""
        env_val = os.getenv("PRIME_FRONTEND_URL")
        if env_val:
            return env_val.rstrip("/")
        return str(self.config.get("frontend_url") or self.DEFAULT_FRONTEND_URL).rstrip("/")
