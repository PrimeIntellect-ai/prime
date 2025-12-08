"""Lightweight configuration for SDK packages."""

import json
import os
from pathlib import Path
from typing import Optional


class Config:
    """Minimal configuration class for SDK packages.

    Reads from ~/.prime/config.json and environment variables.
    """

    DEFAULT_BASE_URL: str = "https://api.primeintellect.ai"

    def __init__(self) -> None:
        self.config_dir = Path.home() / ".prime"
        self.config_file = self.config_dir / "config.json"
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from file"""
        if self.config_file.exists():
            try:
                config_data = json.loads(self.config_file.read_text())
                self.config = config_data
            except (json.JSONDecodeError, IOError):
                self.config = {}
        else:
            self.config = {}

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
        """Get API base URL with precedence: env > file > default."""
        env_val = os.getenv("PRIME_API_BASE_URL") or os.getenv("PRIME_BASE_URL")
        if env_val:
            return self._strip_api_v1(env_val)
        return self._strip_api_v1(self.config.get("base_url", self.DEFAULT_BASE_URL))
