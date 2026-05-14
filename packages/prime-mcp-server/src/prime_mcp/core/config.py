"""Lightweight configuration for MCP server."""

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class Config:
    """Minimal configuration class."""

    DEFAULT_BASE_URL: str = "https://api.primeintellect.ai"

    def __init__(self) -> None:
        self.config_dir = Path.home() / ".prime"
        self.config_file = self.config_dir / "config.json"
        self._load_config()

    def _load_config(self) -> None:
        if self.config_file.exists():
            try:
                config_data = json.loads(self.config_file.read_text())
                self.config = config_data
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(
                    "Failed to parse config file %s: %s. Using empty config.",
                    self.config_file,
                    e,
                )
                self.config = {}
        else:
            self.config = {}

    @staticmethod
    def _strip_api_v1(url: str) -> str:
        return url.rstrip("/").removesuffix("/api/v1")

    @property
    def api_key(self) -> str:
        return os.getenv("PRIME_API_KEY") or self.config.get("api_key", "")

    @property
    def team_id(self) -> Optional[str]:
        team_id = os.getenv("PRIME_TEAM_ID")
        if team_id is not None:
            return team_id
        return self.config.get("team_id") or None

    @property
    def base_url(self) -> str:
        env_val = os.getenv("PRIME_API_BASE_URL") or os.getenv("PRIME_BASE_URL")
        if env_val:
            return self._strip_api_v1(env_val)
        return self._strip_api_v1(self.config.get("base_url", self.DEFAULT_BASE_URL))
