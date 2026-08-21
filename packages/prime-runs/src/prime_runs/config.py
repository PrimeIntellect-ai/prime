"""Configuration: ``prime_traces.core.Config`` (``~/.prime/config.json`` plus
environment variables, env taking precedence) with the dashboard URL added."""

import os

from prime_traces.core import Config as _TracesConfig


class Config(_TracesConfig):
    DEFAULT_FRONTEND_URL: str = "https://app.primeintellect.ai"

    @property
    def frontend_url(self) -> str:
        """Dashboard base URL; fallback when a create response omits ``viewer_url``."""
        env_val = os.getenv("PRIME_FRONTEND_URL")
        if env_val:
            return env_val.rstrip("/")
        return str(self.config.get("frontend_url") or self.DEFAULT_FRONTEND_URL).rstrip("/")
