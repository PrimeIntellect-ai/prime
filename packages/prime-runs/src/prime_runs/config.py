"""``prime_traces.core.Config`` (env vars, then ``~/.prime/config.json``) plus the
dashboard URL."""

import os

from prime_traces.core import Config as _TracesConfig


class Config(_TracesConfig):
    DEFAULT_FRONTEND_URL: str = "https://app.primeintellect.ai"

    @property
    def frontend_url(self) -> str:
        env_val = os.getenv("PRIME_FRONTEND_URL")
        if env_val:
            return env_val.rstrip("/")
        return str(self.config.get("frontend_url") or self.DEFAULT_FRONTEND_URL).rstrip("/")
