"""``prime_traces.core.Config`` (env vars, then ``~/.prime/config.json``) plus the
dashboard URL and the legacy-samples opt-out."""

import os

from prime_traces.core import Config as _TracesConfig

from .exceptions import ConfigurationError

LEGACY_SAMPLES_ENV = "PRIME_RUNS_LEGACY_SAMPLES"

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


class Config(_TracesConfig):
    DEFAULT_FRONTEND_URL: str = "https://app.primeintellect.ai"

    @property
    def legacy_samples(self) -> bool:
        """Upload samples to the legacy tables instead of Prime Traces.

        Precedence is ``$PRIME_RUNS_LEGACY_SAMPLES`` > config
        ``runs_legacy_samples`` > off.
        """
        env_val = os.getenv(LEGACY_SAMPLES_ENV)
        if env_val is not None and env_val.strip():
            normalized = env_val.strip().lower()
            if normalized in _TRUE:
                return True
            if normalized in _FALSE:
                return False
            raise ConfigurationError(
                f"{LEGACY_SAMPLES_ENV}={env_val!r} is not one of true or false"
            )
        file_val = self.config.get("runs_legacy_samples", False)
        if isinstance(file_val, str):
            return file_val.strip().lower() in _TRUE
        return bool(file_val)

    @property
    def frontend_url(self) -> str:
        env_val = os.getenv("PRIME_FRONTEND_URL")
        if env_val:
            return env_val.rstrip("/")
        return str(self.config.get("frontend_url") or self.DEFAULT_FRONTEND_URL).rstrip("/")
