"""``prime_traces.core.Config`` (env vars, then ``~/.prime/config.json``) plus the
dashboard URL and the Prime Traces opt-out."""

import os

from prime_traces.core import Config as _TracesConfig

from .exceptions import ConfigurationError

TRACES_OPT_OUT_ENV = "PRIME_TRACES_OPT_OUT"

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


class Config(_TracesConfig):
    DEFAULT_FRONTEND_URL: str = "https://app.primeintellect.ai"

    @property
    def traces_opt_out(self) -> bool:
        """Opt out of Prime Traces: samples upload to the legacy tables instead.

        Precedence is ``$PRIME_TRACES_OPT_OUT`` > config
        ``traces_opt_out`` > off.
        """
        env_val = os.getenv(TRACES_OPT_OUT_ENV)
        if env_val is not None and env_val.strip():
            normalized = env_val.strip().lower()
            if normalized in _TRUE:
                return True
            if normalized in _FALSE:
                return False
            raise ConfigurationError(
                f"{TRACES_OPT_OUT_ENV}={env_val!r} is not one of true or false"
            )
        file_val = self.config.get("traces_opt_out", False)
        if isinstance(file_val, str):
            return file_val.strip().lower() in _TRUE
        return bool(file_val)

    @property
    def frontend_url(self) -> str:
        env_val = os.getenv("PRIME_FRONTEND_URL")
        if env_val:
            return env_val.rstrip("/")
        return str(self.config.get("frontend_url") or self.DEFAULT_FRONTEND_URL).rstrip("/")
