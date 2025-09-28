import json
import os
import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict


class ConfigModel(BaseModel):
    api_key: str = ""
    team_id: str | None = None
    base_url: str = "https://api.primeintellect.ai"
    frontend_url: str = "https://app.primeintellect.ai"
    inference_url: str = "https://api.pinference.ai/api/v1"
    ssh_key_path: str = str(Path.home() / ".ssh" / "id_rsa")
    current_environment: str = "production"

    model_config = ConfigDict(populate_by_name=True)


class Config:
    DEFAULT_BASE_URL: str = "https://api.primeintellect.ai"
    DEFAULT_FRONTEND_URL: str = "https://app.primeintellect.ai"
    DEFAULT_INFERENCE_URL: str = "https://api.pinference.ai/api/v1"
    DEFAULT_SSH_KEY_PATH: str = str(Path.home() / ".ssh" / "id_rsa")

    def __init__(self) -> None:
        self.config_dir = Path.home() / ".prime"
        self.config_file = self.config_dir / "config.json"
        self.environments_dir = self.config_dir / "environments"
        self._ensure_config_dir()
        self._load_config()

    def _ensure_config_dir(self) -> None:
        """Create config directory if it doesn't exist"""
        self.config_dir.mkdir(exist_ok=True)
        self.environments_dir.mkdir(exist_ok=True)
        if not self.config_file.exists():
            self._save_config(
                ConfigModel(
                    api_key="",
                    team_id=None,
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

    def _save_config(self, config: dict) -> None:
        """Save configuration to file"""
        self.config_file.write_text(json.dumps(config, indent=2))
        self.config = config

    @property
    def api_key(self) -> str:
        """Get API key from config file or environment"""
        return self.config.get("api_key", "") or os.getenv("PRIME_API_KEY", "")

    def set_api_key(self, value: str) -> None:
        """Set API key in config file"""
        self.config["api_key"] = value
        self._save_config(self.config)

    @property
    def team_id(self) -> Optional[str]:
        """Get team ID from config file or environment"""
        team_id = self.config.get("team_id", None) or os.getenv("PRIME_TEAM_ID", None)
        return team_id if team_id else None

    def set_team_id(self, value: str | None) -> None:
        """Set team ID in config file"""
        self.config["team_id"] = value if value else None
        self._save_config(self.config)

    @property
    def base_url(self) -> str:
        """Get API base URL from config"""
        base_url: str = self.config.get("base_url", self.DEFAULT_BASE_URL)
        return base_url

    def set_base_url(self, value: str) -> None:
        """Set API base URL in config file"""
        value = value.rstrip("/")
        if value.endswith("/api/v1"):
            value = value[:-7]
        self.config["base_url"] = value
        self._save_config(self.config)

    @property
    def frontend_url(self) -> str:
        """Get frontend URL from config"""
        frontend_url: str = self.config.get("frontend_url", self.DEFAULT_FRONTEND_URL)
        return frontend_url

    def set_frontend_url(self, value: str) -> None:
        """Set frontend URL in config file"""
        value = value.rstrip("/")
        self.config["frontend_url"] = value
        self._save_config(self.config)

    @property
    def inference_url(self) -> str:
        """Get inference URL from config"""
        inference_url: str = self.config.get(
            "inference_url", self.DEFAULT_INFERENCE_URL
        ).rstrip("/")
        return inference_url

    def set_inference_url(self, value: str) -> None:
        """Set frontend URL in config file"""
        value = value.rstrip("/")
        if value.endswith("/api/v1"):
            value = value[:-7]
        self.config["inference_url"] = value
        self._save_config(self.config)

    @property
    def ssh_key_path(self) -> str:
        """Get SSH private key path from config file or environment"""
        return self.config.get("ssh_key_path", self.DEFAULT_SSH_KEY_PATH) or os.getenv(
            "PRIME_SSH_KEY_PATH", self.DEFAULT_SSH_KEY_PATH
        )

    def set_ssh_key_path(self, value: str) -> None:
        """Set SSH private key path in config file"""
        self.config["ssh_key_path"] = str(Path(value).expanduser().resolve())
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
            "base_url": self.base_url,
            "frontend_url": self.frontend_url,
            "inference_url": self.inference_url,
            "ssh_key_path": self.ssh_key_path,
            "current_environment": self.current_environment,
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
            "base_url": self.base_url,
            "frontend_url": self.frontend_url,
            "inference_url": self.inference_url,
        }
        env_file.write_text(json.dumps(env_config, indent=2))

    def load_environment(self, name: str) -> bool:
        """Load a named environment configuration"""
        if name.lower() == "production":
            # Built-in production environment
            self.set_base_url(self.DEFAULT_BASE_URL)
            self.set_frontend_url(self.DEFAULT_FRONTEND_URL)
            self.set_inference_url(self.DEFAULT_INFERENCE_URL)
            self.set_team_id(None)  # Production defaults to personal account
            self.set_current_environment("production")
            return True

        try:
            sanitized_name = self._sanitize_environment_name(name)
            env_file = self.environments_dir / f"{sanitized_name}.json"
            if env_file.exists():
                try:
                    env_config = json.loads(env_file.read_text())
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in environment file {sanitized_name}.json: {e}")

                if "api_key" in env_config:
                    self.set_api_key(env_config["api_key"])
                # Set team_id from environment, defaulting to empty string
                self.set_team_id(env_config.get("team_id", None))
                self.set_base_url(env_config.get("base_url", self.DEFAULT_BASE_URL))
                self.set_frontend_url(env_config.get("frontend_url", self.DEFAULT_FRONTEND_URL))
                self.set_inference_url(env_config.get("inference_url", self.DEFAULT_INFERENCE_URL))
                self.set_current_environment(name)
                return True
        except ValueError:
            # Re-raise sanitization errors
            raise
        return False

    def update_current_environment_file(self) -> None:
        """Update the current environment's saved file with current config"""
        if self.current_environment != "production":
            # Only update custom environments, not the built-in production
            try:
                sanitized_name = self._sanitize_environment_name(self.current_environment)
                env_file = self.environments_dir / f"{sanitized_name}.json"
                if env_file.exists():
                    env_config = {
                        "api_key": self.api_key,
                        "team_id": self.team_id,
                        "base_url": self.base_url,
                        "frontend_url": self.frontend_url,
                        "inference_url": self.inference_url,
                    }
                    env_file.write_text(json.dumps(env_config, indent=2))
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
