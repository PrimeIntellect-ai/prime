import json
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class ConfigModel(BaseModel):
    api_key: str = ""
    team_id: str = ""
    base_url: str = "https://api.primeintellect.ai"
    frontend_url: str = "https://app.primeintellect.ai"
    ssh_key_path: str = str(Path.home() / ".ssh" / "id_rsa")
    current_environment: str = "production"

    model_config = ConfigDict(populate_by_name=True)


class Config:
    DEFAULT_BASE_URL: str = "https://api.primeintellect.ai"
    DEFAULT_FRONTEND_URL: str = "https://app.primeintellect.ai"
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
                    team_id="",
                    base_url=self.DEFAULT_BASE_URL,
                    frontend_url=self.DEFAULT_FRONTEND_URL,
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
        """Get API key from environment or config file"""
        return os.getenv("PRIME_API_KEY") or self.config.get("api_key", "")

    def set_api_key(self, value: str) -> None:
        """Set API key in config file"""
        self.config["api_key"] = value
        self._save_config(self.config)

    @property
    def team_id(self) -> str:
        """Get team ID from environment or config file"""
        return os.getenv("PRIME_TEAM_ID") or self.config.get("team_id", "")

    def set_team_id(self, value: str) -> None:
        """Set team ID in config file"""
        self.config["team_id"] = value
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
    def ssh_key_path(self) -> str:
        """Get SSH private key path from environment or config file"""
        return os.getenv("PRIME_SSH_KEY_PATH") or self.config.get(
            "ssh_key_path", self.DEFAULT_SSH_KEY_PATH
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

    def view(self) -> dict:
        """Get all config values"""
        return {
            "api_key": self.api_key,
            "team_id": self.team_id,
            "base_url": self.base_url,
            "frontend_url": self.frontend_url,
            "ssh_key_path": self.ssh_key_path,
            "current_environment": self.current_environment,
        }

    def save_environment(self, name: str) -> None:
        """Save current configuration as a named environment"""
        env_file = self.environments_dir / f"{name}.json"
        env_config = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "frontend_url": self.frontend_url,
        }
        env_file.write_text(json.dumps(env_config, indent=2))

    def load_environment(self, name: str) -> bool:
        """Load a named environment configuration"""
        if name.lower() == "production":
            # Built-in production environment
            self.set_base_url(self.DEFAULT_BASE_URL)
            self.set_frontend_url(self.DEFAULT_FRONTEND_URL)
            self.set_current_environment("production")
            return True

        env_file = self.environments_dir / f"{name}.json"
        if env_file.exists():
            env_config = json.loads(env_file.read_text())
            if "api_key" in env_config:
                self.set_api_key(env_config["api_key"])
            self.set_base_url(env_config.get("base_url", self.DEFAULT_BASE_URL))
            self.set_frontend_url(env_config.get("frontend_url", self.DEFAULT_FRONTEND_URL))
            self.set_current_environment(name)
            return True
        return False

    def update_current_environment_file(self) -> None:
        """Update the current environment's saved file with current config"""
        if self.current_environment != "production":
            # Only update custom environments, not the built-in production
            env_file = self.environments_dir / f"{self.current_environment}.json"
            if env_file.exists():
                env_config = {
                    "api_key": self.api_key,
                    "base_url": self.base_url,
                    "frontend_url": self.frontend_url,
                }
                env_file.write_text(json.dumps(env_config, indent=2))

    def list_environments(self) -> list[str]:
        """List all saved environment names"""
        environments = ["production"]  # Built-in environment
        if self.environments_dir.exists():
            for env_file in self.environments_dir.glob("*.json"):
                environments.append(env_file.stem)
        return environments
