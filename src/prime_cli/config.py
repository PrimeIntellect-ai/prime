import json
import os
from pathlib import Path


class Config:
    DEFAULT_BASE_URL: str = "https://api.primeintellect.ai"
    DEFAULT_SSH_KEY_PATH: str = str(Path.home() / ".ssh" / "id_rsa")

    def __init__(self) -> None:
        self.config_dir = Path.home() / ".prime"
        self.config_file = self.config_dir / "config.json"
        self._ensure_config_dir()
        self._load_config()

    def _ensure_config_dir(self) -> None:
        """Create config directory if it doesn't exist"""
        self.config_dir.mkdir(exist_ok=True)
        if not self.config_file.exists():
            self._save_config(
                {
                    "api_key": "",
                    "team_id": "",
                    "base_url": self.DEFAULT_BASE_URL,
                    "ssh_key_path": self.DEFAULT_SSH_KEY_PATH,
                }
            )

    def _load_config(self) -> None:
        """Load configuration from file"""
        if self.config_file.exists():
            self.config = json.loads(self.config_file.read_text())
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
    def ssh_key_path(self) -> str:
        """Get SSH private key path from environment or config file"""
        return os.getenv("PRIME_SSH_KEY_PATH") or self.config.get(
            "ssh_key_path", self.DEFAULT_SSH_KEY_PATH
        )

    def set_ssh_key_path(self, value: str) -> None:
        """Set SSH private key path in config file"""
        self.config["ssh_key_path"] = str(Path(value).expanduser().resolve())
        self._save_config(self.config)

    def view(self) -> dict:
        """Get all config values"""
        return {
            "api_key": self.api_key,
            "team_id": self.team_id,
            "base_url": self.base_url,
            "ssh_key_path": self.ssh_key_path,
        }
