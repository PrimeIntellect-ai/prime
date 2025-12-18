"""Configuration file utilities."""

from pathlib import Path
from typing import Any

from typing_extensions import Self

import toml
import typer
from pydantic import BaseModel
from rich.console import Console


def load_toml(path: str, console: Console | None = None) -> dict[str, Any]:
    """Load and parse a TOML configuration file.

    Args:
        path: Path to the TOML file.
        console: Optional Rich console for error output.

    Returns:
        Dictionary with configuration values.

    Raises:
        typer.Exit: If the config file doesn't exist or is invalid TOML.
    """
    console = console or Console()
    p = Path(path)

    if not p.exists():
        console.print(f"[red]Error:[/red] Config file not found: {path}")
        raise typer.Exit(1)

    try:
        return toml.load(p)
    except toml.TomlDecodeError as e:
        console.print(f"[red]Error:[/red] Invalid TOML in {path}: {e}")
        raise typer.Exit(1)


class BaseConfig(BaseModel):
    """Base configuration class with TOML + CLI merge support.

    Subclass this to define command-specific configs. The class structure
    defines the expected TOML schema.

    Example:
        class MyConfig(BaseConfig):
            name: str | None = None
            count: int = 10
            nested: NestedConfig = Field(default_factory=NestedConfig)

        # Load from TOML with CLI overrides
        cfg = MyConfig.from_sources(
            toml_path="config.toml",
            name=cli_name,
            count=cli_count,
        )
    """

    @classmethod
    def from_sources(
        cls,
        toml_path: str | None = None,
        console: Console | None = None,
        **cli_overrides: Any,
    ) -> Self:
        """Load config with precedence: CLI > TOML > defaults.

        Args:
            toml_path: Optional path to TOML config file.
            console: Rich console for error messages.
            **cli_overrides: CLI argument values. None values are ignored.
                For nested fields, use underscore notation (e.g., wandb_project
                maps to the 'project' field inside the 'wandb' section).

        Returns:
            Validated config instance with merged values.
        """
        # Start with TOML data or empty dict
        data: dict[str, Any] = {}
        if toml_path:
            data = load_toml(toml_path, console)

        # Apply CLI overrides (skip None values)
        for key, value in cli_overrides.items():
            if value is None:
                continue

            # Check if this is a direct field
            if key in cls.model_fields:
                data[key] = value
                continue

            # Handle underscore notation for nested fields (e.g., wandb_project)
            if "_" in key:
                parts = key.split("_", 1)
                prefix, suffix = parts[0], parts[1]
                if prefix in cls.model_fields:
                    # Ensure nested dict exists and set the value
                    if prefix not in data:
                        data[prefix] = {}
                    if isinstance(data[prefix], dict):
                        data[prefix][suffix] = value
                    continue

            # If we get here, just set it directly (may fail validation)
            data[key] = value

        return cls.model_validate(data)
