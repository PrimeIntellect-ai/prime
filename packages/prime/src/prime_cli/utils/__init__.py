"""Shared utilities for CLI commands."""

# Re-export the most commonly used functions
from .config import BaseConfig, load_toml
from .display import build_table, output_data_as_json, status_color, validate_output_format
from .formatters import (
    format_ip_display,
    format_price,
    format_resources,
    obfuscate_env_vars,
    obfuscate_secrets,
)
from .prompt import confirm_or_skip
from .time_utils import human_age, iso_timestamp, sort_by_created

__all__ = [
    "output_data_as_json",
    "validate_output_format",
    "build_table",
    "status_color",
    "human_age",
    "iso_timestamp",
    "sort_by_created",
    "obfuscate_env_vars",
    "obfuscate_secrets",
    "format_ip_display",
    "format_price",
    "format_resources",
    "confirm_or_skip",
    "load_toml",
    "BaseConfig",
]
