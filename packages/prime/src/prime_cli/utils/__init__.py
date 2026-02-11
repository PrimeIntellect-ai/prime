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
    strip_ansi,
)
from .params import optional_team_params
from .prompt import (
    any_provided,
    confirm_or_skip,
    prompt_for_value,
    require_selection,
    select_item_interactive,
    validate_env_var_name,
)
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
    "strip_ansi",
    "any_provided",
    "confirm_or_skip",
    "prompt_for_value",
    "select_item_interactive",
    "require_selection",
    "validate_env_var_name",
    "load_toml",
    "BaseConfig",
    "optional_team_params",
]
