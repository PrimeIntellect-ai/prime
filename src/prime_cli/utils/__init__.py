"""Shared utilities for CLI commands."""

# Re-export the most commonly used functions
from .display import output_data_as_json, validate_output_format, build_table, status_color
from .time_utils import human_age, iso_timestamp, sort_by_created
from .formatters import obfuscate_env_vars, format_ip_display, format_price, format_resources
from .prompt import confirm_or_skip

__all__ = [
    "output_data_as_json",
    "validate_output_format", 
    "build_table",
    "status_color",
    "human_age",
    "iso_timestamp",
    "sort_by_created",
    "obfuscate_env_vars",
    "format_ip_display",
    "format_price",
    "format_resources",
    "confirm_or_skip",
]
