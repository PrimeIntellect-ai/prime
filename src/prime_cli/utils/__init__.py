"""Shared utilities for CLI commands."""

# Re-export the most commonly used functions
from .display import build_table, output_data_as_json, status_color, validate_output_format
from .formatters import format_ip_display, format_price, format_resources, obfuscate_env_vars
from .prompt import confirm_or_skip
from .time_utils import human_age, iso_timestamp, sort_by_created
from .sandbox import expand_home_in_path, parse_cp_arg, format_sandbox_for_list, format_sandbox_for_details

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
    "expand_home_in_path",
    "parse_cp_arg",
    "format_sandbox_for_list",
    "format_sandbox_for_details",
]
