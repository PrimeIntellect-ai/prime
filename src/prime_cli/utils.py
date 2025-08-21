"""
Shared utilities for CLI commands.

This module provides backward compatibility imports.
For new code, import from specific utils submodules.
"""

# Backward compatibility imports
from .utils.display import output_data_as_json, validate_output_format
from .utils.prompt import confirm_or_skip

__all__ = [
    "output_data_as_json",
    "validate_output_format", 
    "confirm_or_skip",
]
