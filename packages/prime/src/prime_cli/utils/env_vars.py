"""Utilities for parsing environment variables and secrets."""

import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional


def parse_env_file(
    file_path: Path,
    on_warning: Optional[Callable[[str], None]] = None,
) -> Dict[str, str]:
    """Parse a .env file and return a dict of key-value pairs.

    Supports:
    - KEY=VALUE
    - KEY="VALUE" (quoted values)
    - KEY='VALUE' (single-quoted values)
    - # comments
    - Empty lines (ignored)

    Args:
        file_path: Path to the .env file
        on_warning: Optional callback for warning messages

    Returns:
        Dict of environment variable key-value pairs
    """
    env_vars: Dict[str, str] = {}
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                if on_warning:
                    on_warning(
                        f"Skipping invalid line {line_num} in {file_path}: "
                        f"missing '=' separator"
                    )
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", key):
                if on_warning:
                    on_warning(
                        f"Skipping invalid key '{key}' in {file_path}: "
                        f"must start with letter/underscore, contain only alphanumeric/underscore"
                    )
                continue
            env_vars[key] = value
    return env_vars


class EnvParseError(Exception):
    """Error raised when parsing environment variables fails."""

    pass


def parse_env_arg(
    arg: str,
    on_warning: Optional[Callable[[str], None]] = None,
) -> Dict[str, str]:
    """Parse a single -e/--env argument.

    Accepts:
    - KEY=VALUE: Use the provided value
    - KEY: Read value from environment variable $KEY
    - path/to/file.env: If it's an existing file, parse it as an env file

    Args:
        arg: The argument string to parse
        on_warning: Optional callback for warning messages

    Returns:
        Dict of key-value pairs (single entry for KEY=VALUE/KEY, multiple for file)

    Raises:
        EnvParseError: If the argument is invalid or env var is not set
    """
    path = Path(arg)
    if path.is_file():
        return parse_env_file(path, on_warning=on_warning)

    if "=" in arg:
        key, _, value = arg.partition("=")
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", key):
            raise EnvParseError(
                f"Invalid environment variable key '{key}': "
                f"must start with letter/underscore, contain only alphanumeric/underscore"
            )
        return {key: value}

    key = arg.strip()
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", key):
        raise EnvParseError(
            f"Invalid environment variable key '{key}': "
            f"must start with letter/underscore, contain only alphanumeric/underscore"
        )
    value = os.environ.get(key)
    if value is None:
        raise EnvParseError(
            f"Environment variable '{key}' is not set. "
            f"Either set it or use KEY=VALUE syntax."
        )
    return {key: value}


def collect_env_vars(
    env_args: Optional[List[str]] = None,
    env_files: Optional[List[str]] = None,
    on_warning: Optional[Callable[[str], None]] = None,
) -> Dict[str, str]:
    """Collect environment variables from various sources.

    Priority (later sources override earlier):
    1. env_files (--env-file arguments)
    2. env_args (-e/--env-var arguments, highest priority)

    Args:
        env_args: List of -e/--env-var argument values
        env_files: List of --env-file paths
        on_warning: Optional callback for warning messages

    Returns:
        Merged dict of all environment variables

    Raises:
        EnvParseError: If a file is not found or parsing fails
    """
    result: Dict[str, str] = {}

    if env_files:
        for file_path in env_files:
            path = Path(file_path)
            if not path.is_file():
                raise EnvParseError(f"Env file not found: {file_path}")
            result.update(parse_env_file(path, on_warning=on_warning))

    if env_args:
        for arg in env_args:
            result.update(parse_env_arg(arg, on_warning=on_warning))

    return result

