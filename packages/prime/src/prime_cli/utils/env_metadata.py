"""Utilities for reading and managing environment metadata."""
import json
from pathlib import Path
from typing import Any, Dict, Optional


def get_environment_metadata(env_path: Path) -> Optional[Dict[str, Any]]:
    """Read environment metadata from .prime/.env-metadata.json with backwards compatibility.
    
    Checks both the new location (.prime/.env-metadata.json) and old location
    (.env-metadata.json) for backwards compatibility.
    
    This function only checks the provided path - it does not search multiple directories.
    Use find_environment_metadata() if you need to search multiple locations.
    
    Args:
        env_path: Path to the environment directory
        
    Returns:
        Dictionary containing environment metadata, or None if not found
    """
    # Try new location first
    metadata_path = env_path / ".prime" / ".env-metadata.json"
    if not metadata_path.exists():
        # Fall back to old location for backwards compatibility
        metadata_path = env_path / ".env-metadata.json"
    
    if not metadata_path.exists():
        return None
    
    try:
        with open(metadata_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def find_environment_metadata(
    env_name: Optional[str] = None,
    env_path: Optional[Path] = None,
    module_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Search for environment metadata in multiple common locations.
    
    Searches directories in the following order:
    1. env_path (if provided)
    2. ./environments/{module_name} (if module_name provided)
    3. ./environments/{env_name} (if env_name provided)
    4. ./{env_name} (if env_name provided)
    5. ./{module_name} (if module_name provided)
    6. ./ (current directory)
    
    Args:
        env_name: Environment name (e.g., "simpleqa")
        env_path: Optional explicit path to check first
        module_name: Optional module name (e.g., "simple_qa" for env_name "simpleqa")
        
    Returns:
        Dictionary containing environment metadata from the first location found, or None
    """
    possible_env_dirs = []
    
    # 1. Check explicit path first if provided
    if env_path:
        possible_env_dirs.append(env_path)
    
    # 2-5. Check common locations based on provided names
    if module_name:
        possible_env_dirs.append(Path("./environments") / module_name)
    if env_name:
        possible_env_dirs.append(Path("./environments") / env_name)
        possible_env_dirs.append(Path(".") / env_name)
    if module_name:
        possible_env_dirs.append(Path(".") / module_name)
    
    # 6. Check current directory last
    possible_env_dirs.append(Path("."))
    
    # Search through directories and return first match
    for env_dir in possible_env_dirs:
        metadata = get_environment_metadata(env_dir)
        if metadata:
            return metadata
    
    return None

