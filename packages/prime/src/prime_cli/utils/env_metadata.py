"""Utilities for reading and managing environment metadata."""
import json
from pathlib import Path
from typing import Any, Dict, Optional


def get_environment_metadata(env_path: Path) -> Optional[Dict[str, Any]]:
    """Read environment metadata from .prime/.env-metadata.json with backwards compatibility.
    
    Checks both the new location (.prime/.env-metadata.json) and old location
    (.env-metadata.json) for backwards compatibility.
    
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

