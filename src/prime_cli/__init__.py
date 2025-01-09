"""Prime Intellect CLI"""

from importlib.metadata import version
from typing import Dict, List, Optional

from .api.availability import AvailabilityClient, GPUAvailability
from .api.client import APIClient
from .api.pods import Pod, PodConfig, PodList, PodsClient
from .config import Config

__version__ = version("prime-cli")

# Initialize config and API clients
config = Config()
api_client = APIClient()
pods_client = PodsClient(api_client)
availability_client = AvailabilityClient(api_client)


def set_api_key(api_key: str) -> None:
    """Set the API key for authentication"""
    config.set_api_key(api_key)


def get_config() -> Config:
    """Get the current configuration"""
    return config


def get_pods(offset: int = 0, limit: int = 100) -> PodList:
    """Get a list of pods"""
    return pods_client.list(offset=offset, limit=limit)


def get_pod(pod_id: str) -> Pod:
    """Get details of a specific pod"""
    return pods_client.get(pod_id)


def create_pod(pod_config: PodConfig) -> Pod:
    """Create a new pod"""
    return pods_client.create(pod_config.__dict__)


def terminate_pod(pod_id: str) -> None:
    """Delete a pod"""
    pods_client.delete(pod_id)


def get_availability(
    regions: Optional[List[str]] = None,
    gpu_count: Optional[int] = None,
    gpu_type: Optional[str] = None,
) -> Dict[str, List[GPUAvailability]]:
    """Get GPU availability information.

    Args:
        regions: Optional list of regions to filter by
        gpu_count: Optional number of GPUs to filter by
        gpu_type: Optional GPU type to filter by

    Returns:
        Dictionary mapping GPU types to lists of availability information
    """
    return availability_client.get(
        regions=regions,
        gpu_count=gpu_count,
        gpu_type=gpu_type,
    )
