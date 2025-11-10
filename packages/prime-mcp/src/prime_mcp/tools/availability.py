from typing import Any, Optional

from prime_mcp.client import make_prime_request


async def check_gpu_availability(
    gpu_type: Optional[str] = None,
    regions: Optional[list[str]] = None,
    socket: Optional[str] = None,
    security: Optional[str] = None,
    gpu_count: Optional[int] = None,
) -> dict[str, Any]:
    """Check GPU availability across different providers.

    Args:
        gpu_type: GPU model (e.g., "A100_80GB", "H100_80GB", "RTX4090_24GB", etc.)
        regions: List of regions to filter (options: "africa", "asia_south", "asia_northeast",
            "australia", "canada", "eu_east", "eu_north", "eu_west", "middle_east",
            "south_america", "united_states")
        socket: Socket for selected GPU model
            (options: "PCIe", "SXM2", "SXM3", "SXM4", "SXM5", "SXM6")
        security: Security type ("secure_cloud" or "community_cloud")
        gpu_count: Number of GPUs to filter by

    Returns:
        Available GPU instances matching the criteria
    """
    params = {}
    if regions:
        params["regions"] = regions
    if gpu_type:
        params["gpu_type"] = gpu_type
    if socket:
        params["socket"] = socket
    if security:
        params["security"] = security
    if gpu_count:
        params["gpu_count"] = gpu_count

    response_data = await make_prime_request("GET", "availability/", params=params)

    if not response_data:
        return {"error": "Unable to fetch GPU availability"}

    return response_data


async def check_cluster_availability(
    regions: Optional[list[str]] = None,
    gpu_count: Optional[int] = None,
    gpu_type: Optional[str] = None,
    socket: Optional[str] = None,
    security: Optional[str] = None,
) -> dict[str, Any]:
    """Check cluster availability for multi-node deployments.

    Args:
        regions: List of regions to filter (options: "africa", "asia_south", "asia_northeast",
            "australia", "canada", "eu_east", "eu_north", "eu_west", "middle_east",
            "south_america", "united_states")
        gpu_count: Desired number of GPUs
        gpu_type: GPU model (e.g., "H100_80GB", "A100_80GB", "RTX4090_24GB", etc.)
        socket: Socket for selected GPU model ("PCIe", "SXM2", "SXM3", "SXM4", "SXM5", "SXM6")
        security: Security type ("secure_cloud", "community_cloud")

    Returns:
        Available cluster configurations grouped by GPU type
    """
    params = {}
    if regions:
        params["regions"] = regions
    if gpu_count:
        params["gpu_count"] = gpu_count
    if gpu_type:
        params["gpu_type"] = gpu_type
    if socket:
        params["socket"] = socket
    if security:
        params["security"] = security

    response_data = await make_prime_request("GET", "availability/clusters", params=params)

    if not response_data:
        return {"error": "Unable to fetch cluster availability"}

    return response_data
