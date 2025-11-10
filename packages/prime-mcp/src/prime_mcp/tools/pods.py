from typing import Any, Optional

from prime_mcp.client import make_prime_request


async def create_pod(
    cloud_id: str,
    gpu_type: str,
    provider_type: str,
    name: Optional[str] = None,
    gpu_count: int = 1,
    socket: str = "PCIe",
    disk_size: Optional[int] = None,
    vcpus: Optional[int] = None,
    memory: Optional[int] = None,
    max_price: Optional[float] = None,
    image: str = "ubuntu_22_cuda_12",
    custom_template_id: Optional[str] = None,
    data_center_id: Optional[str] = None,
    country: Optional[str] = None,
    security: Optional[str] = None,
    auto_restart: Optional[bool] = None,
    jupyter_password: Optional[str] = None,
    env_vars: Optional[dict[str, str]] = None,
    team_id: Optional[str] = None,
) -> dict[str, Any]:
    """Create a new GPU pod (compute instance).

    Args:
        cloud_id: Required cloud provider ID from availability check
        gpu_type: GPU model name
        provider_type: Provider type (e.g., "runpod", "fluidstack", etc.)
        name: Name for the pod (optional)
        gpu_count: Number of GPUs (default: 1, must be > 0)
        socket: GPU socket type (default: "PCIe")
        disk_size: Disk size in GB (must be > 0 if specified)
        vcpus: Number of virtual CPUs (must be > 0 if specified)
        memory: Memory in GB (must be > 0 if specified)
        max_price: Maximum price per hour as float
        image: Environment image (default: "ubuntu_22_cuda_12")
        custom_template_id: Custom template ID if using custom_template image
        data_center_id: Specific data center ID (required for some providers)
        country: Country code for filtering
        security: Security level ("secure_cloud", "community_cloud")
        auto_restart: Auto-restart on failure (default: False)
        jupyter_password: Password for Jupyter notebook access
        env_vars: Environment variables as key-value pairs
        team_id: Team ID for organization accounts

    Returns:
        Created pod details
    """
    # Validate required parameters
    if gpu_count <= 0:
        return {"error": "gpu_count must be greater than 0"}

    if disk_size is not None and disk_size <= 0:
        return {"error": "disk_size must be greater than 0 if specified"}

    if vcpus is not None and vcpus <= 0:
        return {"error": "vcpus must be greater than 0 if specified"}

    if memory is not None and memory <= 0:
        return {"error": "memory must be greater than 0 if specified"}

    # Build request body according to API specification
    request_body = {
        "pod": {
            "cloudId": cloud_id,
            "gpuType": gpu_type,
            "gpuCount": gpu_count,
            "socket": socket,
            "image": image,
        },
        "provider": {"type": provider_type},
    }

    # Add optional pod fields
    if name:
        request_body["pod"]["name"] = name
    if data_center_id:
        request_body["pod"]["dataCenterId"] = data_center_id
    if disk_size is not None:
        request_body["pod"]["diskSize"] = disk_size
    if vcpus is not None:
        request_body["pod"]["vcpus"] = vcpus
    if memory is not None:
        request_body["pod"]["memory"] = memory
    if max_price is not None:
        request_body["pod"]["maxPrice"] = max_price
    if country:
        request_body["pod"]["country"] = country
    if security:
        request_body["pod"]["security"] = security
    if auto_restart is not None:
        request_body["pod"]["autoRestart"] = auto_restart
    if jupyter_password:
        request_body["pod"]["jupyterPassword"] = jupyter_password
    if custom_template_id:
        request_body["pod"]["customTemplateId"] = custom_template_id
    if env_vars:
        request_body["pod"]["envVars"] = [{"key": k, "value": v} for k, v in env_vars.items()]

    # Add team if specified
    if team_id:
        request_body["team"] = {"teamId": team_id}

    response_data = await make_prime_request("POST", "pods/", json_data=request_body)

    if not response_data:
        return {"error": "Unable to create pod"}

    return response_data


async def list_pods(offset: int = 0, limit: int = 100) -> dict[str, Any]:
    """List all pods in your account.

    Args:
        offset: Number of pods to skip for pagination (default: 0, min: 0)
        limit: Maximum number of pods to return (default: 100, min: 0)

    Returns:
        Response containing list of pods with their details
    """
    params = {
        "offset": max(0, offset),
        "limit": max(0, limit),
    }

    response_data = await make_prime_request("GET", "pods/", params=params)

    if not response_data:
        return {"error": "Unable to fetch pods list"}

    return response_data


async def get_pods_history(
    limit: int = 100,
    offset: int = 0,
    sort_by: str = "terminatedAt",
    sort_order: str = "desc",
) -> dict[str, Any]:
    """Get pods history with sorting and pagination options.

    Args:
        limit: Maximum number of history entries to return (default: 100, min: 0)
        offset: Number of entries to skip for pagination (default: 0, min: 0)
        sort_by: Field to sort by (default: "terminatedAt", options: "terminatedAt", "createdAt")
        sort_order: Sort order (default: "desc", options: "asc", "desc")

    Returns:
        Response containing historical pod data
    """
    params = {
        "limit": max(0, limit),
        "offset": max(0, offset),
        "sort_by": sort_by,
        "sort_order": sort_order,
    }

    response_data = await make_prime_request("GET", "pods/history", params=params)

    if not response_data:
        return {"error": "Unable to fetch pods history"}

    return response_data


async def get_pods_status(pod_ids: Optional[list[str]] = None) -> dict[str, Any]:
    """Get pods status information.

    Args:
        pod_ids: List of specific pod IDs to get status for (optional)

    Returns:
        Response containing pod status information
    """
    params = {}
    if pod_ids:
        params["pod_ids"] = pod_ids

    response_data = await make_prime_request("GET", "pods/status", params=params)

    if not response_data:
        return {"error": "Unable to fetch pods status"}

    return response_data


async def get_pod_details(pod_id: str) -> dict[str, Any]:
    """Get detailed information about a specific pod.

    Args:
        pod_id: Unique identifier of the pod

    Returns:
        Detailed pod information
    """
    response_data = await make_prime_request("GET", f"pods/{pod_id}")

    if not response_data:
        return {"error": f"Unable to fetch details for pod ID: {pod_id}"}

    return response_data


async def delete_pod(pod_id: str) -> dict[str, Any]:
    """Delete/terminate a pod.

    Args:
        pod_id: Unique identifier of the pod to delete

    Returns:
        Pod deletion response with status
    """
    response_data = await make_prime_request("DELETE", f"pods/{pod_id}")

    if not response_data:
        return {"error": f"Unable to delete pod ID: {pod_id}"}

    return response_data
