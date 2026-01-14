"""API client for GitHub Actions runners."""

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from prime_cli.core import APIClient, APIError

from .pods import Pod, PodList, PodsClient


class Runner(BaseModel):
    """Represents a GitHub Actions runner running on a pod."""

    pod_id: str = Field(..., alias="podId")
    pod_name: Optional[str] = Field(None, alias="podName")
    repo: Optional[str] = None
    org: Optional[str] = None
    status: str
    gpu_type: str = Field(..., alias="gpuType")
    gpu_count: int = Field(..., alias="gpuCount")
    labels: List[str] = Field(default_factory=list)
    created_at: str = Field(..., alias="createdAt")

    model_config = ConfigDict(populate_by_name=True)


class RunnerConfig(BaseModel):
    """Configuration for creating a GitHub Actions runner."""

    repo: Optional[str] = None
    org: Optional[str] = None
    token: str
    labels: List[str] = Field(default_factory=list)
    name: Optional[str] = None
    gpu_type: str = Field(default="A100", alias="gpuType")
    gpu_count: int = Field(default=1, alias="gpuCount")
    disk_size: int = Field(default=100, alias="diskSize")

    model_config = ConfigDict(populate_by_name=True)


class RunnersClient:
    """Client for managing GitHub Actions runners on pods."""

    # Tag used to identify runner pods
    RUNNER_TAG = "github-actions-runner"

    def __init__(self, client: APIClient) -> None:
        self.client = client
        self.pods_client = PodsClient(client)

    def list_runner_pods(self, limit: int = 100) -> List[Pod]:
        """List all pods that are GitHub Actions runners.

        Filters pods by name prefix to identify runners.
        """
        try:
            pods_list = self.pods_client.list(limit=limit)
            # Filter pods that have the runner naming convention
            runner_pods = [
                pod for pod in pods_list.data if pod.name and pod.name.startswith("gha-runner-")
            ]
            return runner_pods
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to list runner pods: {e.response.text}")
            raise APIError(f"Failed to list runner pods: {str(e)}")

    def get_runner_pod(self, pod_id: str) -> Pod:
        """Get details of a specific runner pod."""
        return self.pods_client.get(pod_id)

    def create_runner_pod(
        self,
        cloud_id: str,
        gpu_type: str,
        socket: str,
        gpu_count: int,
        provider: str,
        name: str,
        disk_size: int = 100,
        vcpus: Optional[int] = None,
        memory: Optional[int] = None,
        image: str = "pytorch_24",
        data_center_id: Optional[str] = None,
        team_id: Optional[str] = None,
        runner_token: Optional[str] = None,
        runner_repo: Optional[str] = None,
        runner_org: Optional[str] = None,
        runner_labels: Optional[List[str]] = None,
    ) -> Pod:
        """Create a new pod configured as a GitHub Actions runner.

        Args:
            cloud_id: Cloud provider ID
            gpu_type: Type of GPU (e.g., 'A100_80GB')
            socket: Socket configuration
            gpu_count: Number of GPUs
            provider: Provider name (e.g., 'prime')
            name: Pod name (should start with 'gha-runner-')
            disk_size: Disk size in GB
            vcpus: Number of vCPUs
            memory: Memory in GB
            image: Base image to use
            data_center_id: Data center ID
            team_id: Team ID
            runner_token: GitHub Actions runner registration token
            runner_repo: Repository in format 'owner/repo' (for repo-level runner)
            runner_org: Organization name (for org-level runner)
            runner_labels: Additional labels for the runner
        """
        # Build environment variables for runner configuration
        env_vars: List[dict[str, Any]] = []

        if runner_token:
            env_vars.append({"key": "RUNNER_TOKEN", "value": runner_token})

        if runner_repo:
            env_vars.append({"key": "RUNNER_REPO", "value": runner_repo})

        if runner_org:
            env_vars.append({"key": "RUNNER_ORG", "value": runner_org})

        if runner_labels:
            env_vars.append({"key": "RUNNER_LABELS", "value": ",".join(runner_labels)})

        # Add runner name
        env_vars.append({"key": "RUNNER_NAME", "value": name})

        pod_config: dict[str, Any] = {
            "pod": {
                "name": name,
                "cloudId": cloud_id,
                "gpuType": gpu_type,
                "socket": socket,
                "gpuCount": gpu_count,
                "diskSize": disk_size,
                "vcpus": vcpus,
                "memory": memory,
                "image": image,
                "dataCenterId": data_center_id,
                "maxPrice": None,
                "country": None,
                "security": None,
                "jupyterPassword": None,
                "autoRestart": False,
                "customTemplateId": None,
                "envVars": env_vars,
            },
            "provider": {"type": provider},
            "disks": None,
            "team": {"teamId": team_id} if team_id else None,
        }

        return self.pods_client.create(pod_config)

    def terminate_runner(self, pod_id: str) -> None:
        """Terminate a runner pod."""
        self.pods_client.delete(pod_id)
