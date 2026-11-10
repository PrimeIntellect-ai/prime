"""Dedicated inference deployments API client (`prime inference deployments`)."""

from typing import Any, Dict, Optional

from prime_cli.core import APIClient


class InferenceDeploymentsClient:
    """Client for the /inference/deployments routes; responses stay raw for --json."""

    def __init__(self, client: APIClient) -> None:
        self.client = client

    def get_deployment(self, deployment_id: str) -> Dict[str, Any]:
        return self.client.get(f"/inference/deployments/{deployment_id}")

    def list_deployments(self, team_id: Optional[str]) -> Dict[str, Any]:
        params = {"teamId": team_id} if team_id else None
        return self.client.get("/inference/deployments", params=params)

    def scale_deployment(self, deployment_id: str, replicas: int) -> Dict[str, Any]:
        return self.client.post(
            f"/inference/deployments/{deployment_id}/scale", json={"replicas": replicas}
        )
