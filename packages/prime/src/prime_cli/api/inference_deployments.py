"""Inference deployment API client (`/api/v1/inference`)."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from prime_cli.core import APIClient


class InferenceCatalogModel(BaseModel):
    id: str
    slug: str
    repo: str
    revision: str
    display_name: str = Field(..., alias="displayName")
    quant: Optional[str] = None
    # Raw nvidia.com/gpu.product label; None means any GPU.
    gpu_product: Optional[str] = Field(None, alias="gpuProduct")
    min_gpus: int = Field(1, alias="minGpus")
    max_seq_len: Optional[int] = Field(None, alias="maxSeqLen")
    enabled: bool
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True)


class InferenceClusterCapability(BaseModel):
    prime_cluster_id: str = Field(..., alias="primeClusterId")
    name: str
    display_name: Optional[str] = Field(None, alias="displayName")
    inference_enabled: bool = Field(..., alias="inferenceEnabled")
    dynamo_profile: Optional[str] = Field(None, alias="dynamoProfile")
    gpu_products: List[str] = Field(default_factory=list, alias="gpuProducts")
    # Internal tailscale address: only used to derive readiness, never displayed.
    gateway_endpoint: Optional[str] = Field(None, alias="gatewayEndpoint")

    model_config = ConfigDict(populate_by_name=True)

    @property
    def ready(self) -> bool:
        return self.inference_enabled and bool(self.gateway_endpoint)


class InferenceCatalog(BaseModel):
    models: List[InferenceCatalogModel]
    clusters: List[InferenceClusterCapability]


class InferenceDeployment(BaseModel):
    """A dedicated inference deployment of a catalog model or a training checkpoint."""

    id: str
    deployment_name: Optional[str] = Field(None, alias="deploymentName")
    status: str
    status_phase: Optional[str] = Field(None, alias="statusPhase")
    error_message: Optional[str] = Field(None, alias="errorMessage")
    served_model_alias: Optional[str] = Field(None, alias="servedModelAlias")
    # OpenAI-compatible base URL; set once the deployment is RUNNING.
    endpoint_url: Optional[str] = Field(None, alias="endpointUrl")
    checkpoint: Optional[Any] = None
    is_on_demand: Optional[bool] = Field(None, alias="isOnDemand")
    created_at: Optional[str] = Field(None, alias="createdAt")
    dashboard_url: Optional[str] = Field(None, alias="dashboardUrl")

    model_config = ConfigDict(populate_by_name=True, extra="allow")


class InferenceDeploymentsClient:
    def __init__(self, client: APIClient) -> None:
        self.client = client

    def get_catalog(self, team_id: Optional[str] = None) -> Dict[str, Any]:
        params = {"teamId": team_id} if team_id else None
        return self.client.get("/inference/catalog", params=params)

    def deploy(
        self,
        model_catalog_id: str,
        team_id: Optional[str] = None,
        deployment_name: Optional[str] = None,
        replicas: int = 1,
        prime_cluster_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"modelCatalogId": model_catalog_id, "replicas": replicas}
        if team_id:
            body["teamId"] = team_id
        if deployment_name:
            body["deploymentName"] = deployment_name
        if prime_cluster_id:
            body["primeClusterId"] = prime_cluster_id
        return self.client.post("/inference/deploy", json=body)

    def deploy_checkpoint(
        self, run_id: str, step: int, team_id: Optional[str] = None
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"runId": run_id, "step": step}
        if team_id:
            body["teamId"] = team_id
        return self.client.post("/inference/checkpoint-deploy", json=body)

    def get_deployment(self, deployment_id: str) -> Dict[str, Any]:
        return self.client.get(f"/inference/deployments/{deployment_id}")
