"""Inference deployment API client (`/api/v1/inference`)."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from prime_cli.core import APIClient


class InferenceEngine(BaseModel):
    """One way to run a model: this many GPUs of this product."""

    gpu_product: str = Field(..., alias="gpuProduct")
    gpus: int

    model_config = ConfigDict(populate_by_name=True)


class InferenceCatalogModel(BaseModel):
    id: str
    slug: str
    repo: str
    # Null for a model pinned to its repo's default branch.
    revision: Optional[str] = None
    display_name: str = Field(..., alias="displayName")
    quant: Optional[str] = None
    # Every GPU product the model runs on, in the operator's preference order.
    engines: List[InferenceEngine] = Field(default_factory=list)
    # The preferred engine, flattened by the API for pre-`engines` clients.
    gpu_product: Optional[str] = Field(None, alias="gpuProduct")
    min_gpus: int = Field(1, alias="minGpus")
    max_seq_len: Optional[int] = Field(None, alias="maxSeqLen")
    enabled: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True)


class InferenceClusterCapability(BaseModel):
    prime_cluster_id: str = Field(..., alias="primeClusterId")
    name: str
    display_name: Optional[str] = Field(None, alias="displayName")
    inference_enabled: bool = Field(False, alias="inferenceEnabled")
    dynamo_profile: Optional[str] = Field(None, alias="dynamoProfile")
    gpu_products: List[str] = Field(default_factory=list, alias="gpuProducts")
    # The shared gateway is up. The API reports readiness only; the gateway's
    # address is tailnet-internal and never leaves the backend.
    gateway_ready: bool = Field(False, alias="gatewayReady")
    # Transition shim: before platform#7676 the API sent the address instead of
    # the bit, so a CLI release reaching users ahead of that rollout would read
    # every cluster as not ready. Never rendered. Drop once prod has shipped.
    legacy_gateway_endpoint: Optional[str] = Field(None, alias="gatewayEndpoint")
    # Catalog model id -> GPUs per replica of the engine this cluster runs it
    # with. A model absent from the map is one this cluster cannot run.
    model_gpus: Dict[str, int] = Field(default_factory=dict, alias="modelGpus")

    model_config = ConfigDict(populate_by_name=True)

    @property
    def ready(self) -> bool:
        gateway_up = self.gateway_ready or bool(self.legacy_gateway_endpoint)
        return self.inference_enabled and gateway_up


class InferenceCatalog(BaseModel):
    models: List[InferenceCatalogModel]
    clusters: List[InferenceClusterCapability]


class CheckpointRef(BaseModel):
    """The training run checkpoint a deployment serves."""

    run_id: str = Field(..., alias="runId")
    step: int

    model_config = ConfigDict(populate_by_name=True)

    def __str__(self) -> str:
        return f"run {self.run_id} @ step {self.step}"


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
    checkpoint: Optional[CheckpointRef] = None
    is_on_demand: Optional[bool] = Field(None, alias="isOnDemand")
    # Manual-scaling read model. `replicas` is the desired count, not the Ready
    # one; `scale_state` tracks a pending change (idle/applying/blocked).
    replicas: Optional[int] = None
    gpus_per_replica: Optional[int] = Field(None, alias="gpusPerReplica")
    max_replicas: Optional[int] = Field(None, alias="maxReplicas")
    scale_state: Optional[str] = Field(None, alias="scaleState")
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

    def list_deployments(self, team_id: Optional[str] = None) -> Dict[str, Any]:
        """The team's deployments, or without `team_id` the caller's personal ones."""
        params = {"teamId": team_id} if team_id else None
        return self.client.get("/inference/deployments", params=params)

    def scale_deployment(self, deployment_id: str, replicas: int) -> Dict[str, Any]:
        return self.client.post(
            f"/inference/deployments/{deployment_id}/scale", json={"replicas": replicas}
        )
