"""Per-node metrics for dedicated Model Factory clusters.

Contract: ``GET /api/v1/model-factory/clusters/{cluster_id}/metrics/nodes``.
Every metric name below is used verbatim as both the ``current`` field and
the ``series`` key. A series key is absent when the metric has no data;
when present it covers the whole window on the ``step_seconds`` grid, with
``value: None`` for a gap.

Every model ignores unknown keys, so newer servers can add metrics without
breaking older CLIs; ``--json`` still echoes them from the raw response.
"""

from typing import Any, Dict, List, Optional
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, PrivateAttr, ValidationError

from prime_cli.core import APIClient, APIError

from .factory import AwareUTCDatetime, _format_validation_error

METRICS_WINDOWS = ("1h", "6h", "24h", "7d")
SUPPORTED_METRICS_SCHEMA_VERSION = 1


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore")


class MetricPoint(_Model):
    """One sample of a time series: unix seconds and value (``None`` is a gap)."""

    timestamp: int
    value: Optional[float] = None


class MetricValues(_Model):
    """Latest values for a node or a single GPU. ``None`` means unobserved.

    Node-level GPU values aggregate over the node's GPUs: utilization is the
    mean, memory (used and total) and power are summed, temperature is the
    hottest GPU. Host fields (cpu, memory, network) are always ``None`` on a
    GPU. InfiniBand fields sum the node's IB/RDMA ports; the error rate counts
    receive, symbol and link-down errors. On an IB port only ``ib_*`` is set.
    """

    gpu_util_percent: Optional[float] = None
    gpu_memory_used_bytes: Optional[int] = None
    gpu_memory_total_bytes: Optional[int] = None
    gpu_temperature_celsius: Optional[float] = None
    gpu_power_watts: Optional[float] = None
    cpu_util_percent: Optional[float] = None
    memory_used_bytes: Optional[int] = None
    memory_total_bytes: Optional[int] = None
    network_rx_bytes_per_second: Optional[float] = None
    network_tx_bytes_per_second: Optional[float] = None
    ib_rx_bytes_per_second: Optional[float] = None
    ib_tx_bytes_per_second: Optional[float] = None
    ib_errors_per_second: Optional[float] = None


class GpuMetrics(_Model):
    """One GPU on a node; ``series`` is keyed by ``MetricValues`` field names."""

    index: int
    current: MetricValues = MetricValues()
    series: Dict[str, List[MetricPoint]] = {}


class IbPortMetrics(_Model):
    """One IB/RDMA port (e.g. ``mlx5_0`` port ``1``).

    Link status and the window's error counts are listed for every node;
    ``current`` and ``series`` are filled only with ``?node=``.
    ``rate_bytes_per_second`` is the negotiated link rate.
    """

    device: str
    port: str
    state: Optional[str] = None
    physical_state: Optional[str] = None
    rate_bytes_per_second: Optional[float] = None
    symbol_errors: Optional[int] = None
    link_downed: Optional[int] = None
    receive_errors: Optional[int] = None
    current: MetricValues = MetricValues()
    series: Dict[str, List[MetricPoint]] = {}


class NodeMetrics(_Model):
    """One node's current values and series.

    ``gpus`` is filled only when the request names this node (``?node=``).
    """

    name: str
    gpu_type: Optional[str] = None
    gpu_count: Optional[int] = None
    state: str
    current: MetricValues = MetricValues()
    series: Dict[str, List[MetricPoint]] = {}
    gpus: List[GpuMetrics] = []
    ib_ports: List[IbPortMetrics] = []


class ClusterNodeMetrics(_Model):
    """Response envelope for the per-node metrics endpoint.

    ``available`` is False when the cluster exists and the caller may see it
    but no metrics can be served (no exporter data, store unreachable);
    ``unavailable_reason`` then carries a short user-facing sentence and
    ``unavailable_code`` one of ``no_cluster_label`` / ``no_data``. An
    unreachable metrics store is a 503, not ``available=False``.
    """

    schema_version: int
    cluster_id: str
    cluster_display_name: str
    window: str
    step_seconds: int
    as_of: AwareUTCDatetime = None
    available: bool
    unavailable_reason: Optional[str] = None
    unavailable_code: Optional[str] = None
    nodes: List[NodeMetrics]

    _raw_response: Dict[str, Any] = PrivateAttr(default_factory=dict)

    @property
    def raw_response(self) -> Dict[str, Any]:
        return self._raw_response


class FactoryMetricsClient:
    """Client for Model Factory cluster metrics."""

    def __init__(self, client: APIClient) -> None:
        self.client = client

    def get_node_metrics(
        self,
        cluster_id: str,
        window: str = "1h",
        node: Optional[str] = None,
        current_only: bool = False,
    ) -> ClusterNodeMetrics:
        """Fetch per-node metrics for one dedicated cluster.

        With ``node``, the response holds only that node, with its per-GPU
        breakdown. ``current_only`` tells the server to skip the range queries
        the series are built from, which is most of the work. Only members of
        the cluster's team can read its metrics; any other cluster id, or a
        node outside the cluster, is a 404.
        """
        if window not in METRICS_WINDOWS:
            raise ValueError(f"window must be one of {', '.join(METRICS_WINDOWS)}")
        endpoint = f"/model-factory/clusters/{quote(cluster_id, safe='')}/metrics/nodes"
        try:
            params: Dict[str, Any] = {"window": window}
            if node is not None:
                params["node"] = node
            if current_only:
                params["current_only"] = "true"
            response = self.client.get(endpoint, params=params)
        except ValueError as exc:
            raise APIError("Cluster metrics API returned a malformed response body.") from exc
        try:
            metrics = ClusterNodeMetrics.model_validate(response)
        except ValidationError as exc:
            raise APIError(_format_validation_error(exc, label="cluster metrics")) from exc
        if metrics.schema_version != SUPPORTED_METRICS_SCHEMA_VERSION:
            raise APIError(
                f"Unsupported cluster metrics schema version: {metrics.schema_version} "
                f"(expected {SUPPORTED_METRICS_SCHEMA_VERSION})"
            )
        metrics._raw_response = response
        return metrics
