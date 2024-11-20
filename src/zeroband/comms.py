import os
import torch.distributed as dist
from datetime import timedelta

TCPSTORE_TIMEOUT = timedelta(seconds=int(os.getenv("ZERO_BAND_GLOBAL_STORE_TIMEOUT_SECONDS", "300")))
TCPSTORE_POLLING_INTERVAL = float(os.getenv("ZERO_BAND_GLOBAL_STORE_POLLING_INTERVAL_SECONDS", "0.1"))
GLOBAL_PG_TIMEOUT = timedelta(seconds=int(os.getenv("ZERO_BAND_GLOBAL_PG_TIMEOUT_SECONDS", "600")))
MAX_JOINERS = 100  # Maximum number of nodes that can join in a single reinit
HEARTBEAT_INTERVAL = int(
    os.getenv("ZERO_BAND_EDM_HEARTBEAT_INTERVAL_SECONDS", "2")
)  # Interval in seconds between heartbeats
HEARTBEAT_TIMEOUT = int(
    os.getenv("ZERO_BAND_EDM_HEARTBEAT_TIMEOUT_SECONDS", "10")
)  # Time in seconds after which a node is considered dead if no heartbeat is received
IPERF_PORT = int(os.getenv("ZERO_BAND_IPERF_PORT", "10101"))
IPERF_IFNAME = os.getenv("GLOO_SOCKET_IFNAME", "eth0")
BENCH_TENSOR_SIZE = 1_000_000


class ElasticDeviceMesh:
    """A class to manage the process groups for elastic training without restarts."""

    local_pg: dist.ProcessGroup
    global_pg: dist.ProcessGroup

    def __init__(self):
        # Initialize local process group
        dist.init_process_group()
        self.local_pg = dist.get_default_group()

        self.global_pccl_communicator = ...
