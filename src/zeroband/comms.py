import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh


class ElasticDeviceMesh:
    """A class to manage the process groups for elastic training without restarts."""

    local_pg: dist.ProcessGroup

    def __init__(self):
        # Initialize local process group
        dist.init_process_group()
        self.local_pg = dist.get_default_group()
        self.cuda_local_mesh = init_device_mesh("cuda", mesh_shape=(self.local_pg.size(),))

        self.global_pccl_communicator = ...
