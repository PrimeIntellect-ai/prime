import time
from pydantic import model_validator
from pydantic_config import BaseConfig
import torch.nn as nn
from zeroband.comms import ElasticDeviceMesh
import torch.distributed as dist
from zeroband.collectives import Compression, all_reduce
from torch.distributed._tensor.api import DTensor

from zeroband.utils.logging import get_logger
from zeroband.utils.world_info import get_world_info


class GlobalDDPConfig(BaseConfig):
    retry_all_reduce: int = 3
    compression: Compression = Compression.NO
    dpu: bool = False
    enable: bool = True

    @model_validator(mode="after")
    def validate_dpu(self):
        if self.dpu:
            raise NotImplementedError("DPU is not implemented yet")

        return self


class GlobalDDP:
    """
    This class implements DDP over internet. It

    :Args:
        model: The model to be trained
        elastic_device_mesh: The elastic device mesh to be used
        dpu: Whether to use delayed parameter updates

    Example usage:

    ```
    global_ddp = GlobalDDP(model, elastic_device_mesh)

    for step in range(num_steps):
        for micro_bs in range(num_micro_bs):
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()

        global_ddp.all_reduce()
        optimizer.step()
        diloco.step(model)
    ```

    """

    flag: str = "global_ddp"

    def __init__(
        self,
        config: GlobalDDPConfig,
        elastic_device_mesh: ElasticDeviceMesh,
    ):
        self.elastic_device_mesh = elastic_device_mesh
        self.config = config

        self._logger = get_logger()
        self.world_info = get_world_info()

    def all_reduce(self, model: nn.Module):
        _start_time = time.perf_counter()

        self.elastic_device_mesh.maybe_reinit_global_pg(admit_joiners=False)
        world_size = self.elastic_device_mesh.global_pg.size()

        self._logger.debug("sync pseudo gradient with world size %d", world_size)

        global_pg = self.elastic_device_mesh.global_pg

        for i in range(self.config.retry_all_reduce):
            try:
                _collective_start_time = time.perf_counter()
                self._logger.debug("Waiting on barrier")
                self.elastic_device_mesh.monitored_barrier(self.flag)

                self._logger.debug("Beginning all reduce")

                total_param = len(list(model.parameters()))
                for j, param in enumerate(model.parameters()):
                    t0 = time.perf_counter()
                    if isinstance(param.grad, DTensor):
                        grad = param.grad.to_local()
                    else:
                        grad = param

                    grad.div_(world_size)

                    all_reduce(self.config.compression, grad, dist.ReduceOp.SUM, global_pg)
                    self._logger.debug(
                        f"{j}/{total_param} all reduce bucket done in {time.perf_counter() - t0:.6f} seconds, numel: {grad.numel()}"
                    )
                break
            except Exception as e:
                self._logger.error(f"Error syncing pseudo gradient: {e}, retry {i+1}/{self.config.retry_all_reduce}")
                global_pg = self.elastic_device_mesh.get_global_pg(maybe_reinit=True)
        else:
            self._logger.error(
                "Failed to sync pseudo gradient after %d retries. Resorting to calculating pseudo-gradient without reduce",
                self.config.retry_all_reduce,
            )

        self._logger.info(f"Global gradient all reduce done in {time.perf_counter() - _start_time:.6f} seconds")
