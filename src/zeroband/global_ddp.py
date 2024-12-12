import time
from typing import Generator, NamedTuple
from pydantic import model_validator
from pydantic_config import BaseConfig
import torch
import torch.nn as nn
from zeroband.comms import ElasticDeviceMesh
import torch.distributed as dist
from zeroband.collectives import Compression, gloo_all_reduce
from torch.distributed._tensor.api import DTensor
from zeroband.utils.logging import get_logger
from zeroband.utils.world_info import get_world_info

from torch.distributed import Work


class GlobalDDPConfig(BaseConfig):
    # retry_all_reduce: int = 3
    compression: Compression = Compression.NO
    dpu: bool = False
    enable: bool = True

    @model_validator(mode="after")
    def validate_compression(self):
        if self.compression != Compression.NO:
            raise NotImplementedError("Compression is not implemented yet")
        return self


def offload_grad_generator(model: nn.Module) -> Generator:
    for param in model.parameters():
        if param.grad is not None:
            if isinstance(param.grad, DTensor):
                yield param.grad.to_local().to("cpu")
            else:
                yield param.grad.to("cpu")


def apply_staling_grad(model: nn.Module, tensors: list[torch.Tensor]):
    for param, tensor in zip(model.parameters(), tensors):
        if isinstance(param.grad, DTensor):
            param.grad.to_local().copy_(tensor)
        else:
            param.grad.copy_(tensor)


def maybe_unwrap_dtensor(tensor: torch.Tensor | DTensor):
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    else:
        return tensor


class AllReduceGradWork(NamedTuple):
    grad: torch.Tensor
    work: Work


class GlobalDDP:
    """
    This class implements DDP over internet. It

    :Args:
        model: The model to be trained
        config: The configuration for the global DDP
        elastic_device_mesh: The elastic device mesh to be used

    Example usage:

    ```
    config = GlobalDDPConfig(dpu=False)
    global_ddp = GlobalDDP(model, config, elastic_device_mesh)

    for step in range(num_steps):
        for micro_bs in range(num_micro_bs):
            loss = model(batch)
            loss.backward()

        global_ddp.all_reduce()
        optimizer.step()
        optimizer.zero_grad()
    ```

    """

    flag: str = "global_ddp"

    def __init__(
        self,
        model: nn.Module,
        config: GlobalDDPConfig,
        elastic_device_mesh: ElasticDeviceMesh,
    ):
        self.elastic_device_mesh = elastic_device_mesh
        self.config = config

        self.world_info = get_world_info()
        self._logger = get_logger()

        self.model = model

        self._stalling_grad_work: list[AllReduceGradWork] | None = None

    def all_reduce(self):
        if not self.config.dpu:
            self._blocking_all_reduce(self.model)
        else:
            new_staling_grad_work = self._async_all_reduce(self.model)

            if self._stalling_grad_work is None:
                # if it is the first step we just store the work for the next call to this function and return
                self._stalling_grad_work = new_staling_grad_work
            else:
                # otherwise we wait for the current staling grad work to finish
                start_time = time.time()
                [all_reduce_grad_work.work.wait() for all_reduce_grad_work in self._stalling_grad_work]
                self._logger.debug(f"Time to wait for staling grads: {time.time() - start_time}")
                # and apply the staling grads to the model
                apply_staling_grad(
                    self.model, [all_reduce_grad_work.grad for all_reduce_grad_work in self._stalling_grad_work]
                )
                # and store the new staling grad work for the next call to this function
                self._stalling_grad_work = new_staling_grad_work

    def _async_all_reduce(self, model: nn.Module) -> list[AllReduceGradWork]:
        """
        Triggered all reduce operation on a list of tensors in a async manner.
        Return a list of async jobs that can be waited on.
        """

        self.elastic_device_mesh.maybe_reinit_global_pg(admit_joiners=False)
        world_size = self.elastic_device_mesh.global_pg.size()

        global_pg = self.elastic_device_mesh.global_pg
        self.elastic_device_mesh.monitored_barrier(self.flag)
        self._logger.debug("Beginning all reduce")

        async_job = []

        for param in offload_grad_generator(model):  # TODO: do we need to offload when doing blocking all reduce ?
            grad = maybe_unwrap_dtensor(param)

            grad.div_(world_size)

            # all_reduce(self.config.compression, grad, dist.ReduceOp.SUM, global_pg) # doing gloo all reduce direclty because of async op

            async_job.append(AllReduceGradWork(grad, gloo_all_reduce(grad, dist.ReduceOp.SUM, global_pg, True)))

        return async_job

    def _blocking_all_reduce(self, tensor: list[torch.Tensor]):
        """
        Triggered all reduce operation on a list of tensors in a blocking manner.
        """
        [all_reduce_grad_work.work.wait() for all_reduce_grad_work in self._async_all_reduce(tensor)]
