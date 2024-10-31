import re
import time
from pydantic_config import BaseConfig
import torch
from torch import nn
from zeroband import utils
from zeroband.collectives import Compression, all_reduce
from zeroband.comms import ElasticDeviceMesh
from zeroband.utils.world_info import get_world_info
from zeroband.utils.logging import get_logger
import torch.distributed as dist
from torch.distributed._tensor.api import DTensor
from functools import lru_cache


class DilocoConfig(BaseConfig):
    outer_lr: float = 0.7
    inner_steps: int
    compression: Compression = Compression.NO

    retry_all_reduce: int = 3


@lru_cache(maxsize=None)
def _find_first_number(s: str) -> int:
    match = re.search(r"\d+", s)
    if match:
        return int(match.group())
    else:
        return -1


class Diloco:
    """
    This class implements the diloco algorithm from  https://arxiv.org/abs/2311.08105 and https://arxiv.org/abs/2407.07852.

    It handles the outer loop as well as the inter node communication.

    There is no VRAM overhead with this implementation as the model is outer optimizer is offloaded to cpu.
    All reduce communication are also done on cpu using GLOO.

    Example usage:

    # Example usage in a training loop:

    diloco = Diloco(config.diloco, model, elastic_device_mesh)

    for outer_step in range(num_outer_steps):
        for inner_step in range(config.diloco.inner_steps):
            # Regular inner training loop
                optimizer.zero_grad()
                loss = model(batch)
                loss.backward()
                optimizer.step()

        diloco.step(model)
    """

    def __init__(
        self,
        config: DilocoConfig,
        model: nn.Module,
        elastic_device_mesh: ElasticDeviceMesh,
    ):
        self.config = config

        if config.compression == Compression.UINT8:
            from zeroband.C.collectives import ring_allreduce as _  # noqa: F401
            # just force compilation

        self.elastic_device_mesh = elastic_device_mesh

        self._logger = get_logger()
        self.world_info = get_world_info()

        self._init_offloaded_optimizer(model=model)

    @torch.no_grad()
    def _init_offloaded_optimizer(self, model):
        self.param_list_cpu = self.get_offloaded_param(model)
        self.outer_optimizer = torch.optim.SGD(
            self.param_list_cpu, lr=self.config.outer_lr, momentum=0.9, nesterov=True
        )
        self._logger.debug("offload model to cpu")

    @torch.no_grad()
    def sync_pseudo_gradient(
        self, model: nn.Module, fake: bool = False, flag: str = "outer", num_effective_peers: int | None = None
    ):
        """
        Sync the pseudo gradient from the local process group to the global process group
        """
        world_size_pre_init = self.elastic_device_mesh.global_pg.size()
        self.elastic_device_mesh.maybe_reinit_global_pg(admit_joiners=False)
        world_size_post_init = self.elastic_device_mesh.global_pg.size()

        if world_size_pre_init == world_size_post_init and num_effective_peers is not None:
            world_size = num_effective_peers
        else:
            world_size = world_size_post_init

        self._logger.debug("sync pseudo gradient %s with world size %d", " fake" if fake else "", world_size)

        global_pg = self.elastic_device_mesh.global_pg
        for i in range(self.config.retry_all_reduce):
            with utils.timer("compute pseudo gradient"):
                for param_offloaded, param in zip(self.param_list_cpu, model.parameters()):
                    if fake:
                        param_offloaded.grad.to_local().zero_()
                    else:
                        param_offloaded.grad.to_local().copy_(param_offloaded.data.to_local())
                        param_offloaded.grad.to_local().sub_(param.data.to_local().to(param_offloaded.data.device))
            try:
                # self.offloaded_grad_flat_tensor.div_(world_size)
                _collective_start_time = time.perf_counter()
                self._logger.debug("Waiting on barrier")
                self.elastic_device_mesh.monitored_barrier(flag)

                self._logger.debug("Beginning all reduce")
                # all_reduce(self.config.compression, self.offloaded_grad_flat_tensor, dist.ReduceOp.SUM, global_pg)
                for j, param in enumerate(self.param_list_cpu):
                    grad = param.grad

                    t0 = time.perf_counter()

                    if self.config.compression == Compression.FP16:
                        grad = grad.half()

                    grad.div_(world_size)

                    all_reduce(self.config.compression, grad, dist.ReduceOp.SUM, global_pg)
                    param.grad.copy_(grad)

                    self._logger.debug(
                        f"{j}/{len(self.param_list_cpu)} all reduce bucket done in {time.perf_counter() - t0:.6f} seconds, numel: {grad.numel()}"
                    )

                self._logger.debug(
                    f"All reduce takes {time.perf_counter() - _collective_start_time:.6f} seconds numels: {self.offloaded_grad_flat_tensor.numel()}"
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

            with utils.timer("update gradient"):
                for param_offloaded, param in zip(self.param_list_cpu, model.parameters()):
                    if fake:
                        param_offloaded.grad.to_local().zero_()
                    else:
                        param_offloaded.grad.to_local().copy_(param_offloaded.data.to_local())
                        param_offloaded.grad.to_local().sub_(param.data.to_local().to(param_offloaded.data.device))

    @torch.no_grad()
    def sync_inner_model(self, model: nn.Module):
        """
        Sync the inner model from the CPU outer model to GPU
        """

        self._logger.debug("sync inner model")
        for param_offloaded, param in zip(self.param_list_cpu, model.parameters()):
            param.data.to_local().copy_(param_offloaded.data.to_local())

    @torch.no_grad()
    def get_offloaded_param(self, model: nn.Module) -> list[nn.Parameter]:
        """
        Offload the model parameters to cpu
        """

        offloaded_params = []
        for param in model.parameters():
            offloaded_param = nn.Parameter(
                DTensor.from_local(
                    param.data.to_local(),
                    device_mesh=self.elastic_device_mesh.cpu_local_mesh,
                    placements=param.data.placements,
                )
            )

            offloaded_param.grad = DTensor.from_local(
                torch.zeros_like(param.data.to_local()),
                device_mesh=self.elastic_device_mesh.cpu_local_mesh,
                placements=param.data.placements,
            )

            # here we pre-allocate the grad DTensor on cpu.
            offloaded_param.requires_grad = True
            offloaded_params.append(offloaded_param)

        return offloaded_params

    @torch.no_grad()
    def step(self, model: nn.Module, fake: bool = False, num_effective_peers: int | None = None, flag: str = "outer"):
        """
        Step the optimizer
        """
        with utils.timer("sync pseudo gradient"):
            self.sync_pseudo_gradient(model, fake=fake, flag=flag, num_effective_peers=num_effective_peers)

        with utils.timer("outer optimizer step"):
            if self.outer_optimizer is not None:
                self.outer_optimizer.step()

        with utils.timer("sync inner model"):
            self.sync_inner_model(model)
