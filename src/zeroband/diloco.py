import re
import time
from typing import List

import pccl
import torch
from pccl import Communicator, AsyncReduceHandle, ReduceOp, Attribute, ReduceOperandDescriptor, DistributionHint, \
    DataType, QuantizationOptions, QuantizationAlgorithm
from torch import nn
from torch.distributed import init_device_mesh

from zeroband.utils.world_info import get_local_world_info
from zeroband.utils.logger import get_logger
from zeroband.config import DilocoConfig, Compression
import torch.distributed as dist
from torch.distributed._tensor.api import DTensor
from functools import lru_cache


@lru_cache(maxsize=None)
def _find_first_number(s: str) -> int:
    match = re.search(r"\d+", s)
    if match:
        return int(match.group())
    else:
        return -1



def all_reduce_multiple_with_retry(communicator: Communicator,
                                   tensors: list[torch.Tensor],
                                   op: ReduceOp,
                                   compression: Compression,
                                   max_in_flight: int = 8):
    """
    Launches concurrent all-reduce operations on a list of tensors,
    waits for them all, and retries if a peer fails or the world size changes.
    Will attempt to target :param max_in_flight: concurrent all-reduce operations.
    The more similar your tensors are in size, the better this in flight system will work.
    Future versions of PCCL may provide a "wait for any of multiple async ops" api to improve this pattern.
    """
    world_size = communicator.get_attribute(Attribute.CURRENT_WORLD_SIZE)

    total_tx = 0
    total_rx = 0

    def launch_all_reduce(x: torch.Tensor, tag: int):
        op_desc = ReduceOperandDescriptor(
            datatype=DataType.FLOAT,
            distribution_hint=DistributionHint.NORMAL
        )
        if compression == Compression.NO:
            quant_desc = QuantizationOptions(
                quantized_datatype=DataType.FLOAT,
                algorithm=QuantizationAlgorithm.NONE
            )
        else:
            quant_desc = QuantizationOptions(
                quantized_datatype=DataType.UINT8,
                algorithm=QuantizationAlgorithm.MIN_MAX
            )

        return communicator.all_reduce_async(
            x, x,
            operand_descriptor=op_desc,
            quantization_options=quant_desc,
            op=op,
            tag=tag
        )

    handles = [None for _ in range(len(tensors))]
    done_handles = set()

    in_flight = 0
    for tensor_index in range(len(tensors)):
        dst_tensor = tensors[tensor_index]

        if in_flight >= max_in_flight:
            break

        handles[tensor_index] = launch_all_reduce(
            dst_tensor,
            tensor_index
        )
        in_flight += 1

    while world_size > 1:
        all_done = True
        for tensor_index in range(len(tensors)):
            handle = handles[tensor_index]
            dst_tensor = tensors[tensor_index]

            if handle is None:
                if tensor_index in done_handles:
                    continue

                if in_flight >= max_in_flight:
                    continue

                handle = handles[tensor_index] = launch_all_reduce(
                    dst_tensor,
                    tensor_index
                )
                in_flight += 1

            is_success, status, info = handle.wait()
            world_size = communicator.get_attribute(Attribute.CURRENT_WORLD_SIZE)
            if not is_success:
                print(f"Reduce failed: {status}; Starting recovery procedure")
                handles[tensor_index] = None
                # Wait for all ongoing ops to finish or fail before retry
                for j in range(len(tensors)):
                    if j == tensor_index:
                        continue
                    h_j = handles[j]
                    if h_j is not None:
                        s_j, _, _ = h_j.wait()
                        if s_j:
                            done_handles.add(j)
                        in_flight -= 1
                    handles[j] = None
                all_done = False
                break

            # success for this handle
            handles[tensor_index] = None
            done_handles.add(tensor_index)

            total_tx += info.tx_bytes
            total_rx += info.rx_bytes

            in_flight -= 1

        if all_done:
            break

    if world_size == 1:
        # If we are alone, just finalize all handles and return
        for h in handles:
            if h is not None:
                h.wait()
        return False

    return True


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
    ):
        self.config = config

        self._logger = get_logger()
        self.local_world_info = get_local_world_info()

        self.cuda_local_mesh = init_device_mesh("cuda", mesh_shape=(self.local_world_info.world_size,))
        self.cpu_local_mesh = init_device_mesh("cpu", mesh_shape=(self.local_world_info.world_size,))

        self._init_offloaded_optimizer(model=model)

    @torch.no_grad()
    def _init_offloaded_optimizer(self, model):
        self.param_list_cpu = self.get_offloaded_param(model)
        self.outer_optimizer = torch.optim.SGD(
            self.param_list_cpu, lr=self.config.outer_lr, momentum=0.9, nesterov=True
        )
        self._logger.debug("offload model to cpu")

    @torch.no_grad()
    def sync_pseudo_gradient(self, model: nn.Module, communicator: Communicator, fake: bool = False):
        """
        Sync the pseudo gradient from the local process group to the global process group
        """
        _start_time = time.perf_counter()

        self._logger.debug("sync pseudo gradient %s with world size %d", "fake" if fake else "", self.local_world_info.world_size)

        for param_offloaded, param in zip(self.param_list_cpu, model.parameters()):
            assert isinstance(param_offloaded.grad, DTensor)
            if fake:
                param_offloaded.grad.to_local().zero_()
            else:
                param_offloaded.grad.to_local().copy_(param_offloaded.data.to_local())
                param_offloaded.grad.to_local().sub_(param.data.to_local().to(param_offloaded.data.device))
        try:
            _collective_start_time = time.perf_counter()

            self._logger.debug("Beginning all reduce")
            reduce_tensors = [self.offloaded_grad_flat_tensor]
            for j, tensor_group in enumerate(self._offloaded_grad_grouped_tensor):
                t0 = time.perf_counter()
                reduce_tensors.append(tensor_group)
                self._logger.debug(
                    f"{j}/{len(self._offloaded_grad_grouped_tensor)} all reduce bucket done in {time.perf_counter() - t0:.6f} seconds, numel: {tensor_group.numel()}"
                )
            all_reduce_multiple_with_retry(communicator, reduce_tensors, ReduceOp.AVG, self.config.compression, max_in_flight=16)

            self._logger.debug(
                f"All reduce takes {time.perf_counter() - _collective_start_time:.6f} seconds numels: {self.offloaded_grad_flat_tensor.numel()}"
            )
        except Exception as e:
            self._logger.error(f"Error syncing pseudo gradient: {e}")

        self._logger.info(f"Sync psuedo-gradient in {time.perf_counter() - _start_time:.6f} seconds")

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
        param_items = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
        numels = sum(param.to_local().numel() for _, param in param_items)

        self.offloaded_data_flat_tensor = torch.empty((numels,), device="cpu", dtype=torch.float32)
        self.offloaded_grad_flat_tensor = torch.zeros((numels,), device="cpu", dtype=torch.float32)
        current_offset = 0
        offloaded_params = []
        param_group_cutoff = []

        prev_id = None
        for name, param in param_items:
            if _find_first_number(name) != prev_id:
                param_group_cutoff.append(current_offset)
                prev_id = _find_first_number(name)

            # so here we copy the DTensor from gpu to cpu. The trick is that we need to recreate the DTensor with the correct
            # cpu devise mesh, otherwise we have a cpu DTensor with a cuda device mesh which will fail to do any communication
            target = param.data.to_local().detach()
            data_tensor = self.offloaded_data_flat_tensor.as_strided(target.size(), target.stride(), current_offset)
            grad_tensor = self.offloaded_grad_flat_tensor.as_strided(target.size(), target.stride(), current_offset)
            current_offset += data_tensor.numel()
            data_tensor.copy_(target)

            offloaded_param = nn.Parameter(
                DTensor.from_local(
                    data_tensor,
                    device_mesh=self.cpu_local_mesh,
                    placements=param.data.placements,
                )
            )

            offloaded_param.grad = DTensor.from_local(
                grad_tensor,
                device_mesh=self.cpu_local_mesh,
                placements=param.data.placements,
            )
            # here we pre-allocate the grad DTensor on cpu.
            offloaded_param.requires_grad = True
            offloaded_params.append(offloaded_param)

        param_group_cutoff.append(current_offset)
        # self._logger.debug(f"Cutoffs: {param_group_cutoff}")

        self._offloaded_grad_grouped_tensor = [
            self.offloaded_grad_flat_tensor.as_strided((j - i,), (1,), i)
            for i, j in zip(param_group_cutoff, param_group_cutoff[1:])
        ]
        # self._logger.debug(
        #     f"Grouped Tensors({len(self._offloaded_grad_grouped_tensor)}){[i.numel() for i in self._offloaded_grad_grouped_tensor]}"
        # )
        return offloaded_params

    @torch.no_grad()
    def step(self, model: nn.Module, communicator: Communicator, fake: bool = False):
        """
        Step the optimizer
        """
        time_start = time.perf_counter()
        self.sync_pseudo_gradient(model, communicator, fake=fake)
        self._logger.info(f"all reduce pseudo gradient in: {time.perf_counter() - time_start} seconds")

        if self.outer_optimizer is not None:
            self.outer_optimizer.step()

        self.sync_inner_model(model)
