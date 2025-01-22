from typing import Any, Iterable, Literal, cast
import torch
from distributed_shampoo import (
    DefaultEigenvalueCorrectedShampooConfig,
    DistributedShampoo,
    FullyShardShampooConfig,
    ShampooPT2CompileConfig,
)
import torch.distributed.fsdp
import torch.distributed.tensor
from torch.distributed._tensor.api import DTensor
from zeroband.config import Config, AdamConfig, SoapConfig, OptimizersConfig
from zeroband.utils.logging import get_logger


def get_optimizer(config: Config, params: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
    """
    Obtain the optimizer for the model.
    """
    if config.train.offload_inner_optimizer:
        # Calls build_optimizer internally after offloading params
        get_logger().info("%" * 150)
        return OffloadedInnerOptimizer(config, params)
    else:
        get_logger().info("=" * 150)
        return build_optimizer(params, config)

def build_optimizer(
    params: Iterable[torch.nn.Parameter], config: Config
) -> torch.optim.Optimizer:
    
    _config: OptimizersConfig = config.optim.optim
    
    if isinstance(_config, AdamConfig):
        opt = torch.optim.AdamW(
            params,
            lr=_config.lr,
            weight_decay=_config.weight_decay,
            betas=(_config.betas1, _config.betas2),
        )
    elif isinstance(_config, SoapConfig):
        opt = DistributedShampoo(
            params,
            lr=_config.lr,
            betas=(_config.betas1, _config.betas2),
            epsilon=1e-12,
            weight_decay=_config.weight_decay,
            max_preconditioner_dim=_config.max_preconditioner_dim,
            precondition_frequency=_config.precondition_frequency,
            use_decoupled_weight_decay=True,
            # This can also be set to `DefaultSOAPConfig` which uses QR decompositions, hence is
            # less expensive and might thereby allow for a smaller `precondition_frequency`.
            preconditioner_config=DefaultEigenvalueCorrectedShampooConfig,
            distributed_config=FullyShardShampooConfig(),
            shampoo_pt2_compile_config=ShampooPT2CompileConfig(
                enable_shampoo_pt2_dynamic_shape=False
            ),
        )
    else:
        raise ValueError(f"Unknown optimizer {_config.optimizer}")
    
    return opt

def timeit(func):
    """
    Decorator to time a function.
    """
    def wrapper(*args, **kwargs):
        import time
        stream = torch.cuda.current_stream()
        start = time.time()
        result = func(*args, **kwargs)
        stream.synchronize()
        end = time.time()
        get_logger().info(f"Time taken by {func.__name__}: {end - start:.4f} seconds")
        return result

    return wrapper

class OffloadedInnerOptimizer(torch.optim.Optimizer):

    def __init__(self, config: Config, parameters: Iterable[torch.nn.Parameter], defaults: dict[str, Any] = {}):
        self.config = config
        self.param_list_gpu = list(parameters)
        self.param_list_cpu = self.offload_inner_opt_params(self.param_list_gpu)
        self.inner_optimizer = build_optimizer(self.param_list_cpu, config)
        # self.inner_optimizer = build_optimizer(self.param_list_gpu, config)
        super().__init__(self.param_list_cpu, defaults=self.inner_optimizer.defaults)
        

    @torch.no_grad()
    def offload_inner_opt_params(
        self, parameters: Iterable[torch.nn.Parameter]
    ) -> list[torch.nn.Parameter]:
        """
        Offload the parameter shards managed by this rank. This works because the optimizer is pointwise.
        """

        param_items = [param for param in parameters if param.requires_grad]
        numels = sum(cast(DTensor, param).to_local().numel() for param in param_items)
        offloaded_params = []

        # Preallocate big buffers on cpu for the local shard.
        current_offset = 0
        self.offloaded_data_flat_tensor = torch.empty((numels,), device="cpu", dtype=torch.float32).pin_memory()
        self.offloaded_grad_flat_tensor = torch.zeros((numels,), device="cpu", dtype=torch.float32).pin_memory()

        # Group the parameters into buckets by transformer block.
        # This is useful for quantization before all reduce.
        for param in param_items:

            # Claim a chunk from the cpu buffers for each parameter.
            assert isinstance(param.data, DTensor)
            target = param.data.to_local().detach()
            data_tensor = self.offloaded_data_flat_tensor.as_strided(target.size(), target.stride(), current_offset)
            grad_tensor = self.offloaded_grad_flat_tensor.as_strided(target.size(), target.stride(), current_offset)
            current_offset += data_tensor.numel()
            data_tensor.copy_(target)

            # Create a new cpu tensor out of that chunk to shadow the dtensor.
            offloaded_param = torch.nn.Parameter(data_tensor)
            offloaded_param.requires_grad = True
            offloaded_param.grad = grad_tensor
            offloaded_params.append(offloaded_param)

        return offloaded_params

    @timeit
    @torch.no_grad()
    def sync_dtoh(self):
        for param_offloaded, param in zip(self.param_list_cpu, self.param_list_gpu):
            assert isinstance(param.data, DTensor)
            assert isinstance(param.grad, DTensor)
            assert param_offloaded.grad is not None
            param_offloaded.data.copy_(param.data.to_local())
            param_offloaded.grad.copy_(param.grad.to_local())

    @timeit
    @torch.no_grad()
    def sync_htod(self):
        for param_offloaded, param in zip(self.param_list_cpu, self.param_list_gpu):
            assert isinstance(param.data, DTensor)
            assert isinstance(param.grad, DTensor)
            assert param_offloaded.grad is not None
            param.data.to_local().copy_(param_offloaded.data)
            param.grad.to_local().copy_(param_offloaded.grad)

    def step(self):
        self.sync_dtoh()
        timeit(self.inner_optimizer.step)()
        self.sync_htod()


__all__ = ["OptimizersConfig", "build_optimizer"]
