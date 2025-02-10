from typing import Iterable

import torch
from torch.distributed.tensor import DTensor

from zeroband.config import Config, AdamConfig, SoapConfig, OptimizersConfig, CPUAdamConfig




def get_optimizer(config: Config, params: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
    """
    Obtain the optimizer for the model.
    """

    _config: OptimizersConfig = config.optim.optim


    if isinstance(_config, AdamConfig):
        opt = torch.optim.AdamW(
            params,
            lr=_config.lr,
            weight_decay=_config.weight_decay,
            betas=(_config.betas1, _config.betas2),
        )
    elif isinstance(_config, SoapConfig):
        from distributed_shampoo import (
            DefaultEigenvalueCorrectedShampooConfig,
            DistributedShampoo,
            FullyShardShampooConfig,
            ShampooPT2CompileConfig,
        )

        opt = DistributedShampoo(
            params,
            lr=_config.lr,
            betas=(_config.betas1, _config.betas2),
            epsilon=_config.eps,
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
    elif isinstance(_config, CPUAdamConfig):
        from CPUOptimizer.cpu_adam import CPUAdam

        # Closes over opt before it's defined. It's cursed but it's how cpython works.
        def pipeline_hook(param):
            # TODO: Thread opt.step_param(). Probably best to leave the explicit thread management to the c++ code.
            #       Then have opt.step() will await all the threads for all the parameter optimizers.

            # TODO: Support DTensors in the C++ extension code.
            # If it's a dtensor, do the optimizer update on the view of the local tensor.
            # This works because Adam is indexwise, and would not work for SOAP, which would have to do an allgather.
            # Or... would it? Aren't all the shards are on CPU anyway? So can't I can just .full_tensor() it? Will figure out tomorrow.
            # The thing that I need to do to immediately get it working is to figure out what the size of the optimizer state
            # is supposed to be and figure out how to pass it in when the parameter is initialized.
            if isinstance(param, DTensor):
                # Time how long it takes to convert the DTensor to a local tensor.
                import time
                start = time.perf_counter()
                _param = param.full_tensor() # Acquire a view into the local shard of the DTensor.
                end = time.perf_counter()
                _grad = param.grad()
                print(f"Time to convert DTensor to full tensor: {end - start}")
                print(f"\033[91mDTensor: {param}\033[0m\n" # DTensor
                      f"\033[91mDTensor grad: {param.grad}\033[0m\n" # DTensor
                      f"\033[91mLocal shard: {type(_param.data)}\033[0m\n"
                      f"\033[91mLocal grad: {type(_param.grad)}\033[0m")
                opt.step_param(_param)
            else:
                opt.step_param(param)

        opt = CPUAdam(
            params,
            lr=_config.lr,
            betas=(_config.betas1, _config.betas2),
            eps=_config.eps,
            weight_decay=_config.weight_decay,
            pipeline_hook=pipeline_hook if _config.pipelined else None,
        )
    else:
        raise ValueError(f"Unknown optimizer {_config.optimizer}")

    return opt


__all__ = ["OptimizersConfig", "get_optimizer"]
