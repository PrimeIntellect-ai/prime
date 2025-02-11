from typing import Iterable

import torch

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
            fused=True,
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
            #       This will probably result in a massive perf improvement.
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
