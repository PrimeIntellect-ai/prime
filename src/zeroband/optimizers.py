from typing import Iterable

import torch
import torch.distributed.fsdp
import torch.distributed.tensor



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
            print("\033[91mRunning pipeline hook!\033[0m")
            print(type(param))
            print(param.device)
            print(param.shape)

            from torch.distributed.tensor import DTensor

            # TODO: Thread opt.step_param(). Probably best to leave the explicit thread management to the c++ code.
            #       Then have opt.step() will await all the threads for all the parameter optimizers.

            # TODO: Support DTensors in the C++ extension code.
            # If it's a dtensor, do the optimizer update on the view of the local tensor.
            # This works because Adam is indexwise, and would not work for SOAP.
            if isinstance(param, DTensor):
                _param = param.to_local() # Returns a view
                print("\033[91mLocal shard:\033[0m")
                print(type(_param))
                print(_param.device)
                print(_param.shape)
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
