from typing import Iterable
import torch
from distributed_shampoo import (
    DefaultEigenvalueCorrectedShampooConfig,
    DistributedShampoo,
    FullyShardShampooConfig,
    ShampooPT2CompileConfig,
)
from zeroband.config import AdamConfig, SoapConfig, OptimizersConfig

def get_optimizer(params: Iterable[torch.nn.Parameter], config: OptimizersConfig) -> torch.optim.Optimizer:
    if isinstance(config, AdamConfig):
        return torch.optim.AdamW(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(config.betas1, config.betas2),
        )
    elif isinstance(config, SoapConfig):
        return DistributedShampoo(
            params,
            lr=config.lr,
            betas=(config.betas1, config.betas2),
            epsilon=1e-12,
            weight_decay=config.weight_decay,
            max_preconditioner_dim=config.max_preconditioner_dim,
            precondition_frequency=config.precondition_frequency,
            use_decoupled_weight_decay=True,
            # This can also be set to `DefaultSOAPConfig` which uses QR decompositions, hence is
            # less expensive and might thereby allow for a smaller `precondition_frequency`.
            preconditioner_config=DefaultEigenvalueCorrectedShampooConfig,
            distributed_config=FullyShardShampooConfig(),
            shampoo_pt2_compile_config=ShampooPT2CompileConfig(enable_shampoo_pt2_dynamic_shape=False),
        )
    else:
        raise ValueError(f"Unknown optimizer {config.optimizer}")


__all__ = ["OptimizersConfig", "get_optimizer"]
