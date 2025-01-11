from typing import Literal, TypeAlias
from pydantic_config import BaseConfig
import torch
from distributed_shampoo import (
    DefaultEigenvalueCorrectedShampooConfig,
    DistributedShampoo,
    FullyShardShampooConfig,
    ShampooPT2CompileConfig,
)

class AdamConfig(BaseConfig):
    type: Literal["adam"] = "adam" # the literal is used to distinguish between the different optimizers configuration in the union type
    lr: float = 4e-4
    weight_decay: float = 0.1
    betas1: float = 0.9
    betas2: float = 0.95

class SoapConfig(BaseConfig):
    type: Literal["soap"] = "soap"
    lr: float = 4e-4
    weight_decay: float = 1e-05
    betas1: float = 0.9
    betas2: float = 0.95

    max_preconditioner_dim: int = 8192
    precondition_frequency: int = 100


OptimizersConfig: TypeAlias = AdamConfig | SoapConfig


def get_optimizer(params: list[torch.nn.Parameter], config: OptimizersConfig) -> torch.optim.Optimizer:
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
