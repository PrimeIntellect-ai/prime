from typing import TypeAlias
from pydantic_config import BaseConfig
import torch
from distributed_shampoo.shampoo_types import EigenvalueCorrectedShampooPreconditionerConfig
from matrix_functions_types import DefaultEighEigenvectorConfig, TopKCompressionEigenvectorConfig
from zeroband.optimizers.muon import Muon, AdamConfig, MuonConfig
from distributed_shampoo import (
    DistributedShampoo,
    FullyShardShampooConfig,
    ShampooPT2CompileConfig,
)


class SoapConfig(BaseConfig):
    lr: float = 4e-4
    weight_decay: float = 1e-05
    betas1: float = 0.9
    betas2: float = 0.95

    max_preconditioner_dim: int = 8192
    precondition_frequency: int = 100

    topk_compression: int | None = None


OptimizersConfig: TypeAlias = AdamConfig | MuonConfig | SoapConfig


def get_optimizer(params: list[torch.nn.Parameter], config: OptimizersConfig) -> torch.optim.Optimizer:
    if isinstance(config, AdamConfig):
        return torch.optim.AdamW(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(config.betas1, config.betas2),
        )
    elif isinstance(config, MuonConfig):
        return Muon(
            params,
            lr=config.lr,
            momentum=config.momentum,
            nesterov=config.nesterov,
            ns_steps=config.pseudo_order_steps,
            adamw_lr=config.adam.lr,
            adamw_betas=(config.adam.betas1, config.adam.betas2),
            adamw_wd=config.adam.weight_decay,
        )
    elif isinstance(config, SoapConfig):
        amortized_computation_config = (
            DefaultEighEigenvectorConfig
            if config.topk_compression is None
            else TopKCompressionEigenvectorConfig(topk_compression=config.topk_compression)
        )
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
            preconditioner_config=EigenvalueCorrectedShampooPreconditionerConfig(
                amortized_computation_config=amortized_computation_config
            ),
            distributed_config=FullyShardShampooConfig(),
            shampoo_pt2_compile_config=ShampooPT2CompileConfig(enable_shampoo_pt2_dynamic_shape=False),
        )
    else:
        raise ValueError(f"Unknown optimizer {config.optimizer}")


__all__ = ["OptimizersConfig", "get_optimizer"]
