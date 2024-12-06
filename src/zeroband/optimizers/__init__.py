from typing import TypeAlias
import torch
from zeroband.optimizers.muon import Muon, AdamConfig, MuonConfig


OptimizersConfig: TypeAlias = AdamConfig | MuonConfig


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
    else:
        raise ValueError(f"Unknown optimizer {config.optimizer}")


__all__ = ["OptimizersConfig", "get_optimizer"]
