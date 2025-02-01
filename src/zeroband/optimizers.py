import torch
from distributed_shampoo import (
    DefaultEigenvalueCorrectedShampooConfig,
    DistributedShampoo,
    DDPShampooConfig,
    ShampooPT2CompileConfig,
)
from zeroband.models.llama.model import Transformer
from zeroband.muon import Muon
from zeroband.config import AdamConfig, SoapConfig, OptimizersConfig, MuonConfig
from zeroband.utils.world_info import get_world_info


def get_optimizer(model: Transformer, config: OptimizersConfig) -> list[torch.optim.Optimizer]:
    if isinstance(config, AdamConfig):
        return [
            torch.optim.AdamW(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
                betas=(config.betas1, config.betas2),
            )
        ]
    elif isinstance(config, SoapConfig):
        return [
            DistributedShampoo(
                model.parameters(),
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
                distributed_config=DDPShampooConfig(),
                shampoo_pt2_compile_config=ShampooPT2CompileConfig(enable_shampoo_pt2_dynamic_shape=False),
            )
        ]
    elif isinstance(config, MuonConfig):
        world_info = get_world_info()
        hidden_matrix_params = [p for n, p in model.layers.named_parameters() if p.ndim >= 2 and "embed" not in n]
        embed_params = [p for n, p in model.named_parameters() if "embed" in n]
        scalar_params = [p for p in model.parameters() if p.ndim < 2]
        head_params = [model.output.weight]

        # init the optimizer(s)
        adam_params = [
            dict(params=head_params, lr=0.008),
            dict(params=embed_params, lr=0.6),
            dict(params=scalar_params, lr=0.04),
        ]
        # small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
        # discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
        optimizer1 = torch.optim.Adam(adam_params, betas=(0.8, 0.95), eps=1e-10, fused=True)
        optimizer2 = Muon(
            hidden_matrix_params,
            lr=config.lr,
            momentum=config.momentum,
            rank=world_info.rank,
            world_size=world_info.world_size,
            compression_ratio=config.compression_ratio,
            compression_step_start=config.compression_step_start,
            lie_compression=config.lie_compression,
        )
        return [optimizer2, optimizer1]
    else:
        raise ValueError(f"Unknown optimizer {config.optimizer}")


__all__ = ["OptimizersConfig", "get_optimizer"]
