from typing import Literal, TypeAlias
from pydantic_config import BaseConfig
import torch
from distributed_shampoo.shampoo_types import AdamGraftingConfig, EigenvalueCorrectedShampooPreconditionerConfig
from matrix_functions_types import DefaultEighEigenvectorConfig, TopKCompressionEigenvectorConfig

from distributed_shampoo import (
    DistributedShampoo,
    FullyShardShampooConfig,
    ShampooPT2CompileConfig,
)


class AdamConfig(BaseConfig):
    type: Literal["adam"] = (
        "adam"  # the literal is used to distinguish between the different optimizers configuration in the union type
    )
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

    topk: TopKCompressionEigenvectorConfig | None = None

    eigen_stats: bool = False


class ShampooConfig(BaseConfig):
    type: Literal["shampoo"] = "shampoo"
    lr: float = 4e-4
    weight_decay: float = 1e-05
    betas1: float = 0.9
    betas2: float = 0.95

    precondition_frequency: int = 100
    max_preconditioner_dim: int = 8192


OptimizersConfig: TypeAlias = AdamConfig | SoapConfig | ShampooConfig

# Constants for large matrix patterns in LLaMA
LLAMA_LARGE_MATRIX_PATTERNS = [
    "attention.wq",  # Attention matrices
    "attention.wk",
    "attention.wv",
    "attention.wo",
    "feed_forward.w1",  # FFN matrices
    "feed_forward.w2",
    "feed_forward.w3",
]


def split_model_parameters(model: torch.nn.Module):
    """
    Split model parameters into large matrices and other parameters.
    Returns a tuple of (other_params, large_matrix_params)
    """
    large_params = []
    other_params = []

    for name, param in model.named_parameters():
        if any(pattern in name for pattern in LLAMA_LARGE_MATRIX_PATTERNS):
            large_params.append(param)
        else:
            # Everything else (including tok_embeddings and output) goes here
            other_params.append(param)

    return other_params, large_params


def get_optimizer(model: torch.nn.Module, config: OptimizersConfig) -> torch.optim.Optimizer:
    if isinstance(config, AdamConfig):
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(config.betas1, config.betas2),
        )
    elif isinstance(config, SoapConfig):
        amortized_computation_config = DefaultEighEigenvectorConfig if config.topk is None else config.topk

        other_params, large_params = split_model_parameters(model)

        param_groups = [
            {
                "params": large_params,
                "preconditioner_config": EigenvalueCorrectedShampooPreconditionerConfig(
                    amortized_computation_config=amortized_computation_config
                ),
                "eigen_stats": config.eigen_stats,
            },
            {
                "params": other_params,
                "preconditioner_config": EigenvalueCorrectedShampooPreconditionerConfig(
                    amortized_computation_config=DefaultEighEigenvectorConfig
                ),
                "eigen_stats": False,
            },
        ]
        # we only apply topk compression to large params

        return DistributedShampoo(
            param_groups,
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
                amortized_computation_config=DefaultEighEigenvectorConfig
            ),
            distributed_config=FullyShardShampooConfig(),
            shampoo_pt2_compile_config=ShampooPT2CompileConfig(enable_shampoo_pt2_dynamic_shape=False),
        )
    elif isinstance(config, ShampooConfig):
        return DistributedShampoo(
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
            grafting_config=AdamGraftingConfig(
                beta2=0.999,
                epsilon=1e-08,
            ),
            distributed_config=FullyShardShampooConfig(),
            shampoo_pt2_compile_config=ShampooPT2CompileConfig(enable_shampoo_pt2_dynamic_shape=False),
        )
    else:
        raise ValueError(f"Unknown optimizer {config.optimizer}")


__all__ = ["OptimizersConfig", "get_optimizer"]
