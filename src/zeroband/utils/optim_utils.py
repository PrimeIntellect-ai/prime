from typing import List, Dict, TypeVar, Type, Tuple, Optional

import torch
from zeroband.config import OptimizerConfig


def make_optimizer(parameters: List[torch.nn.Parameter], config: OptimizerConfig) -> torch.optim.Optimizer:
    """
    Creates an optimizer instance for the supplied parameters according to the given optimizer configuration
    :param parameters the list of parameters to optimize
    :param config the optimizer config
    """
    if config.type == 'sgd':
        return torch.optim.SGD(
            parameters,
            lr=0.0,  # lr will be set later
            momentum=config.momentum,
            nesterov=config.nesterov
        )
    elif config.type == 'adam':
        return torch.optim.Adam(
            parameters,
            lr=0.0,  # lr will be set later
            betas=(config.betas1, config.betas2)
        )
    elif config.type == 'adamw':
        return torch.optim.AdamW(
            parameters,
            lr=0.0,  # lr will be set later
            weight_decay=config.weight_decay,
            betas=(config.betas1, config.betas2),
        )
    else:
        raise ValueError(f"Illegal optimizer type: {config.dtype}")


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float):
    """
    Sets the currently used learning rate for the optimizer
    :param optimizer: the optimizer to set the learning rate for
    :param lr: the learning rate to set
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


OptimT = TypeVar("OptimT", bound=torch.optim.Optimizer)


def add_optimizer_state(shared_state_dict: Dict[str, torch.Tensor],
                        param_name: str,
                        param_optim_state: Dict[str, torch.Tensor],
                        optimizer_type: Type[OptimT]):
    """
    Adds the relevant optimizer state to the shared state dict given the parameter specific optimizer state and
    the type of optimizer used.
    :param shared_state_dict the shared state dict to populate
    :param param_name the name of the parameter
    :param param_optim_state the parameter-specific optimizer state dict
    :param optimizer_type the type of optimizer used.
    """

    def _validate_exists(to_check: List[Tuple[str, Optional[torch.Tensor]]]):
        for key, value in to_check:
            if value is None:
                raise RuntimeError(
                    f"Cannot find '{key}' key in parameter-specific optimizer state for optimizer type f'{optimizer_type.__name__}'. Is Optimizer state initialized?")

    if optimizer_type == torch.optim.SGD:
        momentum_buf = param_optim_state.get("momentum_buffer", None)
        if momentum_buf is not None:
            shared_state_dict[f"{param_name}_momentum_buffer"] = momentum_buf

    elif optimizer_type == torch.optim.Adam or optimizer_type == torch.optim.AdamW:
        exp_avg = param_optim_state.get('exp_avg', None)
        exp_avg_sq = param_optim_state.get('exp_avg_sq', None)
        step_tensor = param_optim_state.get('step', None)
        _validate_exists([
            ('exp_avg', exp_avg),
            ('exp_avg_sq', exp_avg_sq),
            ('step_tensor', step_tensor)
        ])

        # Add optimizer state tensors with associated names
        shared_state_dict[f"{param_name}_m1"] = exp_avg
        shared_state_dict[f"{param_name}_m2"] = exp_avg_sq
        shared_state_dict[f"{param_name}_step"] = step_tensor
