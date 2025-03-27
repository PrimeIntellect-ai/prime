import torch
from zeroband.config import OptimizerConfig


def make_optimizer(model: torch.nn.Module, config: OptimizerConfig):
    """
    Creates an optimizer instance for the parameters of the supplied model according to the given optimizer configuration
    :param model the model to optimize
    :param config the optimizer config
    """
    if config.type == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=0.0,  # lr will be set later
            betas=(config.betas1, config.betas2)
        )
    elif config.type == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=0.0,  # lr will be set later
            weight_decay=config.weight_decay,
            betas=(config.betas1, config.betas2),
        )


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float):
    """
    Sets the currently used learning rate for the optimizer
    :param optimizer: the optimizer to set the learning rate for
    :param lr: the learning rate to set
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
