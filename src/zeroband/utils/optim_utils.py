import torch


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float):
    """
    Sets the currently used learning rate for the optimizer
    :param optimizer: the optimizer to set the learning rate for
    :param lr: the learning rate to set
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
