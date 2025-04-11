import math

from zeroband.config import LearningRateSchedulerConfig


def compute_current_lr(step: int, learning_rate_scheduler_config: LearningRateSchedulerConfig):
    """
    Compute the current learning rate for the given step and learning rate scheduler configuration.
    Will use the given schedule to interpolate between the initial and end learning rate and optionally apply warmup.
    :param step: the current step
    :param learning_rate_scheduler_config: the learning rate scheduler configuration
    :return: the current learning rate for the given step
    """
    if learning_rate_scheduler_config.num_warmup_steps > 0:
        if step < learning_rate_scheduler_config.num_warmup_steps:
            return learning_rate_scheduler_config.lr * (step / learning_rate_scheduler_config.num_warmup_steps)

        # convert step to next phase local unit such that it starts at zero
        step -= learning_rate_scheduler_config.num_warmup_steps

    if learning_rate_scheduler_config.num_stable_steps > 0:
        if step < learning_rate_scheduler_config.num_stable_steps:
            return learning_rate_scheduler_config.lr

        # convert step to next phase local unit such that it starts at zero
        step -= learning_rate_scheduler_config.num_stable_steps

    return _compute_decayed_lr(step, learning_rate_scheduler_config)


def _compute_decayed_lr(step, learning_rate_scheduler_config: LearningRateSchedulerConfig):
    if learning_rate_scheduler_config.decay_type == 'linear':
        return _compute_decayed_lr_linear(step, learning_rate_scheduler_config)
    elif learning_rate_scheduler_config.decay_type == 'cosine':
        return _compute_decayed_lr_cosine(step, learning_rate_scheduler_config)
    elif learning_rate_scheduler_config.decay_type == 'sqrt':
        return _compute_decayed_lr_sqrt(step, learning_rate_scheduler_config)
    else:
        raise ValueError(f"Unsupported scheduler type {learning_rate_scheduler_config.scheduler_type}")


def _compute_decayed_lr_linear(step: int,
                               learning_rate_scheduler_config: LearningRateSchedulerConfig):
    """
    Uses linear decay to compute the current decayed learning rate for the given step and learning rate scheduler configuration.
    :param step: the current phase-local step count
    :param learning_rate_scheduler_config: the learning rate scheduler configuration
    :return: the current learning rate for the given step
    """
    relative = step / learning_rate_scheduler_config.num_decay_steps if learning_rate_scheduler_config.num_decay_steps > 0 else 0
    lr_range = learning_rate_scheduler_config.lr - learning_rate_scheduler_config.end_lr
    return learning_rate_scheduler_config.lr - lr_range * relative


def _compute_decayed_lr_cosine(step: int,
                               learning_rate_scheduler_config: LearningRateSchedulerConfig):
    """
    Uses cosine decay to compute the current decayed learning rate for the given step and learning rate scheduler configuration.
    :param step: the current phase-local step count
    :param learning_rate_scheduler_config: the learning rate scheduler configuration
    :return: the current learning rate for the given step
    """
    relative = step / learning_rate_scheduler_config.num_decay_steps
    lr_range = learning_rate_scheduler_config.lr - learning_rate_scheduler_config.end_lr
    return learning_rate_scheduler_config.lr - lr_range * math.sin(relative * math.pi / 2)


def _compute_decayed_lr_sqrt(step: int,
                             learning_rate_scheduler_config: LearningRateSchedulerConfig):
    """
    Uses sqrt decay to compute the current decayed learning rate for the given step and learning rate scheduler configuration.

    :param step: the current phase-local step count
    :param learning_rate_scheduler_config: the learning rate scheduler configuration
    :return: the current learning rate for the given step
    """
    relative = step / learning_rate_scheduler_config.num_decay_steps
    lr_range = learning_rate_scheduler_config.lr - learning_rate_scheduler_config.end_lr
    sqrt_decay = math.sqrt(relative)
    return learning_rate_scheduler_config.lr - lr_range * sqrt_decay
