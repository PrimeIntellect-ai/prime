import math


def compute_current_lr_linear(step: int, num_total_steps: int,
                              learning_rate_scheduler_config: LearningRateSchedulerConfig):
    """
    Compute the current learning rate for the given step and learning rate scheduler configuration.
    Will use the given schedule to interpolate between the initial and end learning rate and optionally apply warmup.
    :param step: the current step post warmup
    :param num_total_steps: the total number of steps
    :param learning_rate_scheduler_config: the learning rate scheduler configuration
    :return: the current learning rate for the given step
    """
    relative = step / num_total_steps
    lr_range = learning_rate_scheduler_config.initial_lr - learning_rate_scheduler_config.end_lr
    return learning_rate_scheduler_config.initial_lr - lr_range * relative


def compute_current_lr_cosine(step: int, num_total_steps: int,
                              learning_rate_scheduler_config: LearningRateSchedulerConfig):
    """
    Compute the current learning rate for the given step and learning rate scheduler configuration.
    Will use the given schedule to interpolate between the initial and end learning rate and optionally apply warmup.
    :param step: the current step post warmup
    :param num_total_steps: the total number of steps
    :param learning_rate_scheduler_config: the learning rate scheduler configuration
    :return: the current learning rate for the given step
    """
    relative = step / num_total_steps
    lr_range = learning_rate_scheduler_config.initial_lr - learning_rate_scheduler_config.end_lr
    return learning_rate_scheduler_config.initial_lr - lr_range * math.sin(relative * math.pi / 2)


def compute_current_lr(step: int, num_total_steps: int, learning_rate_scheduler_config: LearningRateSchedulerConfig):
    """
    Compute the current learning rate for the given step and learning rate scheduler configuration.
    Will use the given schedule to interpolate between the initial and end learning rate and optionally apply warmup.
    :param step: the current step
    :param num_total_steps: the total number of steps
    :param learning_rate_scheduler_config: the learning rate scheduler configuration
    :return: the current learning rate for the given step
    """
    if learning_rate_scheduler_config.num_warmup_steps > 0:
        if step < learning_rate_scheduler_config.num_warmup_steps:
            return learning_rate_scheduler_config.initial_lr * (step / learning_rate_scheduler_config.num_warmup_steps)

    post_warmup_step = step - learning_rate_scheduler_config.num_warmup_steps

    if learning_rate_scheduler_config.scheduler_type == 'linear':
        return compute_current_lr_linear(post_warmup_step, num_total_steps, learning_rate_scheduler_config)
    elif learning_rate_scheduler_config.scheduler_type == 'cosine':
        return compute_current_lr_cosine(post_warmup_step, num_total_steps, learning_rate_scheduler_config)
    else:
        raise ValueError(f"Unsupported scheduler type {learning_rate_scheduler_config.scheduler_type}")
