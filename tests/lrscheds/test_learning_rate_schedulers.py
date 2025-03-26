import math

import pytest
import matplotlib.pyplot as plt

from zeroband.config import LearningRateSchedulerConfig
from zeroband.lr_scheduler import compute_current_lr


def test_linear_no_warmup():
    config = LearningRateSchedulerConfig(
        initial_lr=1.0,
        end_lr=0.0,
        num_warmup_steps=0,
        num_decay_steps=10,
        scheduler_type='linear'
    )
    assert compute_current_lr(0, config) == pytest.approx(1.0)
    assert compute_current_lr(5, config) == pytest.approx(0.5)
    assert compute_current_lr(9, config) == pytest.approx(0.1)
    assert compute_current_lr(10, config) == pytest.approx(0.0)


def test_linear_with_warmup():
    config = LearningRateSchedulerConfig(
        initial_lr=1.0,
        end_lr=0.0,
        num_warmup_steps=10,
        num_decay_steps=10,
        scheduler_type='linear'
    )
    assert compute_current_lr(0, config) == pytest.approx(0.0)
    assert compute_current_lr(5, config) == pytest.approx(0.5)
    assert compute_current_lr(10, config) == pytest.approx(1.0)
    assert compute_current_lr(15, config) == pytest.approx(0.5)
    assert compute_current_lr(20, config) == pytest.approx(0.0)


def test_cosine_no_warmup():
    config = LearningRateSchedulerConfig(
        initial_lr=1.0,
        end_lr=0.0,
        num_warmup_steps=0,
        num_decay_steps=10,
        scheduler_type='cosine'
    )
    assert compute_current_lr(0, config) == pytest.approx(1.0)
    assert compute_current_lr(5, config) == pytest.approx(1.0 - math.sin(0.5 * math.pi / 2))
    assert compute_current_lr(10, config) == pytest.approx(0.0)


def test_cosine_with_warmup():
    config = LearningRateSchedulerConfig(
        initial_lr=1.0,
        end_lr=0.0,
        num_warmup_steps=10,
        num_decay_steps=10,
        scheduler_type='cosine'
    )
    assert compute_current_lr(0, config) == pytest.approx(0.0)
    assert compute_current_lr(5, config) == pytest.approx(0.5)
    assert compute_current_lr(10, config) == pytest.approx(1.0)
    assert compute_current_lr(15, config) == pytest.approx(1.0 - math.sin(0.5 * math.pi / 2))
    assert compute_current_lr(20, config) == pytest.approx(0.0)


def plot_linear_schedule(warmup: bool, num_stable_steps: int = 0):
    config = LearningRateSchedulerConfig(
        initial_lr=1.0,
        end_lr=0.0,
        num_warmup_steps=10 if warmup else 0,
        num_stable_steps=num_stable_steps,
        num_decay_steps=100,
        scheduler_type='linear'
    )
    lrs = [compute_current_lr(step, config) for step in
           range(config.num_total_steps)]
    plt.plot(lrs, label='linear')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title(
        'Linear Schedule with Warmup' if warmup else 'Linear Schedule' + f' with {config.num_stable_steps} stable steps')
    plt.legend()
    plt.show()


def plot_cosine_schedule(warmup: bool, num_stable_steps: int = 0):
    config = LearningRateSchedulerConfig(
        initial_lr=1.0,
        end_lr=0.0,
        num_warmup_steps=10 if warmup else 0,
        num_stable_steps=num_stable_steps,
        num_decay_steps=100,
        scheduler_type='cosine'
    )
    lrs = [compute_current_lr(step, config) for step in
           range(config.num_total_steps)]
    plt.plot(lrs, label='cosine')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title(
        'Cosine Schedule with Warmup' if warmup else 'Cosine Schedule' + f' with {config.num_stable_steps} stable steps')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_linear_schedule(warmup=True)
    plot_linear_schedule(warmup=False)

    plot_cosine_schedule(warmup=True)
    plot_cosine_schedule(warmup=False)

    plot_linear_schedule(warmup=True, num_stable_steps=10)
    plot_cosine_schedule(warmup=True, num_stable_steps=10)

    plot_linear_schedule(warmup=False, num_stable_steps=10)
    plot_cosine_schedule(warmup=False, num_stable_steps=10)
