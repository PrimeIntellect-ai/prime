import math
import pytest
import matplotlib.pyplot as plt

from zeroband.config import LearningRateSchedulerConfig
from zeroband.lr_scheduler import compute_current_lr


def test_linear_no_warmup():
    config = LearningRateSchedulerConfig(
        lr=1.0,
        end_lr=0.0,
        num_warmup_steps=0,
        num_decay_steps=10,
        decay_type='linear'
    )
    assert compute_current_lr(0, config) == pytest.approx(1.0)
    assert compute_current_lr(5, config) == pytest.approx(0.5)
    assert compute_current_lr(9, config) == pytest.approx(0.1)
    assert compute_current_lr(10, config) == pytest.approx(0.0)


def test_linear_with_warmup():
    config = LearningRateSchedulerConfig(
        lr=1.0,
        end_lr=0.0,
        num_warmup_steps=10,
        num_decay_steps=10,
        decay_type='linear'
    )
    assert compute_current_lr(0, config) == pytest.approx(0.0)
    assert compute_current_lr(5, config) == pytest.approx(0.5)
    assert compute_current_lr(10, config) == pytest.approx(1.0)
    assert compute_current_lr(15, config) == pytest.approx(0.5)
    assert compute_current_lr(20, config) == pytest.approx(0.0)


def test_cosine_no_warmup():
    config = LearningRateSchedulerConfig(
        lr=1.0,
        end_lr=0.0,
        num_warmup_steps=0,
        num_decay_steps=10,
        decay_type='cosine'
    )
    assert compute_current_lr(0, config) == pytest.approx(1.0)
    assert compute_current_lr(5, config) == pytest.approx(1.0 - math.sin(0.5 * math.pi / 2))
    assert compute_current_lr(10, config) == pytest.approx(0.0)


def test_cosine_with_warmup():
    config = LearningRateSchedulerConfig(
        lr=1.0,
        end_lr=0.0,
        num_warmup_steps=10,
        num_decay_steps=10,
        decay_type='cosine'
    )
    assert compute_current_lr(0, config) == pytest.approx(0.0)
    assert compute_current_lr(5, config) == pytest.approx(0.5)
    assert compute_current_lr(10, config) == pytest.approx(1.0)
    assert compute_current_lr(15, config) == pytest.approx(1.0 - math.sin(0.5 * math.pi / 2))
    assert compute_current_lr(20, config) == pytest.approx(0.0)


def test_sqrt_no_warmup():
    config = LearningRateSchedulerConfig(
        lr=1.0,
        end_lr=0.0,
        num_warmup_steps=0,
        num_decay_steps=10,
        decay_type='sqrt'
    )
    # At step 0, no decay; lr should be the initial value.
    assert compute_current_lr(0, config) == pytest.approx(1.0)

    # At step 5, relative = 5/10 = 0.5 so sqrt(0.5) ≈ 0.7071,
    # and decayed_lr = 1.0 - (1.0 * 0.7071) ≈ 0.2929.
    expected_lr_step5 = 1.0 - math.sqrt(0.5)
    assert compute_current_lr(5, config) == pytest.approx(expected_lr_step5)

    # At step 10, relative = 1 so sqrt(1) = 1 and lr should be 0.0.
    assert compute_current_lr(10, config) == pytest.approx(0.0)


def test_sqrt_with_warmup():
    config = LearningRateSchedulerConfig(
        lr=1.0,
        end_lr=0.0,
        num_warmup_steps=10,
        num_decay_steps=10,
        decay_type='sqrt'
    )
    # Warmup phase: linear increase from 0.0 to 1.0.
    assert compute_current_lr(0, config) == pytest.approx(0.0)
    assert compute_current_lr(5, config) == pytest.approx(0.5)
    assert compute_current_lr(10, config) == pytest.approx(1.0)

    # Decay phase: at step 15 (5 steps into decay),
    # relative = (15 - 10) / 10 = 0.5 so sqrt(0.5) ≈ 0.7071,
    # and decayed_lr = 1.0 - 0.7071 ≈ 0.2929.
    expected_lr_step15 = 1.0 - math.sqrt(0.5)
    assert compute_current_lr(15, config) == pytest.approx(expected_lr_step15)

    # At step 20, lr should be 0.0.
    assert compute_current_lr(20, config) == pytest.approx(0.0)


def test_sqrt_no_warmup_with_stable():
    config = LearningRateSchedulerConfig(
        lr=1.0,
        end_lr=0.0,
        num_warmup_steps=0,
        num_stable_steps=5,
        num_decay_steps=10,
        decay_type='sqrt'
    )
    # Stable phase: steps [0, num_stable_steps - 1] should retain full lr.
    for step in range(5):
        assert compute_current_lr(step, config) == pytest.approx(1.0)

    # At step 5, decay phase begins: relative = (5 - 5) / 10 = 0, so lr remains 1.0.
    assert compute_current_lr(5, config) == pytest.approx(1.0)

    # At step 6: relative = (6 - 5) / 10 = 0.1, so lr = 1.0 - sqrt(0.1)
    expected_lr_step6 = 1.0 - math.sqrt(0.1)
    assert compute_current_lr(6, config) == pytest.approx(expected_lr_step6)

    # At the end of decay phase (step 15), lr should be 0.0.
    assert compute_current_lr(15, config) == pytest.approx(0.0)


def test_sqrt_with_warmup_with_stable():
    config = LearningRateSchedulerConfig(
        lr=1.0,
        end_lr=0.0,
        num_warmup_steps=10,
        num_stable_steps=5,
        num_decay_steps=10,
        decay_type='sqrt'
    )
    # Warmup phase: steps [0, 9] increase linearly from 0.0 to 1.0.
    assert compute_current_lr(0, config) == pytest.approx(0.0)
    assert compute_current_lr(5, config) == pytest.approx(0.5)
    assert compute_current_lr(10, config) == pytest.approx(1.0)

    # Stable phase: steps 10 to 14 should retain lr = 1.0.
    for step in range(10, 15):
        assert compute_current_lr(step, config) == pytest.approx(1.0)

    # Decay phase: at step 15, decay begins.
    # Here, decay_step = 15 - (10 + 5) = 0 so lr remains 1.0.
    assert compute_current_lr(15, config) == pytest.approx(1.0)

    # At step 16: relative = (16 - (10+5)) / 10 = 0.1,
    # so lr = 1.0 - sqrt(0.1) ≈ 1.0 - 0.316 = 0.684.
    expected_lr_step16 = 1.0 - math.sqrt(0.1)
    assert compute_current_lr(16, config) == pytest.approx(expected_lr_step16)

    # At the end of decay (step 25), lr should be 0.0.
    assert compute_current_lr(25, config) == pytest.approx(0.0)


def plot_schedule(warmup: bool, decay_type: str, num_stable_steps: int = 0):
    config = LearningRateSchedulerConfig(
        lr=1.0,
        end_lr=0.0,
        num_warmup_steps=10 if warmup else 0,
        num_stable_steps=num_stable_steps,
        num_decay_steps=100,
        decay_type=decay_type
    )
    lrs = [compute_current_lr(step, config) for step in range(config.num_total_steps)]
    plt.plot(lrs, label=decay_type)
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    decay_name = decay_type.capitalize()
    title_extra = f" with {config.num_stable_steps} stable steps" if num_stable_steps > 0 else ""
    title_warmup = " with Warmup" if warmup else ""
    plt.title(f'{decay_name} Schedule{title_warmup}{title_extra}')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_schedule(warmup=True, decay_type='linear')
    plot_schedule(warmup=False, decay_type='linear')

    plot_schedule(warmup=True, decay_type='cosine')
    plot_schedule(warmup=False, decay_type='cosine')

    plot_schedule(warmup=True, decay_type='sqrt')
    plot_schedule(warmup=False, decay_type='sqrt')

    plot_schedule(warmup=True, num_stable_steps=10, decay_type='linear')
    plot_schedule(warmup=True, num_stable_steps=10, decay_type='linear')

    plot_schedule(warmup=False, num_stable_steps=10, decay_type='cosine')
    plot_schedule(warmup=False, num_stable_steps=10, decay_type='cosine')

    plot_schedule(warmup=True, num_stable_steps=10, decay_type='sqrt')
    plot_schedule(warmup=False, num_stable_steps=10, decay_type='sqrt')
