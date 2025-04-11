import subprocess
import pytest


def _test_torchrun(num_gpus, config, extra_args=[]):
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "src/zeroband/train.py",
        f"@configs/debug/{config}",
        *extra_args,
    ]

    process = subprocess.Popen(cmd)
    result = process.wait()
    if result != 0:
        pytest.fail(f"Process  failed {result}")


@pytest.mark.parametrize("num_gpus", [1, 2])
def test_train(num_gpus):
    _test_torchrun(num_gpus=num_gpus, config="normal.toml")
