import copy
import os
from pathlib import Path
import pickle
import subprocess
import pytest
import socket

from zeroband.diloco import Compression


def get_random_available_port_list(num_port):
    # https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
    ports = []

    while len(ports) < num_port:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            new_port = s.getsockname()[1]

        if new_port not in ports:
            ports.append(new_port)

    return ports


def get_random_available_port(num_port):
    return get_random_available_port_list(num_port)[0]


def gpus_to_use(num_nodes, num_gpu, rank):
    return ",".join(map(str, range(rank * num_gpu, (rank + 1) * num_gpu)))


def _test_multi_gpu(num_gpus, config, extra_args=[]):
    num_nodes, num_gpu = num_gpus[0], num_gpus[1]

    processes = []
    ports = get_random_available_port_list(num_nodes)
    for i in range(num_nodes):
        cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpu}",
            "--rdzv-endpoint",
            f"localhost:{ports[i]}",
            "src/zeroband/train.py",
            f"@configs/{config}",
            *extra_args,
        ]

        env = copy.deepcopy(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpus_to_use(num_nodes, num_gpu, i)
        process1 = subprocess.Popen(cmd, env=env)
        processes.append(process1)

    for process in processes:
        result = process.wait()
        if result != 0:
            pytest.fail(f"Process {result} failed {result}")


@pytest.mark.parametrize("num_gpus", [[1, 1], [2, 1], [1, 2]])
def test_multi_gpu(num_gpus):
    _test_multi_gpu(num_gpus, "debug/normal.toml")


@pytest.mark.parametrize("num_gpus", [[1, 2], [2, 2]])
def test_multi_gpu_diloco(num_gpus):
    # we don't test 1,1 and 2,1 because 1 solo gpu failed with fsdp
    _test_multi_gpu(num_gpus, "debug/diloco.toml")


def test_act_ckpt():
    num_gpus = [1, 2]
    _test_multi_gpu(num_gpus, "debug/normal.toml", extra_args=["--train.ac_ckpt"])


def test_act_ckpt_num():
    num_gpus = [1, 2]
    _test_multi_gpu(num_gpus, "debug/normal.toml", extra_args=["--train.ac_ckpt", "2"])


@pytest.mark.parametrize(
    "backend", [Compression.NO, Compression.UINT8]
)  # not adding CINT8 because the compile is too slow
def test_all_reduce_diloco(backend: Compression):
    num_gpus = [2, 1]
    _test_multi_gpu(num_gpus, "debug/diloco.toml", extra_args=["--diloco.compression", backend.value])


def test_z_loss():
    num_gpus = [1, 1]
    _test_multi_gpu(num_gpus, "debug/normal.toml", extra_args=["--optim.z_loss"])


@pytest.mark.parametrize("packing", [True, False])
def test_packing(packing: bool):
    num_gpus = [2, 1]
    packing_arg = "--train.sequence_packing" if packing else "--no-train.sequence_packing"
    _test_multi_gpu(num_gpus, "debug/normal.toml", extra_args=[packing_arg])


def test_ckpt(tmp_path: Path):
    """
    This test just check that we can load the ckpt and resume the training
    """
    num_gpus = [1, 2]

    ckpt_path = tmp_path / "ckpt"
    logging_path_1 = tmp_path / "logging_base"
    _test_multi_gpu(
        num_gpus,
        "debug/exact.toml",
        extra_args=[
            "--ckpt.path",
            f"{ckpt_path}",
            "--ckpt.interval",
            "20",
            "--diloco.inner_steps",
            "20",
            "--optim.total_steps",
            "30",
            "--project",
            f"{logging_path_1}",
        ],
    )

    logging_path_2 = tmp_path / "logging_resume"
    _test_multi_gpu(
        num_gpus,
        "debug/exact.toml",
        extra_args=[
            "--ckpt.resume",
            f"{ckpt_path / 'step_20'}",
            "--diloco.inner_steps",
            "20",
            "--optim.total_steps",
            "30",
            "--project",
            f"{logging_path_2}",
        ],
    )

    with open(logging_path_1, "rb") as f:
        log1 = pickle.load(f)
    with open(logging_path_2, "rb") as f:
        log2 = pickle.load(f)

    print(log1)
    print(log2)

    log1 = {data["step"]: [data["Loss"], data["inner_lr"]] for data in log1 if "Loss" in data.keys()}
    log2 = {data["step"]: [data["Loss"], data["inner_lr"]] for data in log2 if "Loss" in data.keys()}

    common_step = set(log1.keys()) & set(log2.keys())

    for step in common_step:
        assert log1[step][0] == log2[step][0], f"Loss at step {step} is different"
        assert log1[step][1] == log2[step][1], f"Lr at step {step} is different"
