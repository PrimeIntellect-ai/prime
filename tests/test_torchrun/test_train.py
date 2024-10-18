import copy
import os
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


def _test_multi_gpu(num_gpus, config, extra_args=[], diloco=False):
    num_nodes, num_gpu = num_gpus[0], num_gpus[1]

    processes = []
    ports = get_random_available_port_list(num_nodes)
    new_port = get_random_available_port(1)
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

        if diloco:
            new_env = {
                "GLOBAL_RANK": str(i),
                "GLOBAL_UNIQUE_ID": str(i),
                "GLOBAL_ADDR": "localhost",
                "GLOBAL_WORLD_SIZE": str(num_nodes),
                "GLOBAL_PORT": str(new_port),
            }
            env.update(new_env)

        env["CUDA_VISIBLE_DEVICES"] = gpus_to_use(num_nodes, num_gpu, i)
        env["ZERO_BAND_LOG_LEVEL"] = "DEBUG"

        process1 = subprocess.Popen(cmd, env=env)
        processes.append(process1)

    for process in processes:
        result = process.wait()
        if result != 0:
            pytest.fail(f"Process {result} failed {result}")


@pytest.mark.parametrize("num_gpus", [[1, 1], [2, 1], [1, 2]])
def test_multi_gpu(num_gpus):
    _test_multi_gpu(num_gpus, "debug/normal.toml")


@pytest.mark.parametrize("num_gpus", [[2, 1], [2, 2]])
def test_multi_gpu_diloco(num_gpus):
    _test_multi_gpu(num_gpus, "debug/diloco.toml", diloco=True)


def test_act_ckpt():
    num_gpus = [1, 2]
    _test_multi_gpu(num_gpus, "debug/normal.toml", extra_args=["--train.ac_ckpt"])


def test_act_ckpt_num():
    num_gpus = [1, 2]
    _test_multi_gpu(num_gpus, "debug/normal.toml", extra_args=["--train.ac_ckpt", "2"])


@pytest.mark.parametrize("backend", [Compression.NO, Compression.UINT8])
def test_all_reduce_diloco(backend: Compression):
    num_gpus = [2, 1]
    _test_multi_gpu(num_gpus, "debug/diloco.toml", extra_args=["--diloco.compression", backend.value], diloco=True)


def test_z_loss():
    num_gpus = [1, 1]
    _test_multi_gpu(num_gpus, "debug/normal.toml", extra_args=["--optim.z_loss"])


@pytest.mark.parametrize("packing", [True, False])
def test_packing(packing: bool):
    num_gpus = [2, 1]
    packing_arg = "--train.sequence_packing" if packing else "--no-train.sequence_packing"
    _test_multi_gpu(num_gpus, "debug/normal.toml", extra_args=[packing_arg])
