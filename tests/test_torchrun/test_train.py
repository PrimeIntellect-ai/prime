import copy
import os
from pathlib import Path
import pickle
import subprocess
import pytest
import socket

from zeroband.diloco import Compression

import torch

num_gpu = torch.cuda.device_count()


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
                "GLOO_SOCKET_IFNAME": "lo",
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


@pytest.mark.parametrize("num_gpus", [[2, 1], [2, 2]] if num_gpu >= 4 else [[2, 1]])
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
    packing_arg = "--data.sequence_packing" if packing else "--no-data.sequence_packing"
    _test_multi_gpu(num_gpus, "debug/normal.toml", extra_args=[packing_arg])


@pytest.mark.parametrize("diloco", [False, True])
def test_soap(diloco: bool):
    num_gpus = [1, 2] if diloco else [2, 1]
    _test_multi_gpu(
        num_gpus,
        "debug/diloco.toml" if diloco else "debug/normal.toml",
        extra_args=["--optim.optim.precondition_frequency", "1"],
        diloco=diloco,
    )


@pytest.mark.parametrize("soap", [False, True])
def test_ckpt(tmp_path: Path, soap: bool):
    num_gpus = [1, 2]
    v1_file = tmp_path / "v1.log"
    v2_file = tmp_path / "v2.log"
    # v3_file = tmp_path / "v3.log"

    v1_ckpt = tmp_path / "v1_ckpt"
    v2_ckpt = tmp_path / "v2_ckpt"
    # v3_ckpt = tmp_path / "v3_ckpt"

    os.mkdir(v1_ckpt)
    os.mkdir(v2_ckpt)
    # os.mkdir(v3_ckpt)

    _test_multi_gpu(
        num_gpus,
        "debug/diloco.toml",
        extra_args=[
            "--project",
            str(v1_file),
            "--ckpt.path",
            str(v1_ckpt),
            "--ckpt.interval",
            "5",
            "--optim.total_steps",
            "20",
            "--train.log_model_hash",
            "--no-data.sequence_packing",
            "--train.attn_fn",
            "math",
        ]
        + (["--optim.optim.precondition_frequency", "1"] if soap else []),
        diloco=True,
    )
    _test_multi_gpu(
        num_gpus,
        "debug/diloco.toml",
        extra_args=[
            "--project",
            str(v2_file),
            "--ckpt.path",
            str(v2_ckpt),
            "--ckpt.interval",
            "5",
            "--ckpt.resume",
            str(v1_ckpt / "step_5"),
            "--optim.total_steps",
            "20",
            "--train.log_model_hash",
            "--no-data.sequence_packing",
            "--train.attn_fn",
            "math",
        ]
        + (["--optim.optim.precondition_frequency", "1"] if soap else []),
        diloco=True,
    )
    # _test_multi_gpu(
    #     num_gpus,
    #     "debug/diloco.toml",
    #     extra_args=[
    #         "--project",
    #         str(v3_file),
    #         "--ckpt.path",
    #         str(v3_ckpt),
    #         "--ckpt.interval",
    #         "5",
    #         "--ckpt.resume",
    #         str(v2_ckpt / "step_10"),
    #         "--optim.total_steps",
    #         "20",
    #         "--train.log_model_hash",
    #         "--no-data.sequence_packing",
    #         "--train.attn_fn",
    #         "math",
    #     ],
    #     diloco=True,
    # )

    key_to_round = ["Perplexity", "Loss"]
    digit_to_round = [0, 3]

    def read_logs(path: Path):
        with path.open("rb") as f:
            data = pickle.load(f)

        filtered_data = {}
        for entry in data:
            step = entry.pop("step")

            # Round perplexity and loss
            for key, digit in zip(key_to_round, digit_to_round):
                if key in entry:
                    entry[key] = round(entry[key], digit)

            if step in filtered_data:
                filtered_data[step].update(entry)
            else:
                filtered_data[step] = entry

        return filtered_data

    v1_data = read_logs(v1_file)
    v2_data = read_logs(v2_file)
    # v3_data = read_logs(v3_file)

    ## check that loading from v1 to v2 worked

    # first check that the hash of saving is the same as the hash of loading
    assert v1_data[5]["inner_model_hash_save"] == v2_data[5]["inner_model_hash_resume"]
    assert v1_data[5]["inner_optimizer_hash_save"] == v2_data[5]["inner_optimizer_hash_resume"]
    assert v1_data[5]["outer_optimizer_hash_save"] == v2_data[5]["outer_optimizer_hash_resume"]
    assert v1_data[5]["outer_model_hash_save"] == v2_data[5]["outer_model_hash_resume"]

    # then we check that the loss and lr value are the same after loading the ckpt
    for step, data_v2 in v2_data.items():
        if step == 5:
            continue  # not testing 5 as ts the one were we restarted from

        data_v1 = v1_data[step]
        assert abs(data_v1["Loss"] - data_v2["Loss"]) < .1
        assert data_v1["inner_lr"] == data_v2["inner_lr"]
        assert data_v1["total_tokens"] == data_v2["total_tokens"]

    # ## check that the second loading is working
    # ## why ? We had bugs where ckpt was working but not when the training was resuming

    # assert v2_data[10]["inner_model_hash_save"] == v3_data[10]["inner_model_hash_resume"]
    # assert v2_data[10]["inner_optimizer_hash_save"] == v3_data[10]["inner_optimizer_hash_resume"]
    # assert v2_data[10]["outer_optimizer_hash_save"] == v3_data[10]["outer_optimizer_hash_resume"]
    # assert v2_data[10]["outer_model_hash_save"] == v3_data[10]["outer_model_hash_resume"]

    # for step, data_v3 in v3_data.items():
    #     if step == 10:
    #         continue  # not testing 10 as ts the one were we restarted from

    #     data_v2 = v2_data[step]
    #     assert data_v2["Loss"] == data_v3["Loss"]
    #     assert data_v2["inner_lr"] == data_v3["inner_lr"]
    #     assert data_v2["total_tokens"] == data_v3["total_tokens"]
