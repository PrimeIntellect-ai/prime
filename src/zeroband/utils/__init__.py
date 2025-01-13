import hashlib
import socket
import time
import torch
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed._tensor.api import DTensor
from distributed_shampoo import DistributedShampoo


__all__ = ["get_sharding_strategy", "get_peak_flops", "get_num_flop_per_token", "get_num_params"]


def get_sharding_strategy(sharding_strategy: str) -> ShardingStrategy:
    if sharding_strategy == "FULL_SHARD":
        return ShardingStrategy.FULL_SHARD
    elif sharding_strategy == "SHARD_GRAD_OP":
        return ShardingStrategy.SHARD_GRAD_OP
    elif sharding_strategy == "NO_SHARD":
        return ShardingStrategy.NO_SHARD
    elif sharding_strategy == "HYBRID_SHARD":
        return ShardingStrategy.HYBRID_SHARD
    elif sharding_strategy == "_HYBRID_SHARD_ZERO2":
        return ShardingStrategy._HYBRID_SHARD_ZERO2
    else:
        raise ValueError(
            f"Invalid sharding_strategy: {sharding_strategy}. Please choose 'FULL_SHARD', 'SHARD_GRAD_OP', 'NO_SHARD', 'HYBRID_SHARD', or '_HYBRID_SHARD_ZERO2'."
        )


### code above inspired and copied from https://github.com/pytorch/torchtitan/blob/4b3f2e41a084bf79a8540068ed525539d1244edd/torchtitan/utils.py#L119


# hardcoded BF16 type peak flops for NVIDIA A100 and H100 GPU
def get_peak_flops(device_name: str) -> int:
    if "A100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/a100/
        return 312e12
    elif "H100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h100/
        # NOTE: Specifications are one-half lower without sparsity.
        if "NVL" in device_name:
            return 835e12
        elif "PCIe" in device_name:
            return 756e12
        else:  # for H100 SXM and other variants
            return 989e12
    else:  # for other GPU types, assume A100
        return 312e12


def get_num_flop_per_token(num_params: int, model_config, seq_len) -> int:
    l, h, q, t = (  # noqa: E741
        model_config.n_layers,
        model_config.n_heads,
        model_config.dim // model_config.n_heads,
        seq_len,
    )
    # Reasoning behind the factor of 12 for the self-attention part of the formula:
    # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
    # 2. the flash attention does 1 more matmul recomputation in the backward
    #    but recomputation should not be counted in calculating MFU           (+0)
    # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
    # 4. we follow the convention and do not account for sparsity in causal attention
    flop_per_token = 6 * num_params + 12 * l * h * q * t

    return flop_per_token


def get_num_params(model: torch.nn.Module, exclude_embedding: bool = False) -> int:
    num_params = sum(p.numel() for p in model.parameters())
    if exclude_embedding:
        num_params -= model.tok_embeddings.weight.numel()
    return num_params


class PerfCounter:
    """A class to count tokens per second with a rolling window.
    we use a rollowing window because time perf counter is not precise enough in some case
    """

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.tokens = []
        self.times = []

    def count_tokens(self, tokens: int):
        self.tokens.append(tokens)
        self.times.append(time.perf_counter())
        if len(self.tokens) > self.window_size:
            self.tokens.pop(0)
            self.times.pop(0)

    def get_tokens_per_second(self) -> float | None:
        if len(self.tokens) < 2:
            return None
        return sum(self.tokens[1:]) / (self.times[-1] - self.times[0])


TENSOR_SIG_SAMPLE_SIZE = 100


def get_tensor_signature(a: torch.Tensor | torch.nn.Parameter) -> str:
    """
    Get the tensor signature
    """
    while isinstance(a, torch.nn.Parameter):
        a = a.data

    if isinstance(a, DTensor):
        a = a.full_tensor()

    if a.numel() < TENSOR_SIG_SAMPLE_SIZE:
        b = a.as_strided(size=(a.numel(),), stride=(1,))
    else:
        step_size = a.numel() // TENSOR_SIG_SAMPLE_SIZE
        b = a.as_strided(size=(TENSOR_SIG_SAMPLE_SIZE,), stride=(step_size,))
    element_str = "".join([f"{x:.3e}" for x in b])
    element_hash = hashlib.md5(element_str.encode("utf-8")).hexdigest()
    return f"{a.dtype}{a.shape}{a.stride()}<{element_hash}>"


def get_module_signature(module: torch.nn.Module, compress: bool = True) -> str:
    """
    Get the module signature
    """
    state_dict_sig = {name: get_tensor_signature(param) for name, param in module.named_parameters()}
    if compress:
        return hashlib.md5(str(state_dict_sig).encode("utf-8")).hexdigest()
    else:
        return "\n".join(f"{name}: {sig}" for name, sig in state_dict_sig.items())


def get_dict_signature(dict: dict, compress: bool = True) -> str:
    return hashlib.md5(str(dict).encode("utf-8")).hexdigest()


def get_optimizer_signature(optimizer: torch.optim.Optimizer, compress: bool = True) -> str:
    """
    Get the optimizer signature
    """

    if isinstance(optimizer, DistributedShampoo):
        return "mocked signature because shampoo does not support state_dict()"

    def unwrap_tensor(state_dict: dict) -> dict:
        new_dict = {}
        for key, value in state_dict.items():
            if isinstance(value, dict):
                new_dict[key] = unwrap_tensor(value)
            elif isinstance(value, torch.Tensor):
                new_dict[key] = get_tensor_signature(value)
            else:
                new_dict[key] = str(value)
        return new_dict

    state_dict_sig = unwrap_tensor(optimizer.state_dict())

    if compress:
        return hashlib.md5(str(state_dict_sig).encode("utf-8")).hexdigest()
    else:
        return "\n".join(f"{name}: {sig}" for name, sig in state_dict_sig.items())


def get_tensor_list_signature(tensor_list: list[torch.Tensor]) -> str:
    tensors = [get_tensor_signature(tensor) for tensor in tensor_list]
    return hashlib.md5(str(tensors).encode("utf-8")).hexdigest()


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


def get_random_available_port():
    return get_random_available_port_list(1)[0]


class FakeTokenizer(object):
    def __init__(self):
        self.vocab_size = 1000
        self.bos_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 2

    def __len__(self):
        return self.vocab_size