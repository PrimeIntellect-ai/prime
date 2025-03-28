import hashlib
import socket
import torch
from torch.distributed.tensor import DTensor

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
