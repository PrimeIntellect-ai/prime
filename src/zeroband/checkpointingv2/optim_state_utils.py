from typing import Tuple
import torch
from torch import optim
from .common import _to_local_if_dtensor


def initialize_optimizer_state(optimizer: optim.Optimizer) -> None:
    """Initialize the optimizer state with empty tensors"""
    for group in optimizer.param_groups:
        for p in group["params"]:
            p.grad = torch.empty_like(p)

    optimizer.step()
    optimizer.zero_grad()


def get_optimizer_state_numels_dtype_device(optimizer: optim.Optimizer) -> Tuple[int, torch.dtype, torch.device]:
    """Count the number of elements in the optimizer state dict, and return the dtype and device of the tensors"""
    state_dict = optimizer.state_dict()
    numels = 0
    dtype = None
    device = None
    for group in state_dict["param_groups"]:
        for param in group["params"]:
            for _, tensor_value in sorted(state_dict["state"][param].items()):
                if dtype is None:
                    dtype = tensor_value.dtype
                else:
                    assert dtype == tensor_value.dtype, "All tensors in the optimizer state must have the same dtype"
                if device is None:
                    device = tensor_value.device
                else:
                    assert (
                        device == tensor_value.device
                    ), "All tensors in the optimizer state must be on the same device"

                numels += _to_local_if_dtensor(tensor_value).numel()
    return numels, dtype, device


def copy_optimizer_state_to_flat(optimizer: optim.Optimizer, flat_tensor: torch.Tensor) -> None:
    state_dict = optimizer.state_dict()
    offset = 0
    for group in state_dict["param_groups"]:
        for param in group["params"]:
            for _, tensor_value in sorted(state_dict["state"][param].items()):
                _temp_view = flat_tensor.as_strided(tensor_value.shape, tensor_value.stride(), offset)
                _temp_view.copy_(tensor_value, non_blocking=True)
                offset += tensor_value.numel()


def copy_flat_to_optimizer_state(flat_tensor: torch.Tensor, optimizer: optim.Optimizer) -> None:
    state_dict = optimizer.state_dict()
    offset = 0
    for group in state_dict["param_groups"]:
        for param in group["params"]:
            for _, tensor_value in sorted(state_dict["state"][param].items()):
                _temp_view = flat_tensor.as_strided(tensor_value.shape, tensor_value.stride(), offset)
                tensor_value.copy_(_temp_view, non_blocking=True)
                offset += tensor_value.numel()


def back_optimizer_state_with_flat(optimizer: optim.Optimizer) -> torch.Tensor:
    state_dict = optimizer.state_dict()
    numels, dtype, device = get_optimizer_state_numels_dtype_device(optimizer)
    flat_tensor = torch.empty(numels, dtype=dtype, device=device)
    offset = 0
    for group in state_dict["param_groups"]:
        for param in group["params"]:
            for i, tensor_value in sorted(state_dict["state"][param].items()):
                _temp_view = flat_tensor.as_strided(tensor_value.shape, tensor_value.stride(), offset)
                _temp_view.copy_(tensor_value, non_blocking=True)
                offset += tensor_value.numel()
                state_dict["state"][param][i] = _temp_view
