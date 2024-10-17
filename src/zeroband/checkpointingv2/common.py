import torch
from torch.distributed._tensor.api import DTensor


def _to_local_if_dtensor(t: torch.Tensor) -> torch.Tensor:
    if isinstance(t, DTensor):
        return t.to_local()
    return t
