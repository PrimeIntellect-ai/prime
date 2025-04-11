import torch
import contextlib


@contextlib.contextmanager
def default_device(device: str = "cuda"):
    old_device = torch.get_default_device()
    torch.set_default_device(device)
    try:
        yield
    finally:
        # Restore the old default device
        torch.set_default_device(old_device)
