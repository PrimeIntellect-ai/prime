from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from torch.distributed.checkpoint.stateful import Stateful
from zeroband.models.llama.model import Transformer
from zeroband.utils.world_info import get_world_info


@dataclass
class TrainingProgress(Stateful):
    total_tokens: int
    outer_step: int
    step: int

    def state_dict(self) -> dict[str, Any]:
        return {"total_tokens": self.total_tokens, "outer_step": self.outer_step, "step": self.step}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.total_tokens = state_dict["total_tokens"]
        self.outer_step = state_dict["outer_step"]
        self.step = state_dict["step"]


def _local_file_path(path: Path, local_rank: int) -> Path:
    return path / f"local_rank_{local_rank}.pt"


def _pathify(path: str | Path) -> Path:
    if isinstance(path, str):
        return Path(path)
    return path


def save_checkpoint_fsdp_state(
    model: Transformer,
    optimizers: list[torch.optim.Optimizer],
    training_progress: TrainingProgress,
    dataloader: StatefulDataLoader,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    path_root: str | Path,
):
    """
    Checkpoint the model in a way that is compatible with FSDP.
    """
    path_root = _pathify(path_root) / f"step_{training_progress.step}"
    world_info = get_world_info()

    path_file = _local_file_path(path_root, world_info.local_rank)

    if not os.path.exists(path_root):
        os.makedirs(path_root)

    with open(path_file, "wb") as f:
        state = {}
        state["model"] = model.state_dict()
        state["optimizers"] = [optimizer.state_dict() for optimizer in optimizers]
        state["training_progress"] = training_progress
        state["dataloader"] = dataloader.state_dict()
        state["scheduler"] = scheduler.state_dict()

        torch.save(state, f)


def load_checkpoint_fsdp_state(
    model: Transformer,
    optimizers: list[torch.optim.Optimizer],
    training_progress: TrainingProgress,
    dataloader: StatefulDataLoader,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    path: str | Path,
):
    """
    Load the checkpoint state.
    """
    path = _pathify(path)
    world_info = get_world_info()

    path_file = _local_file_path(path, world_info.local_rank)

    if not os.path.exists(path_file):
        raise FileNotFoundError(f"Checkpoint step {training_progress.step} not found at {path_file}")

    with open(path_file, "rb") as f:
        state = torch.load(f, weights_only=False)

    model.load_state_dict(state["model"])

    for optimizer, optimizer_state in zip(optimizers, state["optimizers"]):
        optimizer.load_state_dict(optimizer_state)

    training_progress.total_tokens = state["training_progress"].total_tokens
    training_progress.step = state["training_progress"].step

    dataloader.load_state_dict(state["dataloader"])
    scheduler.load_state_dict(state["scheduler"])
