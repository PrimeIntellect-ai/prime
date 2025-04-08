from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Optional
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from torch.distributed.checkpoint.stateful import Stateful

from zeroband.ccl.ccl_utils import MPIConfig
from zeroband.models.llama.model import Transformer


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


@dataclass
class CheckpointInfo:
    # The number of performed outer steps
    num_performed_outer_steps: int

    # The current shared state revision when this checkpoint was saved.
    # This may differ from num_performed_outer_steps if shared state synchronization
    # is skipped during outer steps when possible (e.g. no peers joined)
    shared_state_revision: int


def save_checkpoint(
        model: Transformer,
        optimizers: list[torch.optim.Optimizer],
        training_progress: TrainingProgress,
        dataloader: StatefulDataLoader,
        path_root: str | Path,
        checkpoint_info: CheckpointInfo,
        mpi_config: Optional[MPIConfig]
):
    """
    Checkpoint the model in a way that is compatible with FSDP.
    """
    path_root = _pathify(path_root) / f"step_{training_progress.step}"

    path_file = _local_file_path(path_root, mpi_config.mpi_rank if mpi_config is not None else 1)

    if not os.path.exists(path_root):
        os.makedirs(path_root)

    state = {
        "model": model.state_dict(),
        "optimizers": [optimizer.state_dict() for optimizer in optimizers],
        "training_progress": training_progress,
        "dataloader": dataloader.state_dict(),

        # checkpoint info
        "num_performed_outer_steps": checkpoint_info.num_performed_outer_steps,
        "shared_state_revision": checkpoint_info.shared_state_revision
    }
    with open(path_file, "wb") as f:
        torch.save(state, f)


def load_checkpoint(
        model: Transformer,
        optimizers: list[torch.optim.Optimizer],
        training_progress: TrainingProgress,
        dataloader: StatefulDataLoader,
        path_root: str | Path,
        mpi_config: Optional[MPIConfig]
) -> CheckpointInfo:
    """
    Load the checkpoint state.
    :return checkpoint meta-data information
    """
    path = _pathify(path_root)

    assert os.path.exists(path), f"Checkpoint directory {path} must exist"
    assert os.path.isdir(path), f"Checkpoint directory {path} must be a directory"

    path_file = _local_file_path(path, mpi_config.mpi_rank if mpi_config is not None else 1)

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
    return CheckpointInfo(
        num_performed_outer_steps=state["num_performed_outer_steps"],
        shared_state_revision=state["shared_state_revision"]
    )
