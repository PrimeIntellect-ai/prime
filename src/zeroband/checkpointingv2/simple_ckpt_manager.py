from typing import Iterable, Optional, List
import torch
from torch.distributed.checkpoint.stateful import Stateful
from zeroband.utils.logging import get_logger
import logging
from pathlib import Path
import multiprocessing as mp
from multiprocessing.synchronize import Event as EventType
from safetensors import safe_open
from safetensors.torch import save_file

from .common import _to_local_if_dtensor


def _is_path(path_or_url: str) -> bool:
    return not path_or_url.startswith("liveckpt://")


class SimpleCkptManager:
    """Save order: Urgents -> Non Tensors -> Tensors
    We assume that all init args are by reference and not by value.
    Order of tensors matters for now as we do not save in fqn format.

    Args:
        tensors (Iterable[torch.Tensor]): List of tensors to be saved. Tensors can also be DTensors.
        non_tensors (Iterable[Stateful]): List of non-tensors to be saved
        urgents (Iterable[Stateful]): List of urgents to be saved
    """

    def __init__(
        self,
        tensors: Iterable[torch.Tensor],
        non_tensors: Iterable[Stateful],
        urgents: Iterable[Stateful],
    ):
        self._logger = get_logger(__name__)
        self._disk_job_queue: List[mp.Process] = []
        self._remote_job_queue: List[mp.Process] = []
        self._urgents_disk_completion_queue: List[EventType] = []
        self._non_tensors_disk_completion_queue: List[EventType] = []

        self._tensors = list(tensors)
        self._non_tensors = list(non_tensors)
        self._urgents = list(urgents)

    def _log_debug_states(self, msg: str = "Observing states for ckpt"):
        from zeroband.utils import get_tensor_signature
        from hashlib import md5

        self._logger.debug(
            "[%s] Num tensors: %d, Num non-tensors: %d, Num urgents: %d",
            msg,
            len(self._tensors),
            len(self._non_tensors),
            len(self._urgents),
        )
        tensor_sigs = [get_tensor_signature(t, full_dtensor=False) for t in self._tensors]
        compressed_tensor_sigs = md5(str(tensor_sigs).encode("utf-8")).hexdigest()
        self._logger.debug("All-tensors signature: %s", compressed_tensor_sigs)
        self._logger.debug(
            "All non-tensor signatures: %s",
            [md5(str(n.state_dict()).encode("utf-8")).hexdigest() for n in self._non_tensors],
        )
        self._logger.debug(
            "All urgent signatures: %s", [md5(str(u.state_dict()).encode("utf-8")).hexdigest() for u in self._urgents]
        )

    def test_path(self, disk_path: Optional[str] = None, remote_path: Optional[str] = None):
        if disk_path is not None:
            Path(disk_path).mkdir(parents=True, exist_ok=True)
        if remote_path is not None:
            pass

    def load(self, rank: int, path_or_url: str):
        if _is_path(path_or_url):
            self._load_from_disk(path_or_url, rank)
        else:
            # Load from remote
            pass
        if self._logger.isEnabledFor(logging.DEBUG):
            self._log_debug_states("After loading")

    def save(self, rank: int, disk_path: Optional[str] = None, remote_path: Optional[str] = None):
        """The save will occur at the path.
        Please specify a unique path for each ckpt, otherwise it will overwrite the previous ckpt.
        """
        self._logger.info("Saving to disk: [%s], remote: [%s]", disk_path, remote_path)
        if self._logger.isEnabledFor(logging.DEBUG):
            self._log_debug_states("Saving")
        if disk_path is not None:
            disk_proc = mp.Process(target=self._save_to_disk, args=(disk_path, rank))
            disk_proc.start()
            self._disk_job_queue.append(disk_proc)
        if remote_path is not None:
            pass

    def _save_to_disk(
        self,
        disk_path: str,
        rank: int,
        urgent_event: Optional[EventType] = None,
        non_tensor_event: Optional[EventType] = None,
    ):
        _disk_path = Path(disk_path)
        _disk_path.mkdir(parents=True, exist_ok=True)
        # TODO: You probably want some packing here but for now we will save them as is
        for i, stateful in enumerate(self._urgents):
            torch.save(stateful.state_dict(), _disk_path / f"__{rank}_{i}_urgent.pt")
            if urgent_event is not None:
                urgent_event.set()
        for i, stateful in enumerate(self._non_tensors):
            torch.save(stateful.state_dict(), _disk_path / f"__{rank}_{i}_non_tensor.pt")
            if non_tensor_event is not None:
                non_tensor_event.set()
        for i, tensor in enumerate(self._tensors):
            _tensors = {"tensor": _to_local_if_dtensor(tensor)}
            save_file(_tensors, str(_disk_path / f"__{rank}_{i}.safetensors"))

    @torch.no_grad
    def _load_from_disk(self, disk_path: str, rank: int):
        _disk_path = Path(disk_path)
        for i, stateful in enumerate(self._urgents):
            _state_dict = torch.load(_disk_path / f"__{rank}_{i}_urgent.pt", weights_only=True)
            stateful.load_state_dict(_state_dict)
        for i, stateful in enumerate(self._non_tensors):
            _state_dict = torch.load(_disk_path / f"__{rank}_{i}_non_tensor.pt", weights_only=True)
            stateful.load_state_dict(_state_dict)
        for i, tensor in enumerate(self._tensors):
            _tensors = {}
            with safe_open(_disk_path / f"__{rank}_{i}.safetensors", framework="pt", device="cpu") as f:
                for key in f.keys():
                    _tensors[key] = f.get_tensor(key)
            _to_local_if_dtensor(tensor).copy_(_tensors["tensor"], non_blocking=False)

    def wait_for_all_jobs(self):
        self.wait_for_disk_jobs()
        self.wait_for_remote_jobs()

    def wait_for_disk_jobs(self):
        for disk_proc in self._disk_job_queue:
            disk_proc.join()
        self._disk_job_queue.clear()

    def wait_for_remote_jobs(self):
        for remote_proc in self._remote_job_queue:
            remote_proc.join()
        self._remote_job_queue.clear()

    def wait_for_urgent_disk_jobs(self):
        for event in self._urgents_disk_completion_queue:
            event.wait()

    def wait_for_non_tensor_disk_jobs(self):
        for event in self._non_tensors_disk_completion_queue:
            event.wait()
