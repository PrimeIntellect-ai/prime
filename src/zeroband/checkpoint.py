from dataclasses import dataclass
import gc
import multiprocessing
import os
import shutil
import threading
import time
from typing import Any
import uuid
import fsspec
from fsspec.generic import rsync as rsync_fsspec
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchdata.stateful_dataloader import StatefulDataLoader
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    set_optimizer_state_dict,
    set_model_state_dict,
    get_model_state_dict,
    get_optimizer_state_dict,
    StateDictOptions,
)
import torch.distributed as dist


from torch.distributed.checkpoint.stateful import Stateful
import warnings
import logging
from torch.distributed._tensor.api import DTensor
from zeroband.utils.state_dict_send_recv import (
    _get_sendable_state_dict,
    recv_state_dict,
    send_state_dict,
    send_tensor_and_state_dict,
)
from distributed_shampoo import DistributedShampoo
from zeroband.utils.logger import get_logger
from zeroband.config import CkptConfig
from zeroband.utils.world_info import get_world_info

## code inspired by torchtitan https://github.com/pytorch/torchtitan/blob/main/torchtitan/checkpoint.py


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


class ModelWrapper(Stateful):
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def state_dict(self) -> dict[str, Any]:
        return get_model_state_dict(self.model, options=StateDictOptions(strict=False))

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        set_model_state_dict(model=self.model, model_state_dict=state_dict, options=StateDictOptions(strict=False))


class OptimizerWrapper(Stateful):
    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
    ) -> None:
        self.model = model
        self.optim = optim

    def state_dict(self) -> dict[str, Any]:
        if isinstance(self.optim, DistributedShampoo):
            return self.optim.distributed_state_dict(key_to_param=self.model.named_parameters())
        else:
            return get_optimizer_state_dict(
                model=self.model, optimizers=self.optim, options=StateDictOptions(flatten_optimizer_state_dict=True)
            )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if isinstance(self.optim, DistributedShampoo):
            self.optim.load_distributed_state_dict(state_dict, key_to_param=self.model.named_parameters())
        else:
            set_optimizer_state_dict(
                model=self.model,
                optimizers=self.optim,
                optim_state_dict=state_dict,
                options=StateDictOptions(flatten_optimizer_state_dict=True),
            )


def cast_dtensor_to_tensor(state_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Traverse a state dict and cast all DTensor in the state dict to tensor
    """
    new_state_dict = {}

    for key, value in state_dict.items():
        if isinstance(value, dict):
            new_state_dict[key] = cast_dtensor_to_tensor(value)
        elif isinstance(value, DTensor):
            new_state_dict[key] = value.to_local()
        else:
            new_state_dict[key] = value
    return new_state_dict


def load_dtensor_state_dict(state_src, loaded_state_dict):
    for key, value in state_src.items():
        if isinstance(value, dict):
            load_dtensor_state_dict(value, loaded_state_dict[key])
        elif isinstance(value, DTensor):
            local_tensor = value.to_local()

            local_tensor.copy_(loaded_state_dict[key])
            loaded_state_dict[key] = value
        else:
            loaded_state_dict[key] = value


class OuterOptimizerWrapper(Stateful):
    def __init__(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer

    def state_dict(self) -> dict[str, Any]:
        # the idea here is to cast any DTensor into local tensor
        state = self.optimizer.state_dict()
        return cast_dtensor_to_tensor(state)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # we pre-init the opt buffer DTensor.
        # !! this assume that the model have grad buffer init
        self.optimizer.step()  # pre init buffer

        ## here the idea is for any DTensor, load the value from the state_dict into the local tensor
        current_state = self.optimizer.state_dict()
        load_dtensor_state_dict(current_state, state_dict)
        self.optimizer.load_state_dict(state_dict)


def non_error_barrier():
    try:
        dist.barrier()
    except Exception as e:
        from zeroband.utils.logger import get_logger
        get_logger().info(f"Error in data checkpointing barrier: {e}, continuing training")


class CkptManager:
    """Its name CkptManager because I (sami) always misstyped chekcpoint.

    Checkpoint are saved in a folder with the following structure:
    ckpt_path/
        step_0/
            _0_0.pt
            _1_0.pt
            ...
        step_1/
            ...
    """

    states: dict[str, Stateful]

    def __init__(
        self,
        config: CkptConfig,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        dataloader: StatefulDataLoader,
        training_progress: TrainingProgress,
        data_rank: int | None,
        diloco_offloaded_param_list: list[nn.Parameter] | None,
        diloco_offloaded_optimizer: Optimizer | None,
    ):
        self.config = config

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.training_progress = training_progress
        self.data_rank = data_rank

        assert (diloco_offloaded_param_list is None) == (
            diloco_offloaded_optimizer is None
        ), "diloco_offloaded_model and diloco_offloaded_optimizer must be both None or both have values"

        self.diloco_offloaded_optimizer = diloco_offloaded_optimizer  # he we don't use Wrapper because it failed
        # which might make the ckpt less generic in term of loading from different number of device. FSDP ckpt seems to be a mess tho
        self.diloco_offloaded_param_list = diloco_offloaded_param_list

        self._init_state()

        self._logger = get_logger(config)
        self.world_info = get_world_info()

        self.non_blocking_process: list[multiprocessing.Process] = []
        self.blocking_process: list[multiprocessing.Process] = []
        self._live_reco_thread: threading.Thread | None = None

        if self.world_info.local_rank == 0:
            if self.config.path is not None:
                self.check_path_access(self.config.path)

            if self.config.remote is not None:
                self.check_path_access(self.config.remote.path)

            if self.config.remote_data_path is not None:
                self.check_path_access(self.config.remote_data_path)

    def check_path_access(
        self,
        ckpt_path: str,
    ):
        rank = uuid.uuid4()
        dummy_file_path = os.path.join(ckpt_path, f".dummy_file_{rank}.txt")

        try:
            # Create the directory if it doesn't exist
            fs, _ = fsspec.core.url_to_fs(ckpt_path)
            fs.makedirs(ckpt_path, exist_ok=True)

            with fsspec.open(dummy_file_path, "w") as f:
                f.write("This is a dummy file for testing access.")
        except Exception as e:
            self._logger.error(f"Error checking path access {ckpt_path}: {e}, aborting training")
            raise e

    def _init_state(self):
        # states can only be stateful object, hence we need to wrap Model and Optimizer
        self.states: dict[str, Stateful] = {
            "model": ModelWrapper(self.model),
            "optimizer": OptimizerWrapper(self.model, self.optimizer),
            "scheduler": self.scheduler,
            # "dataloader": self.dataloader, # ignoring dataloader for now as each rank has its own dataloader
            "training_progress": self.training_progress,
        }

        # if self.diloco_offloaded_optimizer is not None:
        #     # even if the diloco_offloaded target the cpu list model, we still use the gpu model to load and save state.
        #     # main reason is that we actually don't a cpu model but just a list of cpu parameters.
        #     self.states["diloco_optimizer"] = self.diloco_offloaded_optimizer

    @torch.no_grad()
    def save(self, remote: bool = False) -> None:
        """
        Each rank will save the right shard of the model and optimizer.

        Saving is done inplace.

        Save in the subfolder `step_<step>`.

        """

        step_ckpt_path = os.path.join(self.config.path, f"step_{self.training_progress.step}")

        if remote and self.config.remote is not None:
            remote_ckpt_path = os.path.join(self.config.remote.path, f"step_{self.training_progress.step}")

        # if we are not in self recovery mode we save to disk
        time_start = time.perf_counter()
        self._save(step_ckpt_path)
        self._logger.info(f"Saved checkpoint to {step_ckpt_path} in {time.perf_counter() - time_start} seconds")

        # push to remote
        non_error_barrier()
        if self.world_info.local_rank == 0:
            if remote and self.config.remote is not None:
                self._async_save_remote(step_ckpt_path, remote_ckpt_path)

    @torch.no_grad()
    def _save(self, ckpt_path: str):
        self.wait_for_blocking_job()

        catch_warning = self._logger.getEffectiveLevel() <= logging.INFO

        with warnings.catch_warnings():
            # pytorch has an annoying warning when saving the optimizer state https://github.com/pytorch/pytorch/issues/136907
            # we can ignore it if we are not logging in DEBUG mode
            if catch_warning:
                warnings.simplefilter("ignore")

            dcp.save(self.states, checkpoint_id=ckpt_path)

            if self.diloco_offloaded_optimizer:
                with open(os.path.join(ckpt_path, f"__{self.world_info.local_rank}_0.pt"), "wb") as f:
                    state = {}
                    state["optimizer"] = OuterOptimizerWrapper(self.diloco_offloaded_optimizer).state_dict()

                    torch.save(state, f)

            data_path = os.path.join(ckpt_path, "data")
            self.save_data(data_path, self.dataloader, self.world_info.local_rank)

            non_error_barrier()

            if self.config.remote_data_path is not None:
                remote_data_path = os.path.join(
                    self.config.remote_data_path, f"data_{self.data_rank}", f"step_{self.training_progress.step}"
                )
                latest_remote_data_path = os.path.join(self.config.remote_data_path, f"data_{self.data_rank}", "latest")

                self._async_save_remote(data_path, remote_data_path, blocking=False)
                self._async_save_remote(data_path, latest_remote_data_path, blocking=False)

        gc.collect()

    @staticmethod
    def save_data(data_path: str, dataloader, local_rank: int):
        os.makedirs(data_path, exist_ok=True)
        with open(os.path.join(data_path, f"_{local_rank}.pt"), "wb") as f:
            state = {"data_loader": dataloader.state_dict()}
            torch.save(state, f)

    def _async_save_remote(self, ckpt_path: str, remote_ckpt_path: str, blocking: bool = True) -> None:
        """asyncronously rsync a ckpt folder to a remote location. Using fsspec to handle remote cloud storage without to install
        specific libraries (e.g. s3fs).
        """

        def rsync():
            time_start = time.perf_counter()
            self._logger.info(f"start pushing {ckpt_path} to {remote_ckpt_path} asynchronously")
            try:
                rsync_fsspec(ckpt_path, destination=remote_ckpt_path)
            except Exception as e:
                self._logger.error(f"Error pushing {ckpt_path} to {remote_ckpt_path}: {e}")
            self._logger.info(
                f"finish pushing {ckpt_path} to {remote_ckpt_path} in {time.perf_counter() - time_start} seconds"
            )

        processes = multiprocessing.Process(target=rsync, daemon=True)
        processes.start()

        if blocking:
            self.blocking_process.append(processes)
        else:
            self.non_blocking_process.append(processes)

    def wait_for_blocking_job(self):
        for process in self.blocking_process:
            process.join()

        self.blocking_process = []

        if self.world_info.local_rank == 0:
            if self.config.topk is not None:
                delete_topk(self.logger, self.config.path, self.config.topk)

    def _del__(self):
        self.wait_for_blocking_job()

        for process in self.non_blocking_process:
            process.join()

    @torch.no_grad()
    def _load_data(self, resume_ckpt_path: str):
        self._logger.debug(f"loading data from {resume_ckpt_path}")
        world_info = get_world_info()

        data_path = os.path.join(resume_ckpt_path, "data")

        with open(os.path.join(data_path, f"_{world_info.local_rank}.pt"), "rb") as f:
            state = torch.load(f)
            self.dataloader.load_state_dict(state["data_loader"])

    @torch.no_grad()
    def load(
        self,
        resume_ckpt_path: str,
        skip_dataloader: bool = False,
        data_path: str | None = None,
    ) -> None:
        """
        loading should be done after fsdp wrap and optimizer init.
        Each rank will load the right shard of the model and optimizer.
        All rank will load the global states (scheduler, step, total_tokens, dataloader).

        `resume_ckpt_path` should point to a specific step and not to the base ckpt folder. Example: `ckpt_path/step_100`

        Loading is done inplace.

        """
        time_start = time.perf_counter()

        world_info = get_world_info()

        files = os.listdir(resume_ckpt_path)

        if len(files) == 1 and files[0].startswith("diloco_"):
            self._logger.warning(
                f"Loading diloco ckpt from {files[0]}. This is deprecated and will be removed in the future"
            )
            resume_ckpt_path = os.path.join(resume_ckpt_path, files[0])

        dcp.load(self.states, checkpoint_id=resume_ckpt_path)

        if self.config.token_count is not None:
            self.training_progress.total_tokens = self.config.token_count

        self._logger.debug("sync inner model")
        # todo(refactor): here we should rather let the diloco class handle this logic
        if self.diloco_offloaded_param_list is not None:
            for param_offloaded, param in zip(self.diloco_offloaded_param_list, self.model.parameters()):
                param_offloaded.data.to_local().copy_(param.data.to_local())

        if self.diloco_offloaded_optimizer:
            with open(os.path.join(resume_ckpt_path, f"__{world_info.local_rank}_0.pt"), "rb") as f:
                rank_state_dict = torch.load(f)

            opt_wrapper = OuterOptimizerWrapper(self.diloco_offloaded_optimizer)
            opt_wrapper.load_state_dict(rank_state_dict["optimizer"])

        if not skip_dataloader:
            if self.config.remote_data_load:
                self.remote_data_load()
            else:
                data_path = resume_ckpt_path if data_path is None else data_path
                self._load_data(data_path)

        self._init_state()

        self._logger.info(f"Loaded checkpoint from {resume_ckpt_path} in {time.perf_counter() - time_start} seconds")

    def remote_data_load(self):
        remote_data_path = os.path.join(self.config.remote_data_path, f"data_{self.data_rank}", "latest")
        id_ = uuid.uuid4()
        dest = f"/tmp/zeroband/data_{id_}"
        rsync_fsspec(remote_data_path, os.path.join(dest, "data"))
        data_path = dest
        self._load_data(data_path)

    @torch.no_grad()
    def recv_ckpt_from_peer(self, global_pg: dist.ProcessGroup):
        assert self.diloco_offloaded_param_list is not None, "recv_ckpt_from_peers is only supported with diloco"

        time_start = time.perf_counter()
        self._logger.debug(f"Start receiving ckpt from rank {self.config.live_recovery_rank_src}")

        jobs = []
        buffers = []
        for i, param in enumerate(self.diloco_offloaded_param_list):
            data = param.data
            if isinstance(param.data, DTensor):
                data = param.data.to_local()

            buffer = torch.empty_like(data)
            buffers.append(buffer)
            jobs.append(global_pg.recv([buffer], self.config.live_recovery_rank_src, i))

        for job in jobs:
            job.wait()

        for buffer, param in zip(buffers, self.model.parameters()):
            data = param.data
            if isinstance(data, DTensor):
                data = data.to_local()
            data.copy_(buffer)

        self._logger.debug("live recovery progress: offloaded model received 1/5")

        outer_opt_state_dict = recv_state_dict(
            global_pg, self.config.live_recovery_rank_src, self.diloco_offloaded_optimizer.state_dict()
        )
        self.diloco_offloaded_optimizer.load_state_dict(outer_opt_state_dict)

        self._logger.debug("live recovery progress: outer optimizer state dict received 2/5")

        training_process_state_dict = recv_state_dict(
            global_pg, self.config.live_recovery_rank_src, self.training_progress.state_dict()
        )
        self.training_progress.load_state_dict(training_process_state_dict)
        self._logger.debug("live recovery progress: training progress state dict received 3/5")

        for group in self.optimizer.param_groups:
            for p in group["params"]:
                p.grad = torch.randn_like(p)

        self.optimizer.step()
        self.optimizer.zero_grad()

        inner_opt_state_dict = recv_state_dict(
            global_pg, self.config.live_recovery_rank_src, self.optimizer.state_dict()
        )
        self.optimizer.load_state_dict(inner_opt_state_dict)

        self._logger.debug("live recovery progress: inner optimizer state dict received 4/5")

        sheduler_state_dict = recv_state_dict(
            global_pg, self.config.live_recovery_rank_src, self.scheduler.state_dict()
        )
        self.scheduler.load_state_dict(sheduler_state_dict)

        self._logger.debug("live recovery progress: scheduler state dict received 5/5")

        self._logger.debug(
            f"Received ckpt from rank {self.config.live_recovery_rank_src} in {time.perf_counter() - time_start} seconds"
        )

    @torch.no_grad()
    def send_ckpt_to_peer(self, global_pg: dist.ProcessGroup, dest_rank: int, blocking: bool = False):
        def async_send():
            assert self.diloco_offloaded_param_list is not None, "send_ckpt_to_peers is only supported with diloco"
            time_start = time.perf_counter()
            self._logger.debug(f"Start sending ckpt to rank {dest_rank}")

            try:
                jobs = []
                for i, param in enumerate(self.diloco_offloaded_param_list):
                    data = param.data
                    if isinstance(data, DTensor):
                        data = data.to_local()
                    jobs.append(global_pg.send([data], dest_rank, i))

                for job in jobs:
                    job.wait()

                send_state_dict(global_pg, self.diloco_offloaded_optimizer.state_dict(), dest_rank)
                send_state_dict(global_pg, self.training_progress.state_dict(), dest_rank)

                inner_optimizer_non_tensor_state_dict, inner_optimizer_tensors = _get_sendable_state_dict(
                    self.optimizer.state_dict()
                )
                send_tensor_and_state_dict(
                    global_pg, dest_rank, inner_optimizer_non_tensor_state_dict, inner_optimizer_tensors
                )

                send_state_dict(global_pg, self.scheduler.state_dict(), dest_rank)
            except RuntimeError as e:
                self._logger.error(f"Error sending ckpt to rank {dest_rank}: {e}")
            else:
                self._logger.debug(f"Sent ckpt to rank {dest_rank} in {time.perf_counter() - time_start} seconds")

        thread = threading.Thread(target=async_send)
        thread.start()
        self._logger.debug("Live recovery thread started")
        if blocking:
            thread.join()
        else:
            self._live_reco_thread = thread


def delete_topk(logger: logging.Logger, ckpt_path: str, topk: int):
    checkpoints_to_delete = get_checkpoints_to_delete(ckpt_path, topk)
    for ckpt_path in checkpoints_to_delete:
        shutil.rmtree(ckpt_path, ignore_errors=True)
    if len(checkpoints_to_delete) > 0:
        logger.info(f"Deleted {checkpoints_to_delete} checkpoints")


def get_checkpoints_to_delete(ckpt_path: str, topk: int) -> list[str]:
    checkpoints = [d for d in os.listdir(ckpt_path) if d.startswith("step_")]
    sorted_checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[1]), reverse=True)
    return [os.path.join(ckpt_path, d) for d in sorted_checkpoints[topk:]]
