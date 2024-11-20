import os
from typing import Literal
import time
import psutil
from pydantic import model_validator
from multiprocessing.process import _children

import torch
from pydantic_config import parse_argv, BaseConfig
from einops import rearrange
from torch.nn import functional as F

from transformers import AutoTokenizer

from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

import torch.distributed as dist
from zeroband import utils
from zeroband.diloco import Diloco, DilocoConfig
from zeroband.comms import PcclCommunicator
from zeroband.loss import cross_entropy_max_z_loss

from zeroband.utils import (
    GPUMemoryMonitor,
    PerfCounter,
    get_module_signature,
    get_optimizer_signature,
    get_tensor_list_signature,
)
from zeroband.utils.activation_ckpt import apply_ac_ckpt
from zeroband.data import TEST_VOCAB_SIZE, get_dataloader, DataConfig
from zeroband.utils.metric_logger import WandbMetricLogger, DummyMetricLogger
from zeroband.utils.monitor import HttpMonitor
from zeroband.models.llama import get_model
from zeroband.utils.profiler import MemoryProfiler
from zeroband.utils.world_info import get_world_info
from zeroband.utils.logging import get_logger
from zeroband.checkpoint import CkptConfig, CkptManager, TrainingProgress
from zeroband.lr_scheduler import get_scheduler


class OptimConfig(BaseConfig):
    lr: float = 4e-4
    weight_decay: float = 0.1
    adam_betas1: float = 0.9
    adam_betas2: float = 0.95

    sched_type: Literal["cosine", "linear", "wsd-sqrt"] = "cosine"
    warmup_steps: int = 1000
    stable_steps: int = 80_000
    total_steps: int = 88_000
    batch_size: int = 512

    z_loss: bool = False
    z_loss_weight: float = 2e-4


class MemoryProfilerConfig(BaseConfig):
    freq: int = 10
    snapshot_dir: str


class TrainConfig(BaseConfig):
    micro_bs: int
    torch_compile: bool = True
    ac_ckpt: bool | int = False
    reshard_after_forward: bool = True  # old shard grad op True mean full shard

    reduce_fp32: bool = False  # should be True if SXM. Keep to false as default for backward compatibility

    log_model_hash: bool = False

    memory_monitor: bool = False
    memory_profiler: MemoryProfilerConfig | None = None

    sequence_packing: bool = True
    attn_fn: Literal["flash", "sdpa"] = "flash"

    @model_validator(mode="after")
    def validate_attn_fn(self):
        if self.attn_fn == "sdpa" and self.sequence_packing:
            raise ValueError("SDPA does not support sequence packing")

        return self


class MonitorConfig(BaseConfig):
    log_flush_interval: int = 10
    base_url: str | None = None
    auth_token: str | None = None


class Config(BaseConfig):
    # main config
    name_model: Literal["debugmodel", "150M", "271M", "1B", "7B", "10B", "13B", "26B", "70B"] = "150M"
    type_model: Literal["llama2", "llama3"] = "llama3"

    project: str = "zeroband"
    run_id: str | None = None
    metric_logger_type: Literal["wandb", "dummy"] = "wandb"
    wandb_resume: bool = False

    # sub config
    diloco: DilocoConfig | None = None
    data: DataConfig = DataConfig()
    optim: OptimConfig = OptimConfig()
    train: TrainConfig
    monitor: MonitorConfig | None = None

    ckpt: CkptConfig = CkptConfig()

    @model_validator(mode="after")
    def ckpt_diloco_step(self):
        if self.ckpt is not None and self.ckpt.interval is not None and self.diloco is not None:
            assert (
                self.ckpt.interval % self.diloco.inner_steps == 0
            ), "ckpt interval must be a multiple of diloco inner steps as we only save at the end of an outer step"
        return self

    @model_validator(mode="after")
    def validate_live_recovery_rank_src(self):
        if self.ckpt is not None and self.ckpt.live_recovery_rank_src is not None and self.diloco is None:
            raise ValueError("live_recovery_rank_src is only supported with diloco")
        return self


def train(config: Config):
    # batch_size is the total batch size for all GPUs
    assert config.optim.batch_size % world_info.local_world_size == 0
    batch_size = config.optim.batch_size // world_info.local_world_size

    assert batch_size % config.train.micro_bs == 0
    gradient_accumulation_steps = batch_size // config.train.micro_bs

    if config.ckpt is not None and config.ckpt.interval is not None and config.diloco is not None:
        assert (
            config.ckpt.interval % config.diloco.inner_steps == 0
        ), "ckpt interval must be a multiple of diloco inner steps as we only save at the end of an outer step"

    if config.type_model == "llama2":
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)
    elif config.type_model == "llama3":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_fast=True)
    else:
        raise ValueError(f"Model type {config.type_model} not supported")

    logger.debug("tokenizer loaded")

    train_dataloader = get_dataloader(
        tokenizer=tokenizer,
        world_size=world_info.world_size,
        rank=world_info.rank,
        batch_size=config.train.micro_bs,
        data_config=config.data,
    )

    model, model_config = get_model(
        config.name_model,
        config.type_model,
        vocab_size=len(tokenizer) if config.name_model != "debugmodel" or not config.data.fake else TEST_VOCAB_SIZE,
        seq_length=config.data.seq_length,
        attn_fn=config.train.attn_fn,
    )

    model = model.to(world_info.local_rank)
    logger.debug("model loaded")

    gpu_peak_flops = utils.get_peak_flops(torch.cuda.get_device_name(torch.device("cuda")))
    logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")

    num_params = utils.get_num_params(model, exclude_embedding=True)
    logger.info(f"Number of parameters: {num_params}")
    num_flop_per_token = utils.get_num_flop_per_token(
        num_params,
        model_config,
        config.data.seq_length,
    )

    if config.train.ac_ckpt:
        num = 1 if isinstance(config.train.ac_ckpt, bool) else config.train.ac_ckpt
        apply_ac_ckpt(model, num)

    pccl_communicator = PcclCommunicator()

    dist.init_process_group()

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32 if config.train.reduce_fp32 else None
    )

    for layer_id, transformer_block in model.layers.items():
        if config.train.reshard_after_forward:
            reshard_after_forward = int(layer_id) < len(model.layers) - 1
        else:
            reshard_after_forward = False
        fully_shard(
            transformer_block,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )
    fully_shard(
        model,
        mp_policy=mp_policy,
        reshard_after_forward=config.train.reshard_after_forward,
    )
    logger.debug("model fsdped")

    # Setup optimizers
    inner_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optim.lr,
        weight_decay=config.optim.weight_decay,
        betas=(config.optim.adam_betas1, config.optim.adam_betas2),
    )

    if config.diloco is not None:
        diloco = Diloco(config.diloco, model, pccl_communicator)

    scheduler = get_scheduler(
        sched_type=config.optim.sched_type,
        optimizer=inner_optimizer,
        num_warmup_steps=config.optim.warmup_steps,
        num_stable_steps=config.optim.stable_steps,
        num_training_steps=config.optim.total_steps,
    )

    training_progress = TrainingProgress(total_tokens=0, outer_step=0, step=0)

    ckpt_manager = CkptManager(
        config=config.ckpt,
        model=model,
        optimizer=inner_optimizer,
        scheduler=scheduler,
        dataloader=train_dataloader,
        training_progress=training_progress,
        data_rank=config.data.data_rank,
        diloco_offloaded_optimizer=diloco.outer_optimizer if config.diloco is not None else None,
        diloco_offloaded_param_list=diloco.param_list_cpu if config.diloco is not None else None,
    )

    if config.train.torch_compile:
        # we need to compile AFTER creating the CKPT manager, DON'T ASK ME WHY
        model = torch.compile(model)
        logger.debug("model compiled")

    if config.ckpt.resume is not None:
        # all is inplace
        ckpt_manager.load(
            resume_ckpt_path=config.ckpt.resume,
            skip_dataloader=config.ckpt.skip_dataloader,
            data_path=config.ckpt.data_path,
        )
        if config.train.log_model_hash:
            logger.info(f"model hash: {get_module_signature(model)}")
            logger.info(f"optimizer hash: {get_optimizer_signature(inner_optimizer)}")

            if config.diloco is not None:
                logger.info(f"outer optimizer hash: {get_optimizer_signature(diloco.outer_optimizer)}")
                logger.info(f"outer model hash: {get_tensor_list_signature(diloco.param_list_cpu)}")

    if world_info.rank == 0:
        logger_cls = WandbMetricLogger if config.metric_logger_type == "wandb" else DummyMetricLogger
        metric_logger = logger_cls(
            project=config.project,
            config={"config": config.model_dump(), "world_info": world_info.json()},
            resume=config.wandb_resume,
        )

    if config.train.memory_monitor:
        gpu_mem_monitor = GPUMemoryMonitor()
    if config.train.memory_profiler is not None:
        memory_profiler = MemoryProfiler(config.train.memory_profiler.freq, config.train.memory_profiler.snapshot_dir)

    if config.monitor is not None:
        monitor = HttpMonitor(config=config.model_dump(), resume=False)
        monitor.set_stage("init")

    train_dataloader_iterator = iter(train_dataloader)

    num_inner_steps = config.diloco.inner_steps if config.diloco is not None else 1
    perf_counter = PerfCounter(window_size=10)

    logger.info("starting training")

    # need_live_recovery = config.ckpt.live_recovery_rank_src is not None
    while True:
        if num_inner_steps > 1:
            # if we don't use diloco we don't print the outer step logs
            logger.info(f"outer_step step: {training_progress.outer_step}")

        time_start_outer = time.perf_counter()

        # if config.diloco is not None:
        #     # this is a patch for now to allow live recovery worker to not affect the all reduce at all

        #     if not need_live_recovery:
        #         elastic_device_mesh.maybe_reinit_global_pg(admit_joiners=True)

        #         maybe_dest_rank = elastic_device_mesh.live_recovery.should_send_ckpt_to()
        #         if maybe_dest_rank is not None:
        #             logger.info(f"Start live recovery to rank {maybe_dest_rank}")
        #             if config.train.log_model_hash:
        #                 logger.info(
        #                     f"live recovery outer optimizer hash: {get_optimizer_signature(diloco.outer_optimizer)}"
        #                 )
        #                 logger.info(
        #                     f"live recovery outer model hash: {get_tensor_list_signature(diloco.param_list_cpu)}"
        #                 )
        #                 logger.info(f"inner optimizer hash: {get_optimizer_signature(inner_optimizer)}")

        #             ckpt_manager.send_ckpt_to_peer(elastic_device_mesh.global_pg, maybe_dest_rank, blocking=True)

        #             elastic_device_mesh.live_recovery.reset()
        #     else:
        #         ## receiving
        #         time_start_live_recovery = time.perf_counter()
        #         logger.info(f"Start live recovery from rank {config.ckpt.live_recovery_rank_src}")

        #         ## we create grad buffer and opts stats mamnually, the value will be overwritten by the ckpt but we need the DTensor to be correctly init before loading it

        #         diloco.outer_optimizer.step()  # need to step to init the DTensor stats

        #         ckpt_manager.recv_ckpt_from_peer(elastic_device_mesh.global_pg)

        #         if config.train.log_model_hash:
        #             logger.info(
        #                 f"live recovery outer optimizer hash: {get_optimizer_signature(diloco.outer_optimizer)}"
        #             )
        #             logger.info(f"live recovery outer model hash: {get_tensor_list_signature(diloco.param_list_cpu)}")
        #             logger.info(f"inner optimizer hash: {get_optimizer_signature(inner_optimizer)}")

        #         need_live_recovery = False

        #         if config.ckpt.remote_data_load:
        #             ckpt_manager.remote_data_load()

        #         logger.info("live recovery done in %f", time.perf_counter() - time_start_live_recovery)

        # at the beginning of the inner steps we allow joiner to arrive.
        # We maybe reinit before the all reduce but only to allow leaving, not to join anymore

        if world_info.rank == 0 and config.monitor is not None:
            monitor.set_stage("inner_loop")

        for inner_step in range(num_inner_steps):
            loss_batch = 0
            z_loss_batch = 0

            for grad_acc_step in range(gradient_accumulation_steps):
                is_accumulating = grad_acc_step < gradient_accumulation_steps - 1
                # no sync if we are accumulating gradients
                model.set_requires_gradient_sync(not is_accumulating)

                batch = next(train_dataloader_iterator)
                input_ids = batch["input_ids"].to("cuda")
                labels = batch["labels"].to("cuda")
                if config.train.sequence_packing:
                    seqlens = batch["seqlens"].to("cuda")
                    # seqlens has a dynamic shape but fixed dimension, this allow to still torch compile
                    # https://pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html
                    torch._dynamo.mark_dynamic(seqlens, 0)
                else:
                    seqlens = None

                logits = model(tokens=input_ids, seqlens=seqlens).contiguous()
                flatten_logits = rearrange(logits, "b seq vocab -> (b seq) vocab")
                flatten_labels = rearrange(labels, "b seq -> (b seq)")

                if config.optim.z_loss:
                    ce_loss, z_loss = cross_entropy_max_z_loss(
                        flatten_logits, flatten_labels, config.optim.z_loss_weight
                    )
                    ce_loss /= gradient_accumulation_steps
                    z_loss /= gradient_accumulation_steps

                    del logits
                    loss = ce_loss + z_loss
                    loss.backward()

                else:
                    loss = F.cross_entropy(flatten_logits, flatten_labels) / gradient_accumulation_steps
                    del logits
                    loss.backward()

                if config.optim.z_loss:
                    loss_batch += ce_loss.clone().detach()
                    z_loss_batch += z_loss.clone().detach()
                else:
                    loss_batch += loss.clone().detach()

            dist.all_reduce(tensor=loss_batch, op=dist.ReduceOp.AVG)
            if config.optim.z_loss:
                dist.all_reduce(tensor=z_loss_batch, op=dist.ReduceOp.AVG)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            inner_optimizer.step()
            scheduler.step()
            inner_optimizer.zero_grad()

            # logging
            training_progress.step += 1
            inner_lr = [group["lr"] for group in inner_optimizer.param_groups][0]

            # syncing loss across all data parallel rank within a nodes

            new_tokens = config.data.seq_length * config.optim.batch_size
            perf_counter.count_tokens(new_tokens)

            if config.diloco is None:
                training_progress.total_tokens += new_tokens
            else:
                # we count the total tokens with respect to all diloco workers
                # might need to tweak this as some worker might fail to join the all reduce later

                # training_progress.total_tokens += new_tokens * elastic_device_mesh.global_pg.size() # todo need to know the size
                training_progress.total_tokens += new_tokens

            remaining_cpu_ram = psutil.virtual_memory().available / (1024 * 1024 * 1024)

            metrics = {
                "Loss": loss_batch.item(),
                "step": training_progress.step,
                "inner_lr": inner_lr,
                "Perplexity": torch.exp(loss_batch).item(),
                "total_tokens": training_progress.total_tokens,
                "time": time.time(),
                "remaining_cpu_ram": remaining_cpu_ram,
            }

            if config.optim.z_loss:
                metrics["z_loss"] = z_loss_batch.item()

            if config.train.memory_monitor:
                peak_gpu_stats = gpu_mem_monitor.get_peak_stats()
                metrics.update(peak_gpu_stats)

            log = f"step: {training_progress.step}, loss: {loss_batch.item():.4f}"

            tokens_per_second = perf_counter.get_tokens_per_second()

            if tokens_per_second is not None:
                metrics["tokens_per_second"] = tokens_per_second
                metrics["mfu"] = (
                    100 * num_flop_per_token * tokens_per_second / gpu_peak_flops / world_info.local_world_size
                )
                log += f", tokens_per_second: {tokens_per_second:.2f}, mfu: {metrics['mfu']:.2f}"

            if config.diloco is not None:
                metrics["num_peers"] = 1  # elastic_device_mesh.global_pg.size()
                log += f", diloco_peers: {metrics['num_peers']}"

            if world_info.rank == 0:
                metric_logger.log(metrics)
                if config.monitor is not None:
                    monitor.log(metrics)

            logger.info(log)

            if config.train.memory_profiler is not None:
                memory_profiler.step()

        if config.diloco is not None:
            if config.train.log_model_hash:
                logger.debug("Pre diloco model: %s", get_module_signature(model))

            if world_info.rank == 0 and config.monitor is not None:
                monitor.set_stage("outer_loop")

            time_start_inner = time.perf_counter()
            diloco.step(model=model, flag=training_progress.outer_step)
            diloco_time = time.perf_counter() - time_start_inner

            if config.train.log_model_hash:
                logger.debug("inner diloco model: %s", get_module_signature(model))
                logger.debug(f"outer diloco optimizer hash: {get_optimizer_signature(diloco.outer_optimizer)}")
                logger.debug(f"outer diloco optimizer hash: {get_optimizer_signature(diloco.outer_optimizer)}")
                logger.debug(f"outer diloco model hash: {get_tensor_list_signature(diloco.param_list_cpu)}")

        training_progress.outer_step += 1

        if (
            config.ckpt.interval is not None
            and training_progress.step > 0
            and training_progress.step % config.ckpt.interval == 0
        ):
            # we only allow to checkpoint after a outer step. For non diloco training outer step = 1 anyway

            do_remote = config.ckpt.remote is not None and training_progress.step % config.ckpt.remote.interval == 0
            ckpt_manager.save(remote=do_remote)
            if config.train.log_model_hash:
                logger.debug("Post saved model: %s", get_module_signature(model))
                logger.debug("Post saved optimizer: %s", get_optimizer_signature(inner_optimizer))

                if config.diloco is not None:
                    logger.debug("Post saved outer model: %s", get_tensor_list_signature(diloco.param_list_cpu))
                    logger.debug("optimizer hash: %s", get_optimizer_signature(diloco.outer_optimizer))

        if config.diloco:
            tokens_per_second = (
                config.optim.batch_size
                * config.diloco.inner_steps
                * config.data.seq_length
                / (time.perf_counter() - time_start_outer)
            )
            mfu = 100 * num_flop_per_token * tokens_per_second / gpu_peak_flops / world_info.local_world_size
            logger.info(f"effective mfu: {mfu}")

            if world_info.rank == 0:
                metric_logger.log(
                    {
                        "outer_mfu": mfu,
                        "step": training_progress.step,
                        "outer_step": training_progress.outer_step,
                        "outer_tokens_per_second": tokens_per_second,
                        "all_reduce_step": diloco_time,
                    }
                )

        if config.train.memory_monitor:
            logger.info(f"outer step peak gpu stats: {gpu_mem_monitor.format_peak_states()}")

        if training_progress.step >= config.optim.total_steps:
            # we only allow to break outisde of the inner loop.
            # This avoid ending the training in the middle of a the inner loop
            # Since ckpt strategy and all reduce is done at the outer loop level.
            break

    if world_info.rank == 0:
        metric_logger.finish()
        if config.monitor is not None:
            monitor.finish()

    ckpt_manager.wait_for_blocking_job()

    del pccl_communicator  # allow to clean up for smoother tests transition

    logger.info("Training finished, exiting ...")


if __name__ == "__main__":
    # Allow eager fallback during production so that that the training runs dont die
    # However, in development, we want to know that we broke torch compile
    torch._dynamo.config.suppress_errors = "ZERO_BAND_DEV" not in os.environ
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)

    world_info = get_world_info()
    logger = get_logger()

    torch.cuda.set_device(world_info.local_rank)

    config = Config(**parse_argv())
    logger.debug(f"config: {config.model_dump()}")

    try:
        train(config)
    except Exception as e:
        # Subprocesses can prevent the main process from exiting, so we need to terminate them
        for p in _children:
            p.terminate()

        raise e
