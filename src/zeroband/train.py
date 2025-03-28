import os
import time
from logging import Logger
from typing import TYPE_CHECKING, Optional, Iterator

import torch
import torch.distributed as dist

from torch.distributed import destroy_process_group
from torch.distributed.tensor import DTensor

import wandb

from zeroband.checkpoint import TrainingProgress, load_checkpoint_fsdp_state, save_checkpoint_fsdp_state
from zeroband.config import Config
from zeroband.data import make_dataloader
from zeroband.lr_scheduler import compute_current_lr
from zeroband.models.llama import make_model
from zeroband.models.llama.model import create_block_mask_from_seqlens
from zeroband.utils import optim_utils, sharding_utils, act_checkpointing, metrics_utils

from zeroband.utils.memory_profiler import MemoryProfiler
from zeroband.utils.mfu_tracker import FlopCounter, PrecisionMode, \
    get_flops_promised_torch
from zeroband.utils.tokenizer_utils import make_tokenizer
from zeroband.utils.world_info import WorldInfo, get_world_info
from zeroband.utils.logger import get_logger
from zeroband.utils.profiler import Profiler, ProfilerCollection

from pydantic_config import parse_argv

PRIME_SETUP_PROFILER_PRINT_TIMINGS: bool = os.getenv("PRIME_SETUP_PROFILER_PRINT_TIMINGS") == "1"
PRIME_TRAIN_PROFILER_PRINT_TIMINGS: bool = os.getenv("PRIME_TRAIN_PROFILER_PRINT_TIMINGS") == "1"
PRIME_TRAIN_PROFILER_EXPORT_VIDEO_INTERVAL: int = int(os.getenv("PRIME_TRAIN_PROFILER_EXPORT_VIDEO_INTERVAL", "-1"))


def calc_gradient_accumulation_steps(batch_size: int, micro_bs: int, world_info: WorldInfo) -> int:
    assert batch_size % world_info.world_size == 0
    batch_size = batch_size // world_info.world_size

    assert batch_size % micro_bs == 0, str(
        f"The micro batch size ({micro_bs}) must divide the number of samples on each GPU ({batch_size})"
    )

    return batch_size // micro_bs


def perform_grad_accum_steps(
        config: Config,
        profiler: Profiler,
        flop_counter: FlopCounter,
        training_progress: TrainingProgress,
        train_dataloader_iterator: Iterator,
        grad_accum_steps: int,
        model: torch.nn.Module,
        inner_optimizer: torch.optim.Optimizer,
        device: torch.device) -> (torch.Tensor, float):
    """
    Performs n gradient accumulated micro-steps and returns the total loss of each step
    :return (total_loss, current_lr)
    """
    total_loss = torch.tensor([0.0], dtype=torch.float32, device=device)
    current_lr = 0.0
    for grad_acc_step in range(grad_accum_steps):
        profiler.start_session("grad_acc_step")

        current_lr = compute_current_lr(training_progress.step, config.train.lr_scheduler)
        optim_utils.set_optimizer_lr(inner_optimizer, current_lr)

        with profiler.session("train_dataloader_iterator.__next__"):
            batch = next(train_dataloader_iterator)
            input_ids = batch["input_ids"].to("cuda")
            labels = batch["labels"].to("cuda")
            seqlens = [seqlen.to("cuda") for seqlen in batch["seqlens"]]
            block_mask = create_block_mask_from_seqlens(seqlens)

        with profiler.session("model.forward"):
            logits = model(tokens=input_ids, block_mask=block_mask, flop_counter=flop_counter)

        with profiler.session("torch::nn::functional::cross_entropy"):
            flatten_logits = logits.view(-1, logits.size(-1))  # b seq vocab -> (b * seq) vocab
            flatten_labels = labels.view(-1)  # b seq -> (b * seq)
            loss = torch.nn.functional.cross_entropy(flatten_logits, flatten_labels) / grad_accum_steps
            flop_counter.track_cross_entropy(flatten_logits)

        with profiler.session("loss.backward"):
            loss.backward()

        total_loss += loss.detach().clone()
        profiler.end_session()

    return total_loss, current_lr


def train(logger: Logger, config: Config, world_info: WorldInfo, device: torch.device):
    grad_accum_steps = calc_gradient_accumulation_steps(
        config.train.batch_size, config.hardware.micro_batch_size, world_info
    )

    setup_profiler = Profiler()

    # Load tokenizer
    tokenizer = make_tokenizer(config)

    train_dataloader = make_dataloader(
        tokenizer=tokenizer,
        world_size=world_info.world_size,
        rank=world_info.rank,
        batch_size=config.hardware.micro_batch_size,
        data_config=config.data,
    )
    train_dataloader_iterator = iter(train_dataloader)

    with setup_profiler.session("::make_model"):
        model, model_config = make_model(
            config,
            vocab_size=len(tokenizer),
        )
        num_param_scalars = model.count_parameters()
        logger.info(f"Number of parameters: {num_param_scalars}")

    if config.hardware.act_ckpt:
        with setup_profiler.session("act_checkpointing::enable_activation_checkpointing"):
            num = 1 if isinstance(config.hardware.act_ckpt, bool) else config.hardware.act_ckpt
            act_checkpointing.enable_activation_checkpointing(model, num)

    with setup_profiler.session("sharding_utils::apply_sharding"):
        sharding_utils.apply_sharding(config, model)

    # Setup optimizers
    with setup_profiler.session("optim_utils::make_optimizer"):
        inner_optimizer = optim_utils.make_optimizer(model, config.train.optimizer)

        # TODO MIKE use pccl instead of elastic_device_mesh

        if config.diloco:
            raise NotImplementedError("Diloco is not implemented yet")

    training_progress = TrainingProgress(total_tokens=0, outer_step=0, step=0)

    if world_info.rank == 0 and config.wandb:
        wandb.init(
            project=config.project,
            config={"config": config.model_dump(), "world_info": world_info.json()},
        )

    with setup_profiler.session("torch::compile"):
        if config.hardware.torch_compile:
            model = torch.compile(model) if not TYPE_CHECKING else model

    if config.ckpt.resume is not None:
        with setup_profiler.session("::load_checkpoint_fsdp_state"):
            # all is inplace
            load_checkpoint_fsdp_state(
                model=model,
                optimizers=[inner_optimizer],
                training_progress=training_progress,
                dataloader=train_dataloader,
                path_root=config.ckpt.path,
            )

    memory_profiler: Optional[MemoryProfiler] = None
    if config.hardware.memory_profiler is not None:
        memory_profiler = MemoryProfiler(config.hardware.memory_profiler.freq,
                                         config.hardware.memory_profiler.snapshot_dir)

    num_inner_steps = config.diloco.inner_steps if config.diloco is not None else 1

    if PRIME_SETUP_PROFILER_PRINT_TIMINGS:
        setup_profiler.print_report()

    train_profiler_collection = ProfilerCollection()

    timing_events = []
    while True:
        train_profiler = Profiler()

        if num_inner_steps > 1:
            # if we don't use diloco we don't print the outer step logs
            logger.info(f"outer_step step: {training_progress.outer_step}")

        for _inner_step in range(num_inner_steps):
            train_profiler.start_session("inner_step")

            flop_counter = FlopCounter()

            start_event = torch.cuda.Event(enable_timing=True, blocking=False)
            end_event = torch.cuda.Event(enable_timing=True, blocking=False)

            start_event.record()

            with train_profiler.session("::perform_grad_accum_steps"):
                loss_batch: torch.Tensor
                inner_lr: float
                loss_batch, inner_lr = perform_grad_accum_steps(config, train_profiler, flop_counter,
                                                                    training_progress,
                                                                    train_dataloader_iterator,
                                                                    grad_accum_steps,
                                                                    model,
                                                                    inner_optimizer,
                                                                    device)

            dist.all_reduce(tensor=loss_batch, op=dist.ReduceOp.AVG)

            with train_profiler.session("torch::nn::utils::clip_grad_norm_"):
                # compute pow, plus (assert clip is rare, no 3N)
                flop_counter.track_backward_flops(2 * num_param_scalars)

                grad_norm: DTensor = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # type: ignore
                grad_norm = grad_norm.full_tensor()  # type: ignore

            with train_profiler.session("inner_optimizer.step"):
                flop_counter.track_optimizer_step(inner_optimizer, num_param_scalars)
                inner_optimizer.step()
            inner_optimizer.zero_grad()

            end_event.record()
            timing_events.append((start_event, end_event))

            # logging
            training_progress.step += 1

            # syncing loss across all data parallel rank within a nodes
            new_tokens = config.data.seq_length * config.train.batch_size

            # find next available timing event from some step in the past
            # that the gpu has already finished executing.
            # Realistically, this should at most be -1 steps into the past
            time_seconds = None
            for pair in timing_events:
                start_event, end_event = pair
                if end_event.query():
                    end_event.synchronize()
                    time_seconds = start_event.elapsed_time(end_event) * 1e-3
                    timing_events.remove(pair)
                    break

            tokens_per_second = None
            if time_seconds is not None:
                tokens_per_second = new_tokens / time_seconds

            if config.diloco is None:
                training_progress.total_tokens += new_tokens

            # else:
            # we count the total tokens with respect to all diloco workers
            # might need to tweak this as some worker might fail to join the all reduce later

            # TODO MIKE use pccl instead of elastic_device_mesh

            # training_progress.total_tokens += new_tokens * elastic_device_mesh.global_pg.size()

            tflops_max = get_flops_promised_torch(device, PrecisionMode.PRECISION_BF16)

            metrics = {
                "loss/train": loss_batch.item(),
                "step": training_progress.step,
                "inner_lr": inner_lr,
                "Perplexity": torch.exp(loss_batch).item(),
                "total_tokens": training_progress.total_tokens,
                "time": time.time(),
                "grad_norm": grad_norm.item(),
                'tflops_max': tflops_max
            }

            if time_seconds is not None:
                tflops_per_second = (flop_counter.get_performed_flops() * 1e-12) / time_seconds
                mfu = (tflops_per_second / tflops_max) * 100.0

                metrics.update({
                    "mfu": mfu,
                    "tflops": tflops_per_second
                })

            metrics.update({
                "inner_lr": inner_lr,
                "tokens_per_second": tokens_per_second
            })

            if config.diloco is not None:
                # TODO MIKE use pccl instead of elastic_device_mesh
                # metrics["num_peers"] = elastic_device_mesh.global_pg.size()
                metrics["num_peers"] = 1

            if world_info.rank == 0 and config.wandb:
                wandb.log(metrics)

            log = metrics_utils.build_metrics_string(metrics, whitelist_keys={'step', 'loss', 'mfu', 'tflops', 'tokens_per_second', 'tflops_max'})
            logger.info(log)

            if memory_profiler is not None:
                memory_profiler.step()
            train_profiler.end_session()

        # post inner steps
        if PRIME_TRAIN_PROFILER_PRINT_TIMINGS:
            train_profiler.print_report()

        export_interval = PRIME_TRAIN_PROFILER_EXPORT_VIDEO_INTERVAL
        if export_interval != -1:
            train_profiler_collection.add_profiler(train_profiler, f'Step {training_progress.outer_step}')

            # this is slightly not nice, but inner steps seems like the better unit to use here
            # despite the fact that we are rendering full outer steps per frame which may or may not be = 1 inner step
            if training_progress.step > 0 and training_progress.step % export_interval == 0:
                train_profiler_collection.render_as_video(f'profiler_video_{training_progress.step}.mp4', fps=10)

        if config.diloco is not None:
            ...
            # diloco.step(model=model, flag=str(training_progress.outer_step))

        training_progress.outer_step += 1

        if (
                config.ckpt.interval is not None
                and training_progress.step > 0
                and training_progress.step % config.ckpt.interval == 0
        ):
            # we only allow to checkpoint after a outer step. For non diloco training outer step = 1 anyway
            save_checkpoint_fsdp_state(
                model=model,
                optimizers=[inner_optimizer],
                training_progress=training_progress,
                dataloader=train_dataloader,
                path_root=config.ckpt.path,
            )

        if training_progress.step >= config.train.lr_scheduler.num_total_steps:
            # we only allow to break outside of the inner loop.
            # This avoid ending the training in the middle of a the inner loop
            # Since ckpt strategy and all reduce is done at the outer loop level.
            break

        if world_info.rank == 0:
            wandb.finish()

    if config.hardware.memory_profiler is not None:
        logger.debug(f"Max memory used: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")

    logger.info("Training finished, exiting ...")
    destroy_process_group()


def main():
    # Allow eager fallback during production so that the training runs don't die
    # However, in development, we want to know that we broke torch compile
    torch._dynamo.config.suppress_errors = "ZERO_BAND_DEV" not in os.environ  # type: ignore
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)

    config = Config(**parse_argv())  # type: ignore
    world_info = get_world_info()
    logger = get_logger(config)

    # torch.set_default_device("cuda")
    torch.cuda.set_device(world_info.local_rank)
    device = torch.device(f'cuda:{torch.cuda.current_device()}')

    train(logger, config, world_info, device)


if __name__ == "__main__":
    main()
