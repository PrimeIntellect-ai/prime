import os
import time
from logging import Logger
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from torch.distributed import destroy_process_group
from torch.distributed.tensor import DTensor
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy  # type: ignore

import wandb

from zeroband.checkpoint import TrainingProgress, load_checkpoint_fsdp_state, save_checkpoint_fsdp_state
from zeroband.config import Config
from zeroband.data import DEBUG_VOCAB_SIZE, get_dataloader
from zeroband.lr_scheduler import compute_current_lr
from zeroband.models.llama import make_model
from zeroband.models.llama.model import create_block_mask_from_seqlens
from zeroband.utils import (
    PerfCounter,
    optim_utils
)
from zeroband.utils.activation_ckpt import apply_ac_ckpt
from zeroband.utils.profiler import MemoryProfiler
from zeroband.utils.tokenizer_utils import make_tokenizer
from zeroband.utils.world_info import WorldInfo, get_world_info
from zeroband.utils.logger import get_logger
from zeroband.utils.stopwatch import Stopwatch

from pydantic_config import parse_argv

def calc_gradient_accumulation_steps(batch_size: int, micro_bs: int, world_info: WorldInfo) -> int:
    assert batch_size % world_info.world_size == 0
    batch_size = batch_size // world_info.world_size

    assert batch_size % micro_bs == 0, str(
        f"The micro batch size ({micro_bs}) must divide the number of samples on each GPU ({batch_size})"
    )

    return batch_size // micro_bs


def train(logger: Logger, config: Config, world_info: WorldInfo):
    grad_accum_steps = calc_gradient_accumulation_steps(
        config.train.batch_size, config.hardware.micro_batch_size, world_info
    )

    sw = Stopwatch(config)
    sw.start("train()")

    # Load tokenizer
    tokenizer = make_tokenizer(config)

    train_dataloader = get_dataloader(
        tokenizer=tokenizer,
        world_size=world_info.world_size,
        rank=world_info.rank,
        batch_size=config.hardware.micro_batch_size,
        data_config=config.data,
    )
    train_dataloader_iterator = iter(train_dataloader)

    with sw.record_block("Get Model"):
        model, model_config = make_model(
            config,
            vocab_size=len(tokenizer),
        )

    perf_counter = PerfCounter(window_size=10, model=model, model_config=model_config, seq_len=config.data.seq_length)

    logger.info(f"Number of parameters: {perf_counter.num_params}")

    with sw.record_block("Shard Model"):
        if config.hardware.act_ckpt:
            num = 1 if isinstance(config.hardware.act_ckpt, bool) else config.hardware.act_ckpt
            apply_ac_ckpt(model, num)

        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32 if config.hardware.reduce_fp32 else None
        )

        offload_policy = CPUOffloadPolicy(pin_memory=True) if config.hardware.fsdp_cpu_offload else None

        for layer_id, transformer_block in model.layers.items():
            if config.hardware.reshard_after_forward:
                reshard_after_forward = int(layer_id) < len(model.layers) - 1
            else:
                reshard_after_forward = False
            fully_shard(
                transformer_block,
                mp_policy=mp_policy,
                reshard_after_forward=reshard_after_forward,
                offload_policy=offload_policy,
            )
        fully_shard(
            model,
            mp_policy=mp_policy,
            reshard_after_forward=config.hardware.reshard_after_forward,
            offload_policy=offload_policy,
        )

    # Setup optimizers
    with sw.record_block("Optimizer Setup"):
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

    with sw.record_block("Compile Model"):
        if config.hardware.torch_compile:
            model = torch.compile(model) if not TYPE_CHECKING else model

    if config.ckpt.resume is not None:
        with sw.record_block("Resume Checkpoint"):
            # all is inplace
            load_checkpoint_fsdp_state(
                model=model,
                optimizers=[inner_optimizer],
                training_progress=training_progress,
                dataloader=train_dataloader,
                path_root=config.ckpt.path,
            )

    if config.hardware.memory_profiler is not None:
        memory_profiler = MemoryProfiler(config.hardware.memory_profiler.freq,
                                         config.hardware.memory_profiler.snapshot_dir)

    num_inner_steps = config.diloco.inner_steps if config.diloco is not None else 1

    logger.debug("Finished setup in %f seconds", sw.elapsed())

    while True:
        if num_inner_steps > 1:
            # if we don't use diloco we don't print the outer step logs
            logger.info(f"outer_step step: {training_progress.outer_step}")

        for _inner_step in range(num_inner_steps):
            sw.start("inner_step")

            loss_batch = 0

            with sw.record_block("Grad Acc Steps"):
                for grad_acc_step in range(grad_accum_steps):
                    sw.start("grad_acc_step")

                    current_lr = compute_current_lr(training_progress.step, config.train.lr_scheduler)
                    optim_utils.set_optimizer_lr(inner_optimizer, current_lr)

                    with sw.record_block("Load batch"):
                        batch = next(train_dataloader_iterator)
                        input_ids = batch["input_ids"].to("cuda")
                        labels = batch["labels"].to("cuda")
                        seqlens = [seqlen.to("cuda") for seqlen in batch["seqlens"]]
                        block_mask = create_block_mask_from_seqlens(seqlens)

                    with sw.record_block("Run forward()"):
                        logits = model(tokens=input_ids, block_mask=block_mask).contiguous()
                        flatten_logits = logits.reshape(-1, logits.size(-1))  # b seq vocab -> (b * seq) vocab
                        flatten_labels = labels.reshape(-1)  # b seq -> (b * seq)

                    with sw.record_block("Loss Calculation"):
                        loss = torch.nn.functional.cross_entropy(flatten_logits, flatten_labels) / grad_accum_steps

                    with sw.record_block("Run backward()"):
                        loss.backward()

                    loss_batch += loss.detach().clone()

            dist.all_reduce(tensor=loss_batch, op=dist.ReduceOp.AVG)

            with sw.record_block("Clip Grad"):
                grad_norm: DTensor = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # type: ignore
                grad_norm = grad_norm.full_tensor()  # type: ignore

            with sw.record_block("Optimizer Step"):
                inner_optimizer.step()

            inner_optimizer.zero_grad()

            # logging
            training_progress.step += 1
            inner_lr = [group["lr"] for group in inner_optimizer.param_groups][0]

            # syncing loss across all data parallel rank within a nodes
            new_tokens = config.data.seq_length * config.train.batch_size
            perf_counter.count_tokens(new_tokens)

            if config.diloco is None:
                training_progress.total_tokens += new_tokens
            # else:
            # we count the total tokens with respect to all diloco workers
            # might need to tweak this as some worker might fail to join the all reduce later

            # TODO MIKE use pccl instead of elastic_device_mesh

            # training_progress.total_tokens += new_tokens * elastic_device_mesh.global_pg.size()

            assert isinstance(loss_batch, torch.Tensor)
            metrics = {
                "Loss": loss_batch.item(),
                "step": training_progress.step,
                "inner_lr": inner_lr,
                "Perplexity": torch.exp(loss_batch).item(),
                "total_tokens": training_progress.total_tokens,
                "time": time.time(),
                "grad_norm": grad_norm.item(),
            }

            log = f"step: {training_progress.step}, loss: {loss_batch.item():.4f}"

            tokens_per_second = perf_counter.get_tokens_per_second()

            if tokens_per_second is not None:
                metrics["inner_lr"] = inner_lr
                metrics["tokens_per_second"] = tokens_per_second
                metrics["mfu"] = perf_counter.get_mfu()
                log += f", inner_lr: {inner_lr}, tokens_per_second: {tokens_per_second:.2f}, mfu: {metrics['mfu']:.2f}"

            if config.diloco is not None:
                # TODO MIKE use pccl instead of elastic_device_mesh
                # metrics["num_peers"] = elastic_device_mesh.global_pg.size()

                metrics["num_peers"] = 1
                log += f", diloco_peers: {metrics['num_peers']}"

            if world_info.rank == 0 and config.wandb:
                wandb.log(metrics)

            logger.info(log)

            if config.hardware.memory_profiler is not None:
                memory_profiler.step()

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
            # we only allow to break outisde of the inner loop.
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

    train(logger, config, world_info)


if __name__ == "__main__":
    main()
