import os
import time
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy  # type: ignore
from torch.autograd.profiler import record_function

from zeroband.checkpoint import TrainingProgress, load_checkpoint_fsdp_state, save_checkpoint_fsdp_state
from zeroband.config import Config
from zeroband.data import TEST_VOCAB_SIZE, get_dataloader
from zeroband.loss import compute_cross_entropy_loss
from zeroband.lr_scheduler import get_scheduler
from zeroband.models.llama import get_model
from zeroband.models.llama.model import create_block_mask_from_seqlens
from zeroband.optimizers import get_optimizer
from zeroband.utils import (
    FakeTokenizer,
    PerfCounter,
    get_peak_flops,
    get_num_params,
    get_num_flop_per_token,
)
from zeroband.utils.metric_logger import WandbMetricLogger, DummyMetricLogger
from zeroband.utils.activation_ckpt import apply_ac_ckpt
from zeroband.utils.profiler import MemoryProfiler
from zeroband.utils.world_info import WorldInfo, get_world_info
from zeroband.utils.logger import get_logger
from zeroband.utils.stopwatch import Stopwatch

from transformers import AutoTokenizer
from pydantic_config import parse_argv


def get_gradient_accumulation_steps(batch_size: int, micro_bs: int, world_info: WorldInfo) -> int:
    assert batch_size % world_info.world_size == 0
    batch_size = batch_size // world_info.world_size

    assert batch_size % micro_bs == 0, str(
        f"The micro batch size ({micro_bs}) must divide the number of samples on each GPU ({batch_size})"
    )

    return batch_size // micro_bs


def train(config: Config):
    gradient_accumulation_steps = get_gradient_accumulation_steps(
        config.optim.batch_size, config.train.micro_bs, world_info
    )

    sw = Stopwatch(config)
    sw.start("train()")

    # Load tokenizer
    if config.data.fake and config.name_model == "debugmodel":
        tokenizer = FakeTokenizer()
    elif config.type_model == "llama2":
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)
    elif config.type_model == "llama3":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_fast=True)
    else:
        raise ValueError(f"Model type {config.type_model} not supported")

    train_dataloader = get_dataloader(
        tokenizer=tokenizer,
        world_size=world_info.world_size,
        rank=world_info.rank,
        batch_size=config.train.micro_bs,
        data_config=config.data,
    )
    train_dataloader_iterator = iter(train_dataloader)

    with sw.record_block("Get Model"):
        model, model_config = get_model(
            config,
            vocab_size=len(tokenizer) if config.name_model != "debugmodel" or not config.data.fake else TEST_VOCAB_SIZE,
        )

    gpu_peak_flops = get_peak_flops(torch.cuda.get_device_name(torch.device("cuda")))
    logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")

    num_params = get_num_params(model, exclude_embedding=True)
    logger.info(f"Number of parameters: {num_params}")
    num_flop_per_token = get_num_flop_per_token(
        num_params,
        model_config,
        config.data.seq_length,
    )

    with sw.record_block("Shard Model"):
        if config.train.ac_ckpt:
            num = 1 if isinstance(config.train.ac_ckpt, bool) else config.train.ac_ckpt
            apply_ac_ckpt(model, num)

        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32 if config.train.reduce_fp32 else None
        )

        offload_policy = CPUOffloadPolicy(pin_memory=True) if config.train.fsdp_cpu_offload else None

        for layer_id, transformer_block in model.layers.items():
            if config.train.reshard_after_forward:
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
            reshard_after_forward=config.train.reshard_after_forward,
            offload_policy=offload_policy,
        )

    # Setup optimizers
    with sw.record_block("Optimizer Setup"):
        inner_optimizer = get_optimizer(config, model.parameters())

        # TODO MIKE use pccl instead of elastic_device_mesh

        if config.diloco:
            raise NotImplementedError("Diloco is not implemented yet")

        scheduler = get_scheduler(
            sched_type=config.optim.sched_type,
            optimizer=inner_optimizer,
            num_warmup_steps=config.optim.warmup_steps,
            num_stable_steps=config.optim.stable_steps,
            num_training_steps=config.optim.total_steps,
        )

        training_progress = TrainingProgress(total_tokens=0, outer_step=0, step=0)

    if world_info.rank == 0:
        logger_cls = WandbMetricLogger if config.metric_logger_type == "wandb" else DummyMetricLogger
        metric_logger = logger_cls(
            project=config.project,
            logger_config={"config": config.model_dump(), "world_info": world_info.json()},
            resume=config.wandb_resume,
        )
    else:
        metric_logger = None

    with sw.record_block("Compile Model"):
        if config.train.torch_compile:
            # we need to compile AFTER creating the CKPT manager, DON'T ASK ME WHY
            model = torch.compile(model) if not TYPE_CHECKING else model

    if config.ckpt.resume is not None:
        with sw.record_block("Resume Checkpoint"):
            # all is inplace
            load_checkpoint_fsdp_state(
                model=model,
                optimizers=[inner_optimizer],
                training_progress=training_progress,
                dataloader=train_dataloader,
                scheduler=scheduler,
                path_root=config.ckpt.path,
            )

    if config.train.memory_profiler is not None:
        memory_profiler = MemoryProfiler(config.train.memory_profiler.freq, config.train.memory_profiler.snapshot_dir)

    num_inner_steps = config.diloco.inner_steps if config.diloco is not None else 1
    perf_counter = PerfCounter(window_size=10)

    logger.debug("Finished setup in %f seconds", sw.elapsed())

    while True:
        if num_inner_steps > 1:
            # if we don't use diloco we don't print the outer step logs
            logger.info(f"outer_step step: {training_progress.outer_step}")

        for _inner_step in range(num_inner_steps):
            sw.start("inner_step")

            loss_batch = 0
            z_loss_batch = 0

            with sw.record_block("Grad Acc Steps"):
                for grad_acc_step in range(gradient_accumulation_steps):
                    sw.start("grad_acc_step")

                    is_accumulating = grad_acc_step < gradient_accumulation_steps - 1
                    # no sync if we are accumulating gradients
                    model.set_requires_gradient_sync(not is_accumulating)

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
                        ce_loss, z_loss = compute_cross_entropy_loss(
                            flatten_logits,
                            flatten_labels,
                            z_weight=config.optim.z_loss_weight if config.optim.z_loss else None,
                            num_chunks=config.optim.num_chunks,
                            fused_linear_weight=model.output.weight if config.train.fused_linear_ce else None,
                        )

                        del logits
                        del flatten_logits
                        del flatten_labels

                        if config.optim.z_loss:
                            assert z_loss is not None
                            ce_loss /= gradient_accumulation_steps
                            z_loss /= gradient_accumulation_steps
                            loss = ce_loss + z_loss
                        else:
                            loss = ce_loss / gradient_accumulation_steps

                    with sw.record_block("Run backward()"):
                        loss.backward()

                    with record_function("Clone Loss"):
                        # No need to time, takes 0 seconds
                        if config.optim.z_loss:
                            assert z_loss is not None
                            loss_batch += ce_loss.detach().clone()
                            z_loss_batch += z_loss.detach().clone()
                        else:
                            loss_batch += loss.detach().clone()

            with sw.record_block("Loss allreduce()"):
                # Launch both allreduces at the same time to hide latency
                dist.all_reduce(tensor=loss_batch, op=dist.ReduceOp.AVG)
                if config.optim.z_loss:
                    dist.all_reduce(tensor=z_loss_batch, op=dist.ReduceOp.AVG)

            with sw.record_block("Clip Grad"):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).full_tensor()  # type: ignore (is a dtensor)

            with sw.record_block("Optimizer Step"):
                inner_optimizer.step()
                scheduler.step()

            with sw.record_block("Optimizer Zero Grad"):
                inner_optimizer.zero_grad()

            # logging
            training_progress.step += 1
            inner_lr = [group["lr"] for group in inner_optimizer.param_groups][0]

            # syncing loss across all data parallel rank within a nodes
            new_tokens = config.data.seq_length * config.optim.batch_size
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

            if config.optim.z_loss:
                assert isinstance(z_loss_batch, torch.Tensor)
                metrics["z_loss"] = z_loss_batch.item()

            log = f"step: {training_progress.step}, loss: {loss_batch.item():.4f}"

            tokens_per_second = perf_counter.get_tokens_per_second()
            if tokens_per_second is not None:
                metrics["tokens_per_second"] = tokens_per_second
                metrics["mfu"] = (
                    100 * num_flop_per_token * tokens_per_second / gpu_peak_flops / world_info.local_world_size
                )
                log += f", tokens_per_second: {tokens_per_second:.2f}, mfu: {metrics['mfu']:.2f}"

            if config.diloco is not None:
                # TODO MIKE use pccl instead of elastic_device_mesh
                # metrics["num_peers"] = elastic_device_mesh.global_pg.size()

                metrics["num_peers"] = 1
                log += f", diloco_peers: {metrics['num_peers']}"

            if world_info.rank == 0:
                assert metric_logger is not None
                metric_logger.log(metrics)

            logger.info(log)

            if config.train.memory_profiler is not None:
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
                scheduler=scheduler,
                path_root=config.ckpt.path,
            )

        if training_progress.step >= config.optim.total_steps:
            # we only allow to break outisde of the inner loop.
            # This avoid ending the training in the middle of a the inner loop
            # Since ckpt strategy and all reduce is done at the outer loop level.
            break

    if world_info.rank == 0:
        assert metric_logger is not None
        metric_logger.finish()

    if config.train.memory_profiler is not None:
        logger.debug(f"Max memory used: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

    logger.info("Training finished, exiting ...")


if __name__ == "__main__":
    # Allow eager fallback during production so that that the training runs dont die
    # However, in development, we want to know that we broke torch compile
    torch._dynamo.config.suppress_errors = "ZERO_BAND_DEV" not in os.environ  # type: ignore
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)

    config = Config(**parse_argv())  # type: ignore
    world_info = get_world_info()
    logger = get_logger(config)

    # torch.set_default_device("cuda")
    torch.cuda.set_device(world_info.local_rank)

    train(config)
