import os
import time
from typing import TYPE_CHECKING
from multiprocessing.process import _children  # type: ignore

import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy  # type: ignore
from torch.autograd.profiler import record_function

from zeroband.checkpoint import CkptManager, TrainingProgress
from zeroband.comms import ElasticDeviceMesh
from zeroband.config import Config, resolve_env_vars
from zeroband.data import TEST_VOCAB_SIZE, get_dataloader
from zeroband.diloco import Diloco
from zeroband.loss import compute_cross_entropy_loss
from zeroband.lr_scheduler import get_scheduler
from zeroband.models.llama import get_model
from zeroband.models.llama.model import create_block_mask_from_seqlens
from zeroband.optimizers import get_optimizer
from zeroband.utils import (
    FakeTokenizer,
    PerfCounter,
    get_module_signature,
    get_optimizer_signature,
    get_tensor_list_signature,
    get_peak_flops,
    get_num_params,
    get_num_flop_per_token,
)
from zeroband.utils.metric_logger import MetricLogger, WandbMetricLogger, DummyMetricLogger
from zeroband.utils.activation_ckpt import apply_ac_ckpt
from zeroband.utils.profiler import MemoryProfiler
from zeroband.utils.world_info import get_world_info
from zeroband.utils.logger import get_logger
from zeroband.utils.stopwatch import Stopwatch

from transformers import AutoTokenizer
from pydantic_config import parse_argv


def log_hash_training_state(
    config: Config,
    model: torch.nn.Module,
    inner_optimizer: torch.optim.Optimizer,
    diloco: Diloco | None,
    metric_logger: MetricLogger | None,
    step: int,
    id: str = "",
):
    """Log the hash of the model and optimizer. This function is slow"""
    if config.train.log_model_hash:
        inner_model_hash = get_module_signature(model)
        inner_optimizer_hash = get_optimizer_signature(inner_optimizer)

        logger.debug(f"inner diloco model {id} : {inner_model_hash}")
        logger.debug(f"inner optimizer hash {id} : {inner_optimizer_hash}")

        metrics = {
            "step": step,
            f"inner_model_hash_{id}": inner_model_hash,
            f"inner_optimizer_hash_{id}": inner_optimizer_hash,
        }

        if config.diloco is not None and diloco is not None:
            outer_optimizer_hash = get_optimizer_signature(diloco.outer_optimizer)
            outer_model_hash = get_tensor_list_signature(diloco.param_list_cpu)  # type: ignore

            logger.debug(f"outer diloco optimizer hash {id} : {outer_optimizer_hash}")
            logger.debug(f"outer diloco model hash {id} : {outer_model_hash}")

            metrics.update(
                {f"outer_optimizer_hash_{id}": outer_optimizer_hash, f"outer_model_hash_{id}": outer_model_hash}
            )
        if world_info.rank == 0:
            assert metric_logger is not None
            metric_logger.log(metrics)


def train(config: Config):
    # batch_size is the total batch size for all GPUs
    assert config.optim.batch_size % world_info.local_world_size == 0
    batch_size = config.optim.batch_size // world_info.local_world_size

    assert batch_size % config.train.micro_bs == 0, (
        f"The micro batch size ({config.train.micro_bs}) must divide the number of samples on each GPU ({batch_size})."
    )
    gradient_accumulation_steps = batch_size // config.train.micro_bs

    if config.ckpt is not None and config.ckpt.interval is not None and config.diloco is not None:
        assert config.ckpt.interval % config.diloco.inner_steps == 0, (
            "ckpt interval must be a multiple of diloco inner steps as we only save at the end of an outer step"
        )

    sw = Stopwatch(config)
    sw.start("train()")

    # Load tokenizer
    with sw.record_block("Load Tokenizer"):
        if config.data.fake and config.name_model == "debugmodel":
            tokenizer = FakeTokenizer()
        elif config.type_model == "llama2":
            tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)
        elif config.type_model == "llama3":
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_fast=True)
        else:
            raise ValueError(f"Model type {config.type_model} not supported")

    with sw.record_block("Get Dataloader"):
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

        elastic_device_mesh = ElasticDeviceMesh(
            enable=config.diloco is not None, live_recovery_rank_src=config.ckpt.live_recovery_rank_src
        )

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
                mesh=elastic_device_mesh.cuda_local_mesh,
                reshard_after_forward=reshard_after_forward,
                offload_policy=offload_policy,
            )
        fully_shard(
            model,
            mp_policy=mp_policy,
            mesh=elastic_device_mesh.cuda_local_mesh,
            reshard_after_forward=config.train.reshard_after_forward,
            offload_policy=offload_policy,
        )

    # Setup optimizers
    with sw.record_block("Optimizer Setup"):
        inner_optimizer = get_optimizer(config, model.parameters())

        diloco = Diloco(config.diloco, model, elastic_device_mesh) if config.diloco is not None else None

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
            diloco_offloaded_optimizer=diloco.outer_optimizer if config.diloco is not None else None,  # type: ignore
            diloco_offloaded_param_list=diloco.param_list_cpu if config.diloco is not None else None,  # type: ignore
        )

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
            ckpt_manager.load(
                resume_ckpt_path=config.ckpt.resume,
                skip_dataloader=config.ckpt.skip_dataloader,
                data_path=config.ckpt.data_path,
            )
            log_hash_training_state(
                config, model, inner_optimizer, diloco, metric_logger, step=training_progress.step, id="resume"
            )

    if config.train.memory_profiler is not None:
        memory_profiler = MemoryProfiler(config.train.memory_profiler.freq, config.train.memory_profiler.snapshot_dir)

    num_inner_steps = config.diloco.inner_steps if config.diloco is not None else 1
    perf_counter = PerfCounter(window_size=10)

    logger.debug("Finished setup in %f seconds", sw.elapsed())

    need_live_recovery = config.ckpt.live_recovery_rank_src is not None
    while True:
        if num_inner_steps > 1:
            # if we don't use diloco we don't print the outer step logs
            logger.info(f"outer_step step: {training_progress.outer_step}")

        time_start_outer = time.perf_counter()

        if config.diloco is not None:
            assert diloco is not None
            # this is a patch for now to allow live recovery worker to not affect the all reduce at all

            if not need_live_recovery:
                elastic_device_mesh.maybe_reinit_global_pg(admit_joiners=True)

                maybe_dest_rank = elastic_device_mesh.live_recovery.should_send_ckpt_to()
                if maybe_dest_rank is not None:
                    logger.info(f"Start live recovery to rank {maybe_dest_rank}")
                    ckpt_manager.send_ckpt_to_peer(elastic_device_mesh.global_pg, maybe_dest_rank, blocking=True)

                    elastic_device_mesh.live_recovery.reset()
            else:
                ## receiving
                time_start_live_recovery = time.perf_counter()
                logger.info(f"Start live recovery from rank {config.ckpt.live_recovery_rank_src}")

                ## we create grad buffer and opts stats mamnually, the value will be overwritten by the ckpt but we need the DTensor to be correctly init before loading it

                diloco.outer_optimizer.step()  # need to step to init the DTensor stats

                ckpt_manager.recv_ckpt_from_peer(elastic_device_mesh.global_pg)

                log_hash_training_state(
                    config,
                    model,
                    inner_optimizer,
                    diloco,
                    metric_logger,
                    step=training_progress.step,
                    id="live_reco_recv",
                )
                need_live_recovery = False

                if config.ckpt.remote_data_load:
                    ckpt_manager.remote_data_load()

                logger.info("live recovery done in %f", time.perf_counter() - time_start_live_recovery)

        # at the beginning of the inner steps we allow joiner to arrive.
        # We maybe reinit before the all reduce but only to allow leaving, not to join anymore

        for inner_step in range(num_inner_steps):
            logger.debug("Starting inner step.")
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
                        # TODO/NOTE: We could overlap sending the batch with communication
                        #            although to be honest the perf impact is minimal
                        batch = next(train_dataloader_iterator)
                        input_ids = batch["input_ids"].to("cuda")
                        labels = batch["labels"].to("cuda")
                        if config.train.sequence_packing:
                            seqlens = [seqlen.to("cuda") for seqlen in batch["seqlens"]]
                            block_mask = create_block_mask_from_seqlens(seqlens) if seqlens is not None else None
                        else:
                            seqlens = None
                            block_mask = None

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

                    elapsed = sw.stop("grad_acc_step")
                    logger.debug(f"Grad acc step {grad_acc_step} completed in {elapsed:.2f} seconds")

            with sw.record_block("Loss allreduce()"):
                # Launch both allreduces at the same time to hide latency
                loss_allreduce = dist.all_reduce(
                    tensor=loss_batch, op=dist.ReduceOp.AVG, group=elastic_device_mesh.local_pg, async_op=True
                )
                if config.optim.z_loss:
                    z_loss_allreduce = dist.all_reduce(
                        tensor=z_loss_batch, op=dist.ReduceOp.AVG, group=elastic_device_mesh.local_pg, async_op=True
                    )

                assert isinstance(loss_allreduce, torch.distributed.Work)
                loss_allreduce.wait()
                if config.optim.z_loss:
                    assert isinstance(z_loss_allreduce, torch.distributed.Work)
                    z_loss_allreduce.wait()

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
            else:
                # we count the total tokens with respect to all diloco workers
                # might need to tweak this as some worker might fail to join the all reduce later
                training_progress.total_tokens += new_tokens * elastic_device_mesh.global_pg.size()

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
                metrics["num_peers"] = elastic_device_mesh.global_pg.size()
                log += f", diloco_peers: {metrics['num_peers']}"

            if world_info.rank == 0:
                assert metric_logger is not None
                metric_logger.log(metrics)

            logger.info(log)

            if config.train.memory_profiler is not None:
                memory_profiler.step()

            elapsed = sw.stop("inner_step")
            logger.debug(f"Inner step {inner_step} completed in {elapsed:.2f} seconds")

        if config.diloco is not None:
            assert diloco is not None
            time_start_inner = time.perf_counter()
            diloco.step(model=model, flag=str(training_progress.outer_step))
            diloco_time = time.perf_counter() - time_start_inner

            log_hash_training_state(
                config, model, inner_optimizer, diloco, metric_logger, step=training_progress.step, id="outer_step"
            )

        training_progress.outer_step += 1

        if (
            config.ckpt.interval is not None
            and training_progress.step > 0
            and training_progress.step % config.ckpt.interval == 0
        ):
            # we only allow to checkpoint after a outer step. For non diloco training outer step = 1 anyway

            do_remote = config.ckpt.remote is not None and training_progress.step % config.ckpt.remote.interval == 0
            ckpt_manager.save(remote=do_remote)
            log_hash_training_state(
                config, model, inner_optimizer, diloco, metric_logger, step=training_progress.step, id="save"
            )

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
                assert metric_logger is not None
                metric_logger.log(
                    {
                        "outer_mfu": mfu,
                        "step": training_progress.step,
                        "outer_step": training_progress.outer_step,
                        "outer_tokens_per_second": tokens_per_second,
                        "all_reduce_step": diloco_time,
                    }
                )

        if training_progress.step >= config.optim.total_steps:
            # we only allow to break outisde of the inner loop.
            # This avoid ending the training in the middle of a the inner loop
            # Since ckpt strategy and all reduce is done at the outer loop level.
            break

    if world_info.rank == 0:
        assert metric_logger is not None
        metric_logger.finish()

    ckpt_manager.wait_for_blocking_job()

    del elastic_device_mesh  # allow to clean up for smoother tests transition

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
    resolve_env_vars(config)
    world_info = get_world_info()
    logger = get_logger(config)

    # torch.set_default_device("cuda")
    torch.cuda.set_device(world_info.local_rank)

    def pretty_dict(d, indent=2):
        for key, value in d.items():
            if isinstance(value, dict):
                logger.debug(" " * indent + f"{key}:")
                pretty_dict(value, indent + 2)
            else:
                logger.debug(" " * indent + f"{key}: {value}")

    logger.debug("config:")
    pretty_dict(config.model_dump())

    try:
        if config.train.torch_profiler and world_info.rank == 0:
            # NOTE(apaz-cli): I cannot seem to get the memory profiler to work.
            # Running into this issue: https://github.com/pytorch/pytorch/issues/64345
            # In the meantime, we can use the memory snapshotter.

            logger.debug("Running train() with profiler.")
            prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                # profile_memory=True,
                # with_stack=True,
            )
            try:
                prof.__enter__()
                train(config)
            finally:
                logger.debug("Exiting profiler context.")
                prof.__exit__(None, None, None)

            logger.info("Exporting chrome trace.")
            prof.export_chrome_trace("logs/profile.json.gz")

            width = 30
            logger.info("\n" + "*" * width + " GPU TIME " + "*" * width)
            logger.info(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

            logger.info("\n" + "*" * width + " GPU MEM " + "*" * width)
            logger.info(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

            # logger.info("Exporting memory timeline.")
            # prof.export_memory_timeline(f"logs/mem_timeline.html", device="cuda:0")
        else:
            train(config)
    except Exception as e:
        # Subprocesses can prevent the main process from exiting, so we need to terminate them
        logger.info("Caught an exception, terminating children")
        logger.info(e)
        for p in _children:
            p.terminate()

        raise e
