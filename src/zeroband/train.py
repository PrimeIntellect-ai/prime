import os
import threading
import time
from dataclasses import asdict
from logging import Logger
from typing import TYPE_CHECKING, Optional, Iterator, List, Dict, Tuple

import torch
import torch.distributed as dist
from pccl import SharedState, TensorInfo, Attribute, Communicator, PCCLError, ReduceOp

from torch.distributed import destroy_process_group
from torch.distributed.tensor import DTensor

import wandb
from torch.optim import Optimizer

from zeroband.ccl import ccl_utils, pccl_utils
from zeroband.ccl.ccl_utils import MPIConfig
from zeroband.checkpoint import TrainingProgress, load_checkpoint, save_checkpoint, CheckpointInfo
from zeroband.config import Config, LearningRateSchedulerConfig
from zeroband.data import make_dataloader
from zeroband.lr_scheduler import compute_current_lr
from zeroband.models.llama import make_model
from zeroband.models.llama.model import create_block_mask_from_seqlens
from zeroband.utils import optim_utils, sharding_utils, act_checkpointing, metrics_utils, torch_utils

from zeroband.utils.memory_profiler import MemoryProfiler
from zeroband.utils.mfu_tracker import FlopCounter, PrecisionMode, \
    get_flops_promised_pt
from zeroband.utils.misc_utils import IntRef
from zeroband.utils.tokenizer_utils import get_tokenizer_info
from zeroband.utils.logger import get_logger
from zeroband.utils.profiler import Profiler, ProfilerCollection

from pydantic_config import parse_argv

PRIME_SETUP_PROFILER_PRINT_TIMINGS: bool = os.getenv("PRIME_SETUP_PROFILER_PRINT_TIMINGS") == "1"
PRIME_TRAIN_PROFILER_PRINT_TIMINGS: bool = os.getenv("PRIME_TRAIN_PROFILER_PRINT_TIMINGS") == "1"
PRIME_TRAIN_PROFILER_EXPORT_VIDEO_INTERVAL: int = int(os.getenv("PRIME_TRAIN_PROFILER_EXPORT_VIDEO_INTERVAL", "-1"))


def calc_gradient_accumulation_steps(batch_size: int, micro_bs: int, mpi_config: Optional[MPIConfig]) -> int:
    mpi_world_size = mpi_config.mpi_world_size if mpi_config is not None else 1
    assert batch_size % mpi_world_size == 0
    batch_size = batch_size // mpi_world_size

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
            seq_lengths = [seqlen.to("cuda") for seqlen in batch["seqlens"]]
            block_mask = create_block_mask_from_seqlens(seq_lengths)

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


def run_inner_steps(
        model: torch.nn.Module,
        train_dataloader_iterator: iter,
        inner_optimizer: torch.optim.Optimizer,
        device: torch.device,

        logger: Logger,
        memory_profiler: MemoryProfiler,

        local_world_size: int,

        mpi_config: Optional[MPIConfig],
        train_profiler: Profiler,
        config: Config,
        training_progress: TrainingProgress,
        grad_accum_steps: int,

        timing_events: List[Tuple[torch.cuda.Event, torch.cuda.Event]]
):
    num_inner_steps = config.diloco.inner_steps if config.diloco is not None else 1
    num_param_scalars = model.count_parameters()

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

        if mpi_config is not None:
            dist.all_reduce(tensor=loss_batch, op=dist.ReduceOp.AVG)

        with train_profiler.session("torch::nn::utils::clip_grad_norm_"):
            # compute pow, plus (assert clip is rare, no 3N)
            flop_counter.track_backward_flops(2 * num_param_scalars)

            grad_norm: torch.Tensor | DTensor = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                                               1.0)  # type: ignore
            if isinstance(grad_norm, DTensor):
                grad_norm = grad_norm.full_tensor()  # type: ignore

        with train_profiler.session("inner_optimizer.step"):
            flop_counter.track_optimizer_step(inner_optimizer, num_param_scalars)
            inner_optimizer.step()
            inner_optimizer.zero_grad(set_to_none=False)

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
        else:
            training_progress.total_tokens += new_tokens * local_world_size

        tflops_max = get_flops_promised_pt(device, PrecisionMode.PRECISION_BF16)

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
            metrics["num_peers"] = local_world_size

        if (mpi_config is None or mpi_config.mpi_rank == 0) and config.wandb:
            wandb.log(metrics)

        log = metrics_utils.build_metrics_string(metrics, whitelist_keys={'loss/train', 'step', 'loss', 'mfu', 'tflops',
                                                                          'tokens_per_second', 'tflops_max'})
        logger.info(log)

        if memory_profiler is not None:
            memory_profiler.step()
        train_profiler.end_session()


def run_async_outer_step(
        model: torch.nn.Module,
        last_pseudo_grads: List[torch.Tensor],
        outer_parameters_list: List[torch.nn.Parameter],
        outer_optimizer: torch.optim.Optimizer,
        shared_state: SharedState,

        all_reduce_thread: threading.Thread,

        communicator: Communicator,
        train_profiler: Profiler,
        logger: Logger,

        topology_updated: bool,
        iter_num: int,
        num_syncs: IntRef
):
    # await previous all reduce, if one exists
    can_outer_step = False
    if all_reduce_thread is not None:
        all_reduce_thread.join()
        can_outer_step = True

        # populate outer param grads with last pseudo-gradients set by thread
        for pseudo_grad, outer_p in zip(last_pseudo_grads, outer_parameters_list):
            outer_p.grad = pseudo_grad

    # Compute current pseudo grads as difference between outer and inner state.
    # Inner state is advanced by inner steps, outer state is unchanged
    outer_grads: List[torch.Tensor] = []
    param: torch.nn.Parameter  # [ torch.Tensor | DTensor ]
    outer_p: torch.nn.Parameter  # [ torch.Tensor ]
    for param, outer_p in zip(model.parameters(), outer_parameters_list):
        param_data: torch.Tensor | DTensor = param.data
        outer_p_data: torch.Tensor = outer_p.data
        if isinstance(param_data, DTensor):
            param_data = param_data.to_local()
        outer_p.grad = outer_p_data - param_data.to('cpu')
        outer_grads.append(outer_p.grad)

    if can_outer_step:
        outer_optimizer.step()  # Note that there is no zero-grad because grads get re-instantiated every step

        # Copy aggregator result into local model
        sync_inner_with_outer_state(model, outer_parameters_list)

        if topology_updated and iter_num > 0:
            # If the topology was updated and iter_num is > 0
            # then a new peer just joined the run with needs to be properly inserted into
            # the N-1 async pipeline.
            # To do this we first initially sync the weights such that the peer can
            # start computing the current step like the pre-existing peers, however
            # the newly joined peer cannot be "retroactively inserted" into
            # the N-1 async reduce that was started last step.
            # So it needs to "eavesdrop" on the result that the other peers are about to compute
            # with a second shared state re-transmission.
            # Hence, both pre-existing peers and newly joined peer(s) have to perform shared state
            # synchronization.
            # The pre-existing peers first apply the outer optimizer and THEN call run_shared_state_sync
            # because the new peer(s) need to obtain the shared state as it is after the all reduce
            # is applied that they were not part of.
            logger.info(
                "Topology updated mid run; re-running shared state synchronization to properly insert new peer...")
            run_shared_state_sync(shared_state, communicator, model, outer_parameters_list, num_syncs, train_profiler,
                                  False)

    else:
        if topology_updated and iter_num > 0:
            # If the topology was updated and iter_num is > 0 and can_outer_step is False,
            # then WE are the joining peer to an ongoing run.
            # In this case, we have to obtain the shared state from the pre-existing peers.
            # We obtain the shared state first and then simply copy it into the inner model afterwards.
            # Also: late_joiner here means that we tolerate actually receiving bytes here despite that this is the second sync that was performed.
            # This is necessary for the pipeline insertion algorithm to function
            run_shared_state_sync(shared_state, communicator, model, outer_parameters_list, num_syncs, train_profiler,
                                  False)

        # This is the boostrap for the 1-step behind asynchronous training step.
        # Reset the inner state here to be equal to the unmodified outer state.
        # This essentially resets the model back to initialization state.
        # Why do this?
        # a) because the next shared state sync needs to see all outer states as equal.
        # We haven't communicated yet, so we have by definition diverged.
        # But we will hide this for now.
        # b) what we are accomplishing here is as follows:
        # We know that the pseudo-grads constitute a valid update to the weights
        # to decrease the loss when applied to the initial model state.
        # These changes will be applied in the next loop iteration.
        # We will hide the communication with compute of the next iteration.
        # Afterward, we will apply said delta to the still initial weights.
        # At this stage, we haven't done anything questionable at all.
        # We have applied a valid update to exactly the base weights they were grads for.
        # However, now in the next outer step, the reduce of the pseudo-gradients of step two is awaited
        # and these are updates from initial weights also - just derived from different input data.
        # We have already moved on from the initial weights
        # at this point. And yet, we still apply them. This is the 1-step behind assertion
        # that we make that it is reasonable to still apply these gradients, even though they
        # are slightly outdated. From then onwards, outer step updates are always one step behind.
        sync_inner_with_outer_state(model, outer_parameters_list)

    def run_all_reduce():
        nonlocal last_pseudo_grads
        last_pseudo_grads = outer_grads.copy()
        start_time = time.time()
        pccl_utils.all_reduce_multiple_with_retry(
            communicator,
            last_pseudo_grads,
            ReduceOp.AVG
        )
        end_time = time.time()
        print(f"All-Reduce took {end_time - start_time} seconds")

    logger.debug("Launching all reduce...")
    all_reduce_thread = threading.Thread(target=run_all_reduce, name="ReduceThread")

    # NOTE: no zero-grad on outer grads, as they continue to get referenced by this thread.
    all_reduce_thread.start()

    return all_reduce_thread


def run_sync_outer_step(
        model: torch.nn.Module,
        outer_parameters_list: List[torch.nn.Parameter],
        outer_optimizer: torch.optim.Optimizer,
        communicator: Communicator,
        train_profiler: Profiler,
        logger: Logger
):
    # Compute current pseudo grads as difference between outer and inner state.
    # Inner state is advanced by inner steps, outer state is unchanged
    outer_grads: List[torch.Tensor] = []
    param: torch.nn.Parameter  # [ torch.Tensor | DTensor ]
    outer_p: torch.nn.Parameter  # [ torch.Tensor ]
    for param, outer_p in zip(model.parameters(), outer_parameters_list):
        param_data: torch.Tensor | DTensor = param.data
        outer_p_data: torch.Tensor = outer_p.data
        if isinstance(param_data, DTensor):
            param_data = param_data.to_local()
        outer_p.grad = outer_p_data - param_data.to('cpu')
        outer_grads.append(outer_p.grad)

    with train_profiler.session("all_reduce_multiple_with_retry"):
        start_time = time.time()

        all_reduce_success = pccl_utils.all_reduce_multiple_with_retry(
            communicator,
            outer_grads,
            ReduceOp.AVG
        )

        end_time = time.time()
        logger.info(f"All-Reduce took {end_time - start_time} seconds")
        if not all_reduce_success:
            logger.info("All peers left except me... continuing alone.")

    outer_optimizer.step()
    outer_optimizer.zero_grad()

    sync_inner_with_outer_state(model, outer_parameters_list)


def run_outer_step(
        model: torch.nn.Module,
        last_pseudo_grads: List[torch.Tensor],
        outer_parameters_list: List[torch.nn.Parameter],
        outer_optimizer: torch.optim.Optimizer,
        shared_state: SharedState,

        all_reduce_thread: threading.Thread,

        communicator: Communicator,
        train_profiler: Profiler,
        logger: Logger,

        training_progress: TrainingProgress,
        outer_lr_scheduler_config: LearningRateSchedulerConfig,

        topology_updated: bool,
        iter_num: int,
        num_syncs: IntRef,
        delayed_update: bool
) -> Optional[threading.Thread]:

    current_lr = compute_current_lr(training_progress.outer_step, outer_lr_scheduler_config)
    optim_utils.set_optimizer_lr(outer_optimizer, current_lr)

    if delayed_update:
        return run_async_outer_step(model, last_pseudo_grads, outer_parameters_list, outer_optimizer, shared_state,
                                    all_reduce_thread, communicator, train_profiler, logger,
                                    topology_updated, iter_num,
                                    num_syncs)
    else:
        run_sync_outer_step(model, outer_parameters_list, outer_optimizer, communicator, train_profiler, logger)
        return None


def sync_inner_with_outer_state(model: torch.nn.Module, outer_parameters_list: List[torch.nn.Parameter]):
    with torch.no_grad():
        inner_param: torch.nn.Parameter  # [ torch.Tensor | DTensor ]
        outer_param: torch.nn.Parameter  # [ torch.Tensor ]
        for inner_param, outer_param in zip(model.parameters(), outer_parameters_list):
            param_tensor = inner_param.data
            if isinstance(param_tensor, DTensor):
                param_tensor = param_tensor.to_local()
            param_tensor.copy_(outer_param, non_blocking=True)


def run_shared_state_sync(
        shared_state: SharedState,
        communicator: Communicator,
        model: torch.nn.Module, outer_parameters_list: List[torch.nn.Parameter],

        num_syncs: IntRef,
        train_profiler: Profiler,
        late_joiner: bool,
):
    # 3) Sync shared state => ensures we have the same aggregator (outer) parameters
    with train_profiler.session("pccl::sync_shared_state"):
        sync_info = communicator.sync_shared_state(shared_state)
        shared_state.revision += 1
        print(f"sync_info tx_bytes: {sync_info.tx_bytes}, rx_bytes: {sync_info.rx_bytes}")
        num_syncs += 1
        if num_syncs > 1 and not late_joiner:
            assert sync_info.rx_bytes == 0, "Shared state drifted unexpectedly in peers!"

        # initialize inner state on first sync
        if num_syncs == 1:
            print("Initializing inner state...")
            sync_inner_with_outer_state(model, outer_parameters_list)


def make_shared_state(outer_parameters: Dict[str, torch.nn.Parameter],
                      outer_optimizer: Optimizer,
                      iter_num: torch.Tensor):
    # Build the shared state that includes:
    #   - The outer parameters
    #   - The outer optimizer state (e.g. momentum buffers)
    shared_state_dict = {}

    # Reference outer parameters and parameter-specific optimizer state
    name: str
    outer_p: torch.nn.Parameter  # [ torch.Tensor | DTensor ]
    for name, outer_p in outer_parameters.items():
        # add outer parameter parameter as shared state
        shared_state_dict[name] = outer_p

        # add parameter-specific optimizer state
        state = outer_optimizer.state[outer_p]
        optim_utils.add_optimizer_state(shared_state_dict, name, state, type(outer_optimizer))

    # Also make iter_num synchronized shared state
    shared_state_dict['iter_num'] = iter_num

    entries = [
        TensorInfo.from_torch(
            param.data.to_local() if isinstance(param.data, DTensor) else param.data,
            name,
            allow_content_inequality=False
        )
        for name, param in shared_state_dict.items()
    ]
    shared_state = SharedState(entries)
    shared_state.revision = 0
    return shared_state


def train(logger: Logger, config: Config, mpi_config: Optional[MPIConfig], device: torch.device):
    grad_accum_steps = calc_gradient_accumulation_steps(
        config.train.batch_size, config.hardware.micro_batch_size, mpi_config
    )

    setup_profiler = Profiler()

    # Load tokenizer
    tokenizer_info = get_tokenizer_info(config)

    train_dataloader = make_dataloader(
        tokenizer_info=tokenizer_info,
        mpi_world_size=mpi_config.mpi_world_size if mpi_config is not None else 1,
        mpi_rank=mpi_config.mpi_rank if mpi_config is not None else 1,
        batch_size=config.hardware.micro_batch_size,
        data_config=config.data,
    )
    train_dataloader_iterator = iter(train_dataloader)

    with setup_profiler.session("::make_model"):
        with torch_utils.default_device('cuda'):
            model, model_config = make_model(
                config,
                vocab_size=tokenizer_info.vocab_size,
            )
        num_param_scalars = model.count_parameters()
        logger.info(f"Number of parameters: {num_param_scalars}")

    if config.hardware.act_ckpt:
        with setup_profiler.session("act_checkpointing::enable_activation_checkpointing"):
            num = 1 if isinstance(config.hardware.act_ckpt, bool) else config.hardware.act_ckpt
            act_checkpointing.enable_activation_checkpointing(model, num)

    with setup_profiler.session("sharding_utils::apply_sharding"):
        if mpi_config is not None:
            sharding_utils.apply_sharding(config.hardware, model)
        else:
            logger.info("MPI config not set, skipping application of model sharding...")

    # Setup optimizers
    with setup_profiler.session("optim_utils::make_optimizer"):
        inner_optimizer = optim_utils.make_optimizer(list(model.parameters()), config.train.optimizer)

    # -------------------------------------------------------------------------
    # ! PCCL-related state for outer optimizer, pseudo-gradients and shared state
    # All None if diloco is disabled
    outer_optimizer: Optional[torch.optim.Optimizer] = None
    outer_parameters: Dict[str, torch.nn.Parameter] = dict()
    outer_parameters_list: List[torch.nn.Parameter] = []
    shared_state: Optional[SharedState] = None
    # -------------------------------------------------------------------------

    training_progress = TrainingProgress(total_tokens=0, outer_step=0, step=0)

    checkpoint_info: Optional[CheckpointInfo] = None
    if config.ckpt.resume is not None:
        with setup_profiler.session("::load_checkpoint_fsdp_state"):
            # all is inplace
            checkpoint_info = load_checkpoint(
                model=model,
                optimizers=[inner_optimizer],
                training_progress=training_progress,
                dataloader=train_dataloader,
                path_root=config.ckpt.path,
                mpi_config=mpi_config
            )

    iter_num = 0
    if checkpoint_info is not None:
        iter_num = checkpoint_info.num_performed_outer_steps

    # -------------------------------------------------------------------------
    # ! Critical PCCL-related training-loop-state tracking variables !
    iter_num = torch.tensor([iter_num], dtype=torch.int64, device='cpu')
    # -------------------------------------------------------------------------

    if config.diloco:
        for name, local_p in model.named_parameters():
            if isinstance(local_p, DTensor):
                local_p = local_p.to_local()
            outer_p = outer_parameters[name] = torch.nn.Parameter(local_p.detach().cpu())
            outer_parameters_list.append(outer_p)

        with setup_profiler.session("optim_utils::make_optimizer[diloco]"):
            outer_optimizer = optim_utils.make_optimizer(outer_parameters_list, config.train.outer_optimizer)

            # do a dummy step to initialize outer optimizer state
            for op in outer_parameters_list:
                op.grad = torch.zeros_like(op)
            outer_optimizer.step()

        shared_state = make_shared_state(outer_parameters, outer_optimizer,
                                         iter_num)

    if (mpi_config is None or mpi_config.mpi_rank == 0) and config.wandb:
        wandb.init(
            project=config.project,
            config={"config": config.model_dump(),
                    "mpi_config": asdict(mpi_config) if mpi_config is not None else None},
        )

    with setup_profiler.session("torch::compile"):
        if config.hardware.torch_compile:
            model = torch.compile(model) if not TYPE_CHECKING else model

    memory_profiler: Optional[MemoryProfiler] = None
    if config.hardware.memory_profiler is not None:
        memory_profiler = MemoryProfiler(config.hardware.memory_profiler.freq,
                                         config.hardware.memory_profiler.snapshot_dir,
                                         mpi_config)

    num_inner_steps = config.diloco.inner_steps if config.diloco is not None else 1

    # initialize PCCL
    communicator = Communicator(config.pccl.ccoip_host, mpi_config.mpi_rank if mpi_config is not None else 0)
    communicator.connect(n_attempts=15)
    print("Connected to master via PCCL")

    if PRIME_SETUP_PROFILER_PRINT_TIMINGS:
        setup_profiler.print_report()

    train_profiler_collection = ProfilerCollection()

    # -------------------------------------------------------------------------
    # ! Critical PCCL-related training-loop-state tracking variables !
    local_iter_num = 0
    num_syncs = IntRef(0)
    local_world_size: int = communicator.get_attribute(Attribute.LOCAL_WORLD_SIZE)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # ! Critical PCCL-related state for async DiLoCo
    # None / empty if async DiLoCo is not used
    all_reduce_thread: Optional[threading.Thread] = None
    last_pseudo_grads: List[torch.Tensor] = []
    # -------------------------------------------------------------------------

    timing_events = []
    while True:
        local_iter_num += 1

        train_profiler = Profiler()

        if num_inner_steps > 1:
            # if we don't use diloco we don't print the outer step logs
            logger.info(f"outer_step step: {training_progress.outer_step}")

        global_world_size = communicator.get_attribute(Attribute.GLOBAL_WORLD_SIZE)

        topology_updated = False
        if local_iter_num == 1:
            # Assume the topology was updated in the first iteration because we just joined and got accepted
            topology_updated = True

        # Possibly update topology / wait for enough peers
        mpi_ranks_pending = False
        with train_profiler.session("pccl::update_topology"):
            if mpi_config is not None:
                mpi_ranks_pending = global_world_size < mpi_config.mpi_world_size

            if local_iter_num > 1 or mpi_ranks_pending or local_world_size == 1:
                while True:
                    try:
                        if communicator.are_peers_pending():
                            logger.info(
                                "Join-Candidate peers pending; awaiting concurrent collective operations to accept new peers...")
                            if all_reduce_thread is not None:
                                all_reduce_thread.join()
                            communicator.update_topology()
                            topology_updated = True
                        break
                    except PCCLError as e:
                        logger.info(f"Updating PCCL topology failed {e}, retrying...")
                        time.sleep(1)

        if mpi_ranks_pending:
            print("Not all MPI ranks have joined...")
            time.sleep(1)
            continue

        # TODO: Make minimum num pccl peers configurable
        local_world_size = communicator.get_attribute(Attribute.LOCAL_WORLD_SIZE)
        if local_world_size < 2:
            print("Waiting for more workers to join...")
            time.sleep(1)
            continue

        if topology_updated:
            run_shared_state_sync(shared_state, communicator, model, outer_parameters_list, num_syncs, train_profiler,
                                  False)

        run_inner_steps(
            model, train_dataloader_iterator, inner_optimizer, device,

            logger, memory_profiler,

            local_world_size,

            mpi_config, train_profiler, config,
            training_progress, grad_accum_steps, timing_events
        )

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
            with train_profiler.session("outer_step"):
                all_reduce_thread = run_outer_step(model, last_pseudo_grads, outer_parameters_list,
                                                   outer_optimizer, shared_state, all_reduce_thread, communicator,
                                                   train_profiler, logger,
                                                   training_progress, config.train.outer_lr_scheduler,
                                                   topology_updated, iter_num, num_syncs,
                                                   config.diloco.delayed_update)

        iter_num += 1
        training_progress.outer_step = iter_num.item()

        if (
                config.ckpt.interval is not None
                and training_progress.step > 0
                and training_progress.step % config.ckpt.interval == 0
        ):
            # we only allow to checkpoint after a outer step. For non diloco training outer step = 1 anyway
            save_checkpoint(
                model=model,
                optimizers=[inner_optimizer],
                training_progress=training_progress,
                dataloader=train_dataloader,
                path_root=config.ckpt.path,
                checkpoint_info=CheckpointInfo(
                    num_performed_outer_steps=training_progress.outer_step,
                    shared_state_revision=shared_state.revision if shared_state is not None else -1,
                ),
                mpi_config=mpi_config
            )

        if training_progress.step >= config.train.lr_scheduler.num_total_steps:
            # we only allow to break outside of the inner loop.
            # This avoid ending the training in the middle of a the inner loop
            # Since ckpt strategy and all reduce is done at the outer loop level.
            break

        if mpi_config is None or mpi_config.mpi_rank == 0:
            wandb.finish()

    if config.hardware.memory_profiler is not None:
        logger.debug(f"Max memory used: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")

    logger.info("Training finished, exiting ...")
    if mpi_config is not None:
        destroy_process_group()


def main():
    # Allow eager fallback during production so that the training runs don't die
    # However, in development, we want to know that we broke torch compile
    torch._dynamo.config.suppress_errors = "ZERO_BAND_DEV" not in os.environ  # type: ignore
    torch.set_float32_matmul_precision("high")

    mpi_config: Optional[MPIConfig] = ccl_utils.make_mpi_config(
        mpi_rank=os.getenv("RANK"),
        mpi_world_size=os.getenv("WORLD_SIZE")
    )  # may return None

    # Don't set torch seed; Random seed is necessary to prevent unexpected equivalence of rank state

    config = Config(**parse_argv())  # type: ignore
    logger = get_logger(config, mpi_config)

    gpu_ordinal = int(os.getenv("GPU_ORDINAL", os.getenv("RANK", "0")))

    num_total_gpus = torch.cuda.device_count()
    logger.info(f"Using gpu ordinal:{gpu_ordinal}/num_total:{num_total_gpus}")

    torch.cuda.set_device(gpu_ordinal)
    device = torch.device(f'cuda:{torch.cuda.current_device()}')
    logger.info(f"Using device: {torch.cuda.get_device_name(device)}")

    train(logger, config, mpi_config, device)


if __name__ == "__main__":
    main()
