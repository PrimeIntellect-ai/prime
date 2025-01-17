from typing import Callable, Optional, TypeAlias
import torch
import torch.distributed as dist

from zeroband.config import Compression

AllReduceFunc: TypeAlias = Callable[
    [torch.Tensor, dist.ReduceOp, Optional[dist.ProcessGroup], Optional[torch.dtype]], None
]


def gloo_all_reduce(
    tensor: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM, # type: ignore (defined weird)
    group: Optional[dist.ProcessGroup] = None,
) -> None:
    """Wrap gloo all reduce"""
    if group is None:
        group = dist.distributed_c10d._get_default_group()
    if op not in [dist.ReduceOp.SUM, dist.ReduceOp.AVG]:
        raise ValueError(f"Unsupported reduce operation {op}. Only SUM and AVG are supported.")

    # group = cast(dist.ProcessGroup, group) # just type hint stuff for IDE
    if op == dist.ReduceOp.AVG:
        # todo check numerical stability of doing post or pre div
        tensor.div_(group.size())

    dist.all_reduce(tensor, op, group=group)


def all_reduce(
    compression: Compression,
    tensor: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM, # type: ignore
    group: Optional[dist.ProcessGroup] = None,
) -> None:
    if compression == Compression.UINT8:
        from zeroband.C.collectives import ring_allreduce as ring_allreduce_c

        return ring_allreduce_c(tensor, op, group)
    else:
        return gloo_all_reduce(tensor, op, group)


# ===============
# Code purgatory
# ---------------
# This code is still here because it is used by tests
# ring_allreduce is used by tests/test_c/test_collectives.py to make sure the new c impl doesnt deviate too much numerically
BUFFER_COUNT = 2


def ring_allreduce_py(
    tensor: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM, # type: ignore
    group: Optional[dist.ProcessGroup] = None,
    transfer_dtype: Optional[torch.dtype] = None,
    quantization_func: Optional[Callable] = None,
) -> None:
    """
    Perform all-reduce on a tensor using ring algorithm.
    The accumulation will be done in-place on the input tensor.
    The transfers will be done using the specified transfer_dtype.
    """
    if quantization_func is not None:
        if transfer_dtype is not None:
            raise ValueError("Quantization and transfer_dtype cannot be used together")
        transfer_dtype = tensor.dtype
    if transfer_dtype is None:
        transfer_dtype = tensor.dtype
    if group is None:
        group = dist.distributed_c10d._get_default_group()
    if op not in [dist.ReduceOp.SUM, dist.ReduceOp.AVG]:
        raise ValueError(f"Unsupported reduce operation {op}. Only SUM and AVG are supported.")

    world_size = group.size()
    rank = group.rank()

    # Divide the tensor into chunks
    flat_tensor = tensor.as_strided((tensor.numel(),), (1,))
    chunks = flat_tensor.chunk(world_size * BUFFER_COUNT)

    assert flat_tensor.size(0) % (world_size * BUFFER_COUNT) == 0, "Tensor size must be divisible by world size"

    # Temporary buffers for transferring data
    num_buffers = BUFFER_COUNT * world_size
    if quantization_func is not None:
        recv_buffer = [torch.empty_like(chunks[0], dtype=torch.uint8) for _ in range(BUFFER_COUNT)]
        send_buffer = [None for _ in range(BUFFER_COUNT)]
        send_lookup_buffer = [None for _ in range(BUFFER_COUNT)]
        recv_lookup_buffer = [torch.empty(256, dtype=chunks[0].dtype) for _ in range(BUFFER_COUNT)]
        send_lookup_work = [None for _ in range(BUFFER_COUNT)]
        recv_lookup_work = [None for _ in range(BUFFER_COUNT)]
    else:
        recv_buffer = [torch.empty_like(chunks[0], dtype=transfer_dtype) for _ in range(BUFFER_COUNT)]
        send_buffer = [torch.empty_like(chunks[0], dtype=transfer_dtype) for _ in range(BUFFER_COUNT)]
    send_work = [None] * BUFFER_COUNT
    recv_work = [None] * BUFFER_COUNT

    send_rank = (rank + 1) % world_size
    recv_rank = (rank - 1) % world_size
    for step in range(1, world_size * BUFFER_COUNT + 1):
        send_chunk = (rank * BUFFER_COUNT - step) % num_buffers

        if send_work[step % BUFFER_COUNT] is not None:
            send_work[step % BUFFER_COUNT].wait()
            recv_work[step % BUFFER_COUNT].wait()
            if quantization_func is not None:
                send_lookup_work[step % BUFFER_COUNT].wait()
                recv_lookup_work[step % BUFFER_COUNT].wait()
                # print(recv_lookup_buffer[step % BUFFER_COUNT][recv_buffer[step % BUFFER_COUNT].long()])
                chunks[send_chunk].add_(
                    recv_lookup_buffer[step % BUFFER_COUNT][recv_buffer[step % BUFFER_COUNT].long()]
                )
            else:
                chunks[send_chunk].add_(recv_buffer[step % BUFFER_COUNT])

        if step <= (world_size - 1) * BUFFER_COUNT:
            # Send and receive
            if quantization_func is not None:
                send_buffer[step % BUFFER_COUNT], send_lookup_buffer[step % BUFFER_COUNT] = quantization_func(
                    chunks[send_chunk]
                )
                send_lookup_work[step % BUFFER_COUNT] = dist.isend(
                    send_lookup_buffer[step % BUFFER_COUNT], dst=send_rank, group=group, tag=step + 1000
                )
                recv_lookup_work[step % BUFFER_COUNT] = dist.irecv(
                    recv_lookup_buffer[step % BUFFER_COUNT], src=recv_rank, group=group, tag=step + 1000
                )
            else:
                send_buffer[step % BUFFER_COUNT].copy_(chunks[send_chunk])
            send_work[step % BUFFER_COUNT] = dist.isend(
                send_buffer[step % BUFFER_COUNT], dst=send_rank, group=group, tag=step
            )
            recv_work[step % BUFFER_COUNT] = dist.irecv(
                recv_buffer[step % BUFFER_COUNT], src=recv_rank, group=group, tag=step
            )

    if op == dist.ReduceOp.AVG:
        for i in range(BUFFER_COUNT):
            chunks[i + rank * BUFFER_COUNT].divide_(world_size)
    if quantization_func is not None:
        for i in range(BUFFER_COUNT):
            quant_weight, lookup = quantization_func(chunks[i + rank * BUFFER_COUNT])
            chunks[i + rank * BUFFER_COUNT].copy_(lookup[quant_weight.long()])

    if quantization_func is not None:
        recv_buffer = [torch.empty_like(chunks[0], dtype=torch.uint8) for _ in range(BUFFER_COUNT)]
        send_buffer = [None for _ in range(BUFFER_COUNT)]
        send_lookup_buffer = [None for _ in range(BUFFER_COUNT)]
        recv_lookup_buffer = [torch.empty(256, dtype=chunks[0].dtype) for _ in range(BUFFER_COUNT)]
        send_lookup_work = [None for _ in range(BUFFER_COUNT)]
        recv_lookup_work = [None for _ in range(BUFFER_COUNT)]
    send_work = [None] * BUFFER_COUNT
    recv_work = [None] * BUFFER_COUNT

    for step in range(1, world_size * BUFFER_COUNT + 1):
        send_chunk = (rank * BUFFER_COUNT + BUFFER_COUNT - step) % num_buffers

        if send_work[step % BUFFER_COUNT] is not None:
            send_work[step % BUFFER_COUNT].wait()
            recv_work[step % BUFFER_COUNT].wait()
            if quantization_func is not None:
                send_lookup_work[step % BUFFER_COUNT].wait()
                recv_lookup_work[step % BUFFER_COUNT].wait()
                chunks[send_chunk].copy_(
                    recv_lookup_buffer[step % BUFFER_COUNT][recv_buffer[step % BUFFER_COUNT].long()]
                )
            else:
                chunks[send_chunk].copy_(recv_buffer[step % BUFFER_COUNT])

        if step <= (world_size - 1) * BUFFER_COUNT:
            # Send and receive
            if quantization_func is not None:
                send_buffer[step % BUFFER_COUNT], send_lookup_buffer[step % BUFFER_COUNT] = quantization_func(
                    chunks[send_chunk]
                )
                send_lookup_work[step % BUFFER_COUNT] = dist.isend(
                    send_lookup_buffer[step % BUFFER_COUNT], dst=send_rank, group=group, tag=step + 1000
                )
                recv_lookup_work[step % BUFFER_COUNT] = dist.irecv(
                    recv_lookup_buffer[step % BUFFER_COUNT], src=recv_rank, group=group, tag=step + 1000
                )
            else:
                send_buffer[step % BUFFER_COUNT].copy_(chunks[send_chunk])

            send_work[step % BUFFER_COUNT] = dist.isend(
                send_buffer[step % BUFFER_COUNT], dst=send_rank, group=group, tag=step
            )
            recv_work[step % BUFFER_COUNT] = dist.irecv(
                recv_buffer[step % BUFFER_COUNT], src=recv_rank, group=group, tag=step
            )
