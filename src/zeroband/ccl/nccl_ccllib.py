import torch.distributed as dist


def init_nccl(mpi_rank: int, mpi_world_size: int):
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:23456",
        rank=mpi_rank,
        world_size=mpi_world_size
    )

    if dist.is_available() and dist.is_initialized():
        raise RuntimeError("torch.distributed is not available or was not initialized!")

    if dist.is_nccl_available():
        raise RuntimeError("torch.distributed reports NCCL is not available!")
