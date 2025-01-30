from pydantic_config import BaseConfig, parse_argv
import torch
from torch.distributed import destroy_process_group, init_process_group, ReduceOp
import torch.utils.benchmark as benchmark

from zeroband.collectives import Compression, all_reduce
from zeroband.utils.world_info import get_world_info
from zeroband.utils.logger import get_logger

from enum import Enum


class TorchDtype(str, Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    UINT8 = "uint8"


TORCH_DTYPE_MAP = {
    None: None,
    TorchDtype.FLOAT32: torch.float32,
    TorchDtype.FLOAT16: torch.float16,
    TorchDtype.BFLOAT16: torch.bfloat16,
    TorchDtype.UINT8: torch.uint8,
}


class Config(BaseConfig):
    size_model: int = int(1e7)
    n_iters: int = 4
    compression: Compression = Compression.NO


def main(config: Config):
    world_info = get_world_info()

    mat = torch.rand(1, config.size_model)

    logger.info(
        f"\n ======== Benchmark all reduce between {world_info.world_size} gpus over {world_info.nnodes} nodes =========\n"
    )

    t0 = benchmark.Timer(
        stmt="compressed_all_reduce(compression, mat, op=op)",
        globals={
            "compressed_all_reduce": all_reduce,
            "mat": mat,
            "compression": config.compression,
            "op": ReduceOp.SUM,
        },
    )

    measured_time = t0.timeit(config.n_iters).mean

    bandwidth = config.size_model * 4 / 1e6 / measured_time

    logger.info(f"Average time per iteration: {measured_time:.2f} seconds, Average bandwidth: {bandwidth:.4f} MB/s")


if __name__ == "__main__":
    config = Config(**parse_argv())

    torch.set_float32_matmul_precision("high")
    init_process_group(backend="gloo")

    logger = get_logger()
    main(config)
    destroy_process_group()
