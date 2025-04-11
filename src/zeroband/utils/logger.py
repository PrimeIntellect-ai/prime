import logging

from zeroband.ccl.ccl_utils import MPIConfig
from zeroband.config import Config

logger = None


"""
ZERO_BAND_LOG_LEVEL=DEBUG allow to control the log level for all ranks
ZERO_BAND_LOG_ALL_RANK=true allow to control if all ranks should log or only the local rank 0
"""


class CustomFormatter(logging.Formatter):
    def __init__(self, local_rank: int):
        super().__init__()
        self.local_rank = local_rank

    def format(self, record):
        log_format = "{asctime} [{levelname}] [Rank {local_rank}] {message}"
        formatter = logging.Formatter(log_format, style="{", datefmt="%H:%M:%S")
        record.local_rank = self.local_rank  # Add this line to set the local rank in the record
        return formatter.format(record)


def get_logger(config: Config | None = None, mpi_config: MPIConfig | None = None, name: str | None = None) -> logging.Logger:
    global logger  # Add this line to modify the global logger variable
    if logger is not None:
        return logger

    logger = logging.getLogger(name or __name__)

    mpi_rank = mpi_config.mpi_rank if mpi_config is not None else 0
    log_level = config.log_level if config else "DEBUG"
    if mpi_rank == 0:
        logger.setLevel(level=getattr(logging, log_level, logging.INFO))
    else:
        if (not config) or config.log_all_ranks:
            logger.setLevel(level=getattr(logging, log_level, logging.INFO))
        else:
            logger.setLevel(level=logging.CRITICAL)  # Disable logging for non-zero ranks

    handler = logging.StreamHandler()
    handler.setFormatter(CustomFormatter(mpi_rank))
    logger.addHandler(handler)
    logger.propagate = False  # Prevent the log messages from being propagated to the root logger

    return logger
