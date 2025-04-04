from dataclasses import dataclass
from typing import Optional

from mpmath.libmp import mpi_mid

from zeroband.ccl.nccl_ccllib import init_nccl
from zeroband.ccl.pccl_ccllib import init_pccl
from zeroband.config import Config


@dataclass
class MPIConfig:
    """
    We differentiate MPI world size & rank from
    non-MPI world size, which relates to PCCL.

    MPI world size concerns NCCL.
    World size concerns PCCL.
    """
    mpi_rank: int
    mpi_world_size: int


def init_ccl_library(config: Config, mpi_config: Optional[MPIConfig]):
    ccl_library = config.hardware.ccl_library
    if ccl_library == "nccl":
        if mpi_config is None:
            raise RuntimeError("Cannot initialize with ccl library 'nccl' when mpi config is not supplied!")
        init_nccl(mpi_config.mpi_rank, mpi_config.mpi_world_size)
    elif ccl_library == "pccl":
        init_pccl(config)


def make_mpi_config(mpi_rank: Optional[str], mpi_world_size: Optional[str]) -> Optional[MPIConfig]:
    """
    Creates an MPI configuration from the supplied env-var strings
    :param mpi_rank the optionally present env-var string for the mpi rank
    :param mpi_world_size the optionally present env-var string for the mpi world size
    """
    assert (mpi_rank is not None and mpi_world_size is not None) or (
            mpi_rank is None and mpi_world_size is None), "MPI rank and MPI world size must either both be None, or both set"

    if mpi_rank is not None and mpi_world_size is not None:
        mpi_rank = int(mpi_rank) if mpi_rank.isdigit() else None
        mpi_world_size = int(mpi_world_size) if mpi_world_size.isdigit() else None
        return MPIConfig(
            mpi_rank=mpi_rank,
            mpi_world_size=mpi_world_size
        )
    else:
        return None
