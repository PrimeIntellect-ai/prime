from typing import Optional

from pccl import Communicator

from zeroband.config import Config

communicator: Optional[Communicator] = None


def init_pccl(config: Config):
    if config.diloco is None:
        raise RuntimeError("PCCL was configured as CCL-library, but not [pccl] was not configured!")

    global communicator
    communicator = Communicator(config.pccl.ccoip_host, 0)
    communicator.connect(n_attempts=15)
    print("Connected to master via PCCL")
