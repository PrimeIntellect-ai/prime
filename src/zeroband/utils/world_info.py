import os

world_info = None


class LocalWorldInfo:
    """
    Local World information.
    The "local world" shall mean the world within the worker that is contributing as one peer to the training run.
    PCCL does not have concept of ranks and this information is strictly separate from PCCL related state.
    """

    world_size: int
    rank: int

    local_world_size: int
    local_rank: int

    num_nodes: int

    def __init__(self):
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        self.num_nodes = self.world_size // self.local_world_size

    def __repr__(self):
        return f"WorldInfo(world_size={self.world_size}, rank={self.rank}, local_rank={self.local_rank}, local_world_size={self.local_world_size}, num_nodes={self.num_nodes})"

    def json(self) -> dict[str, int | str]:
        return {
            "world_size": self.world_size,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "local_world_size": self.local_world_size,
            "num_nodes": self.num_nodes
        }


def get_local_world_info() -> LocalWorldInfo:
    """
    Return a WorldInfo singleton.
    """
    global world_info
    if world_info is None:
        world_info = LocalWorldInfo()
    return world_info