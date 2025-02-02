import os
import pickle
import torch
from zeroband.utils.logger import get_logger
from zeroband.utils.world_info import get_world_info

_MAX_ENTRIES = 10000


class MemoryProfiler:
    """Pytorch Memory Profiler.
    The output are pickles file that can be visualized here: https://pytorch.org/memory_viz
    """

    def __init__(self, freq: int, snapshot_dir: str):
        torch.cuda.memory._record_memory_history(max_entries=_MAX_ENTRIES)
        self.freq = freq

        self.world_info = get_world_info()
        self.logger = get_logger()
        self.step_num = 0

        os.makedirs(snapshot_dir, exist_ok=True)
        self.snapshot_dir = snapshot_dir

    def log_memory_summary(self, curr_snapshot_dir):
        """Log memory summary and memory allocated"""
        summary = torch.cuda.memory_summary(device=None, abbreviated=False)
        allocated_memory = torch.cuda.memory_allocated()

        # Save the memory summary to a file
        with open(f"{curr_snapshot_dir}/rank{self.world_info.rank}_memory_summary.txt", "w") as summary_file:
            summary_file.write(summary)

        # Save the allocated memory as a text log
        with open(f"{curr_snapshot_dir}/rank{self.world_info.rank}_memory_allocated.txt", "w") as alloc_file:
            alloc_file.write(f"Allocated memory: {allocated_memory / 1024 ** 2:.2f} MB\n")

        # log this information using the logger
        self.logger.info(f"Memory summary and allocation saved for rank {self.world_info.rank} at step {self.step_num}")

    def step(self):
        self.step_num += 1
        if self.step_num % self.freq != 0:
            return

        dir_name = f"iteration_{self.step_num}"

        curr_snapshot_dir = os.path.join(self.snapshot_dir, dir_name)
        if not os.path.exists(curr_snapshot_dir):
            os.makedirs(curr_snapshot_dir, exist_ok=True)

        # Save memory snapshot
        with open(f"{curr_snapshot_dir}/rank{self.world_info.rank}_memory_snapshot.pickle", "wb") as output:
            pickle.dump(torch.cuda.memory._snapshot(), output)

        # Log memory summary and allocated memory
        self.log_memory_summary(curr_snapshot_dir)

        torch.distributed.barrier()
