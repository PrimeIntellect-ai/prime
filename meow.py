from typing import List
import os
import torch.distributed as dist
import torch
import time
import toposolve

rank = int(os.environ['RANK'])
def rprint(*args):
    print(f"[Rank {rank}] {' '.join(map(str, args))}\n", end="")

BENCH_TENSOR_SIZE = 1_000_000

class EDM:
    def __init__(self):
        master_addr = os.environ['MASTER_ADDR']
        master_port = int(os.environ['MASTER_PORT'])
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])

        self.rank = rank
        self.world_size = world_size

        rprint("Creating Store")
        self.global_store = dist.TCPStore(
            host_name=master_addr,
            port=master_port + 1,
            is_master=(rank==0),
            world_size=2,
        )

        rprint("Store created. Creating ProcessGroupGloo")
        self.global_pg = dist.distributed_c10d.ProcessGroupGloo(self.global_store, rank, world_size)
        rprint("ProcessGroupGloo created")

        self._measure_connectivity()
        if rank == 0:
            pings = self.get_pings()
            print(*pings, sep="\n")
            min_dist, path = toposolve.TSPSolver().solve_tsp(pings)
            print(f"Min distance: {min_dist}")
            print(f"Path: {path}")
    
    def get_pings(self) -> List[List[int]]:
        pings = [[1000_000_000] * self.world_size for _ in range(self.world_size)]
        for i in range(self.world_size):
            for j in range(self.world_size):
                if i == j:
                    continue
                pings[i][j] = int(self.global_store.get(f"ping_{i}_{j}"))
        return pings

    def _measure_connectivity(self):
        # Recv from all other peers
        recv_work = []
        tensor = torch.ones(BENCH_TENSOR_SIZE, dtype=torch.float32)
        for i in range(self.world_size):
            if i == self.rank:
                continue
            recv_work.append(self.global_pg.recv([tensor], i, self.rank + self.world_size * i))

        # Ping all other peers
        for i in range(self.world_size):
            if i == self.rank:
                continue
            time_taken = self._ping_peer(i)
            self.global_store.set(f"ping_{self.rank}_{i}", str(time_taken))
        
        # Wait for all recv operations to complete
        for work in recv_work:
            work.wait()

    def _ping_peer(self, peer_rank: int) -> int:
        """Ping a peer and return the time taken in microseconds"""
        tensor = torch.ones(BENCH_TENSOR_SIZE, dtype=torch.float32)
        start_time = time.perf_counter()
        self.global_pg.send([tensor], peer_rank, self.rank * self.world_size + peer_rank).wait()
        end_time = time.perf_counter()
        return int((end_time - start_time) * 1e6)

def main():
    edm = EDM()


if __name__ == "__main__":
    main()
