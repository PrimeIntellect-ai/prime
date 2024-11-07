import os
import torch.distributed as dist
import torch
import time

rank = int(os.environ['RANK'])
def rprint(*args):
    print(f"[Rank {rank}] {' '.join(map(str, args))}\n", end="")

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

        self.measure_connectivity()

    def measure_connectivity(self):
        recv_work = []
        for i in range(self.world_size):
            if i == self.rank:
                continue
            tensor = torch.ones(1_000_000, dtype=torch.float32)
            rprint(f"Recv from peer {i} with tag {self.rank + self.world_size * i}")
            recv_work.append(self.global_pg.recv([tensor], i, self.rank + self.world_size * i))

        self.global_pg.barrier().wait()
        for i in range(self.world_size):
            if i == self.rank:
                continue
            rprint(f"Pinging peer {i}")
            time_taken = self.ping_peer(i)
            rprint(f"Ping to peer {i} took {time_taken} seconds")
        
        for work in recv_work:
            work.wait()

    def ping_peer(self, peer_rank: int) -> float:
        tensor = torch.ones(1_000_000, dtype=torch.float32)
        start_time = time.perf_counter()
        rprint(f"Send from peer {self.rank} to {peer_rank} with tag {self.rank * self.world_size + peer_rank}")
        self.global_pg.send([tensor], peer_rank, self.rank * self.world_size + peer_rank).wait()
        end_time = time.perf_counter()
        return end_time - start_time

def main():
    edm = EDM()


if __name__ == "__main__":
    main()
