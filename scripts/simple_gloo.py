import os
import torch.distributed as dist

master_addr = os.environ["MASTER_ADDR"]
master_port = 12345
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

store = dist.TCPStore(master_addr, master_port, rank == 0, world_size)

dist.init_process_group("gloo", store=store, rank=rank, world_size=world_size)

if rank == 0:
    store.set("j", "k")
else:
    val = store.get("j").decode("utf-8")
    print(f"Rank {rank} got key: {val}")

dist.destroy_process_group()
