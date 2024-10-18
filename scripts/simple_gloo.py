import os
import torch.distributed as dist

master_addr = os.environ["MASTER_ADDR"]
master_port = 12345
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

print("Ho")
store = dist.TCPStore(host_name=master_addr, port=master_port, is_master=(rank == 0), world_size=2)

store.set("j", "k")
print("Hi")
pg = dist.distributed_c10d.ProcessGroupGloo(store, rank, world_size)
print("Hi 1")

del pg
