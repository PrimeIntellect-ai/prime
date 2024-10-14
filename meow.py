import hashlib
import os
import torch
import torch.distributed as dist
from zeroband.C.collectives import ring_allreduce
from zeroband.utils import get_tensor_signature

def tensor_bytes_hash(a: torch.Tensor) -> str:
    return hashlib.md5(str(a.untyped_storage()).encode("utf-8")).hexdigest()

master_addr = os.environ['MASTER_ADDR']
master_port = int(os.environ['MASTER_PORT'])
rank = int(os.environ['RANK'])
world_size = int(os.environ['WORLD_SIZE'])

print("Creating Store")
store = dist.TCPStore(
        host_name=master_addr,
        port=master_port + 1,
        is_master=(rank==0),
        world_size=2,
)

print("Store created. Creating ProcessGroupGloo")
pg = dist.distributed_c10d.ProcessGroupGloo(store, rank, world_size)
print("ProcessGroupGloo created")

#a = torch.randn(world_size * 3)
a = torch.randn(1000 - 1)
b = a.clone()

print(f"[Rank {rank}] {a[:10]}")

ring_allreduce(a, group=pg)
dist.all_reduce(b, group=pg)

print(f"[Rank {rank}] {a[:16]} {get_tensor_signature(a)} {tensor_bytes_hash(a)}")
#if rank == 2:
#    print(f"[Rank {rank}] {a[:16]} {get_tensor_signature(a)} {tensor_bytes_hash(a)}")
#    print(f"[Rank {rank}] {b[:16]} {get_tensor_signature(b)} {tensor_bytes_hash(b)}")

del pg