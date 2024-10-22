import sys
import time
import torch


def allocate_memory(size_bytes):
    # Allocate tensor on CPU
    return torch.ones((size_bytes) // 4, dtype=torch.float32)


if __name__ == "__main__":
    size_gb = float(sys.argv[1])
    size_bytes = int(size_gb * 1024 * 1024 * 1024)

    data = allocate_memory(size_bytes)
    print(f"Allocated {size_gb} GB of RAM using NumPy")
    while True:
        time.sleep(1)
        print(f"Allocated {size_gb} GB of RAM using NumPy, data.shape: {data.shape}")
