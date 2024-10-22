import torch
import psutil
import time


def get_total_ram():
    return psutil.virtual_memory().total


def get_available_ram():
    return psutil.virtual_memory().available


def allocate_memory(size_bytes):
    # Allocate tensor on CPU
    return torch.ones((size_bytes) // 4, dtype=torch.float32)


def main():
    print("Starting memory allocation test...")

    total_ram = get_total_ram()
    print(f"Total physical RAM: {total_ram / (1024**3):.2f} GB")

    # Start with 1% of total RAM
    initial_percentage = 50
    percentage_increment = 10
    current_percentage = initial_percentage

    tensors = []

    while True:
        try:
            available_ram = get_available_ram()
            allocation_size = int(total_ram * (current_percentage / 100))

            # Allocate memory
            tensor = allocate_memory(allocation_size)
            tensors.append(tensor)

            # Get current memory usage
            process = psutil.Process()
            memory_info = process.memory_info()

            print(
                f"Allocated {allocation_size / (1024**2):.2f}MB ({current_percentage}% of total RAM). "
                f"Process memory used: {memory_info.rss / (1024**3):.2f}GB. "
                f"Available RAM: {available_ram / (1024**3):.2f}GB"
            )

            # Increase percentage for next iteration
            current_percentage += percentage_increment

            # Sleep to allow for monitoring
            time.sleep(1)

        except RuntimeError as e:
            print(f"Memory allocation failed: {e}")
            break
        except KeyboardInterrupt:
            print("Test stopped by user.")
            break

    print("Test completed. Check your system monitor to see if swap was used.")


if __name__ == "__main__":
    main()
