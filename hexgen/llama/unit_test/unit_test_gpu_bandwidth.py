import torch
import torch.distributed as dist
import time

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Set the device based on the rank of the process
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    # Size of the tensor varies to measure bandwidth at different data sizes
    sizes = [256 * (1024 ** 2) * (2 ** i) for i in range(10)]  # Sizes from 256MB to larger
    num_rounds = 10  # Number of times to send the data to average the results
    
    # Only proceed if there are exactly 2 GPUs involved in this measurement
    if world_size != 2:
        if rank == 0:
            print("This script requires exactly 2 processes (GPUs) to run.")
        return

    # Loop over different tensor sizes to measure bandwidth
    for size in sizes:
        data = torch.randn(size, device=device)  # Generate random data
        dist.barrier()  # Synchronize all processes before timing
        start_time = time.time()

        if rank == 0:
            for _ in range(num_rounds):
                dist.send(tensor=data, dst=1)
        elif rank == 1:
            for _ in range(num_rounds):
                dist.recv(tensor=data, src=0)

        dist.barrier()  # Ensure completion of all communications before stopping the timer
        end_time = time.time()

        # Only rank 0 prints results
        if rank == 0:
            time_per_round = (end_time - start_time) / num_rounds
            num_bytes = data.numel() * data.element_size()
            bandwidth = (num_bytes / time_per_round) / 1e9  # Convert bytes/s to GB/s
            print(f"Data size: {num_bytes/1e6:.2f} MB, Bandwidth: {bandwidth:.2f} GB/s")

if __name__ == "__main__":
    main()
