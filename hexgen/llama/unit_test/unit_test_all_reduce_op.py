import torch
import torch.distributed as dist

def main():
    # Initialize the process group
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    # Assuming a world size of 4 for this example
    # Create a tensor specific to this rank
    tensor = torch.tensor([rank], dtype=torch.float32).cuda()

    # Create groups
    group_0 = dist.new_group([0, 1])
    group_1 = dist.new_group([2, 3])
    group = [group_0, group_1]
    # Perform the all-reduce operation within the defined group
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group[rank//2])
    print(f"Rank {rank} after all_reduce: {tensor.item()}")

if __name__ == "__main__":
    main()
