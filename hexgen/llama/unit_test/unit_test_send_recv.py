import torch
import torch.distributed as dist

def main():
    # Initialize the process group
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    tensor_size = 100000000  # Example tensor size

    if rank == 0 or rank == 1:
        # Rank 0 sends a tensor to Rank 1 using non-blocking send
        tensor = torch.randn(tensor_size).cuda()
        op = dist.P2POp(dist.isend if rank == 0 else dist.irecv, tensor, 1 if rank == 0 else 0)
        reqs = dist.batch_isend_irecv([op])
        print(1)
        for req in reqs:
            req.wait()
        print(2)
   
    dist.barrier()

    if rank == 0:
        # Now safely initiate the blocking send to rank 2
        dist.send(tensor, 2)

    elif rank == 1:
        # Rank 1 now sends the received tensor to Rank 3 using blocking send
        dist.send(tensor, 3)

    elif rank == 2:
        # Rank 2 receives a tensor from Rank 0 using blocking recv
        tensor = torch.zeros(tensor_size).cuda()
        dist.recv(tensor, 0)
        # send_op = dist.P2POp(dist.isend, tensor, 3)
        # reqs = dist.batch_isend_irecv([send_op])
        # for req in reqs:
        #     req.wait()

    elif rank == 3:
        # Rank 3 receives a tensor from Rank 1 using blocking recv
        tensor = torch.zeros(tensor_size).cuda()
        dist.recv(tensor, 1)
        # recv_op = dist.P2POp(dist.irecv, tensor, 2)
        # reqs = dist.batch_isend_irecv([recv_op])
        # for req in reqs:
        #     req.wait()

if __name__ == "__main__":
    main()

'''
def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    tensor_size = 100000000  # Example tensor size

    if rank == 0 or rank == 1:
        # Non-blocking communication between ranks 0 and 1
        tensor = torch.randn(tensor_size).cuda() if rank == 0 else torch.zeros(tensor_size).cuda()
        ops = [dist.P2POp(dist.isend if rank == 0 else dist.irecv, tensor, 1 if rank == 0 else 0)]
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()  # Ensure completion of non-blocking operations
        print(f'Rank {rank} completed batch_isend_irecv.')

    # Synchronization point to ensure the above non-blocking operations complete
    dist.barrier()

    # Subsequent communication: 0 -> 2 and 1 -> 3
    if rank == 0:
        # Blocking send to rank 2
        dist.send(tensor, 2)
    elif rank == 1:
        # Blocking send to rank 3
        dist.send(tensor, 3)
    elif rank == 2:
        # Blocking receive from rank 0
        tensor = torch.zeros(tensor_size).cuda()
        dist.recv(tensor, 0)
    elif rank == 3:
        # Blocking receive from rank 1
        tensor = torch.zeros(tensor_size).cuda()
        dist.recv(tensor, 1)

    print(f'Rank {rank} completed all operations.')

if __name__ == "__main__":
    main()
'''
