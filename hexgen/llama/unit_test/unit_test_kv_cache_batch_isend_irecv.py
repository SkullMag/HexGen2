import torch
import torch.distributed as dist

def kv_cache_send(structure, last_token_ids, dst):
    send_ops = []
    
    # Prepare send operations for the structured data
    for sub_tuple in structure:
        for tensor in sub_tuple:
            op = dist.P2POp(dist.isend, tensor, dst)
            send_ops.append(op)
    
    # Prepare send operation for last_token_ids
    send_ops.append(dist.P2POp(dist.isend, last_token_ids, dst))

    # Perform batched send operations
    reqs = dist.batch_isend_irecv(send_ops)
    
    # Wait for all operations to complete
    for req in reqs:
        req.wait()

def kv_cache_recv(src, main_tuple_length=32, sub_tuple_length=2, tensor_shape=[4, 32, 1001, 80], last_token_ids_shape=[4, 1]):
    recv_ops = []
    structure = []

    # Prepare to receive the structured data
    for _ in range(main_tuple_length):
        sub_tuple = []
        for _ in range(sub_tuple_length):
            tensor = torch.zeros(tensor_shape, dtype=torch.float32).cuda()
            op = dist.P2POp(dist.irecv, tensor, src)
            recv_ops.append(op)
            sub_tuple.append(tensor)
        structure.append(tuple(sub_tuple))

    # Prepare to receive last_token_ids
    last_token_ids = torch.zeros(last_token_ids_shape, dtype=torch.long).cuda()
    recv_ops.append(dist.P2POp(dist.irecv, last_token_ids, src))

    # Perform batched receive operations
    reqs = dist.batch_isend_irecv(recv_ops)
    
    # Wait for all operations to complete
    for req in reqs:
        req.wait()
    
    return tuple(structure), last_token_ids
