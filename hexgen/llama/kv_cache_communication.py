import time
import torch
import torch.distributed as dist

def send_logits_with_seqlen_og(logits, seqlen_og, dst):
    metadata = torch.tensor([seqlen_og] + [*logits.shape], dtype=torch.long).cuda()
    dist.send(tensor=metadata, dst=dst)
    dist.send(tensor=logits, dst=dst)
    
def recv_logits_with_seqlen_og(src, dim=3):
    metadata = torch.empty(dim, dtype=torch.long).cuda()
    dist.recv(tensor=metadata, src=src)
    seqlen_og = metadata[0]
    shape = tuple(metadata[1:])
    dtype = torch.float16
    logits = torch.empty(shape, dtype=dtype).cuda()
    dist.recv(tensor=logits, src=src)
    return logits, seqlen_og.item()

def send_key_value_memory_dict(key_value_memory_dict, dst):
    shapes = []
    flat_tensors = []
    for tensor in key_value_memory_dict.values():
        shapes.append(tensor.shape)
        flat_tensors.append(tensor.flatten())
    concatenated_tensor = torch.cat(flat_tensors).cuda()
    shape_lengths = torch.tensor([len(shape) for shape in shapes], dtype=torch.long).cuda()
    flat_shapes = torch.tensor([dim for shape in shapes for dim in shape], dtype=torch.long).cuda()
    dist.send(torch.tensor([len(key_value_memory_dict)], dtype=torch.long).cuda(), dst=dst)
    dist.send(shape_lengths, dst=dst)
    dist.send(flat_shapes, dst=dst)
    dist.send(concatenated_tensor, dst=dst)

def recv_key_value_memory_dict(src):
    num_kv = torch.tensor([0], dtype=torch.long).cuda()
    dist.recv(num_kv, src=src)
    shape_lengths = torch.empty(num_kv.item(), dtype=torch.long).cuda()
    dist.recv(shape_lengths, src=src)
    total_length = shape_lengths.sum().item()
    flat_shapes = torch.empty(total_length, dtype=torch.long).cuda()
    dist.recv(flat_shapes, src=src)
    shapes = []
    offset = 0
    for length in shape_lengths:
        shapes.append(tuple(flat_shapes[offset:offset + length].tolist()))
        offset += length
    concatenated_tensor_size = sum([torch.prod(torch.tensor(shape)).item() for shape in shapes])
    concatenated_tensor = torch.empty(concatenated_tensor_size, dtype=torch.float16).cuda()
    dist.recv(concatenated_tensor, src=src)
    key_value_memory_dict = {}
    offset = 0
    for i, shape in enumerate(shapes):
        num_elements = torch.prod(torch.tensor(shape)).item()
        flat_tensor = concatenated_tensor[offset:offset + num_elements]
        tensor = flat_tensor.reshape(shape)
        key_value_memory_dict[i] = tensor
        offset += num_elements
    return key_value_memory_dict

def kv_cache_communication_send(logits, seqlen_og, key_value_memory_dict, dst):
    send_logits_with_seqlen_og(logits, seqlen_og, dst)
    send_key_value_memory_dict(key_value_memory_dict, dst)

def kv_cache_communication_recv(src):
    logits, seqlen_og = recv_logits_with_seqlen_og(src)
    key_value_memory_dict = recv_key_value_memory_dict(src)
    return logits, seqlen_og, key_value_memory_dict

def main():
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    if rank == 0:
        logits = torch.randn(2, 32000, dtype=torch.float16).cuda()
        seqlen_og = 6
        key_value_memory_dict = {}
        for i in range(32):
            tensor = torch.randn(1, 150, 2, 32, 128, dtype=torch.float16)
            key_value_memory_dict[i] = tensor
        kv_cache_communication_send(logits, seqlen_og, key_value_memory_dict, 1)
    elif rank == 1:
        logits, seqlen_og, key_value_memory_dict = kv_cache_communication_recv(0)
        print(logits.shape, seqlen_og, key_value_memory_dict[0].shape)

if __name__ == "__main__":
    main()
