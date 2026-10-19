import torch
import torch.distributed as dist

def send_data(single_tensor, tensor_dict, dst):
    # Send the single small tensor using its original type
    single_tensor_req = dist.isend(tensor=single_tensor, dst=dst)

    # Concatenate and send tensors in the dictionary
    batched_tensor = torch.stack(list(tensor_dict.values()), dim=0)
    batched_tensor_req = dist.isend(tensor=batched_tensor, dst=dst)

    # Wait for both sends to complete
    single_tensor_req.wait()
    batched_tensor_req.wait()

def recv_data(src, num_layers):
    # Receive the single small tensor
    single_tensor_received = torch.zeros(1, dtype=torch.long).cuda()
    single_tensor_req = dist.irecv(tensor=single_tensor_received, src=src)

    # Receive the batched tensor
    batched_tensor_received = torch.zeros(num_layers, 1, 1056, 2, 32, 128, dtype=torch.float16).cuda()
    batched_tensor_req = dist.irecv(tensor=batched_tensor_received, src=src)

    # Wait for both receives to complete
    single_tensor_req.wait()
    batched_tensor_req.wait()

    tensor_dict_received = {i: batched_tensor_received[i] for i in range(num_layers)}

    return single_tensor_received, tensor_dict_received
