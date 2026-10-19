import torch.distributed as dist
import torch

def kv_cache_sender(past_key_values, last_token_ids, dst):
    # Send past_key_values
    # First, send the length of past_key_values to let the receiver know how many tensors to expect
    length_tensor = torch.tensor(len(past_key_values), dtype=torch.long).cuda()
    dist.send(tensor=length_tensor, dst=dst)
    
    for layer_kv in past_key_values:
        for tensor in layer_kv:
            # For each tensor, send its shape first
            shape_tensor = torch.tensor(tensor.shape).cuda()
            dist.send(tensor=shape_tensor, dst=dst)
            # Then, send the tensor data
            tensor_contiguous = tensor.contiguous()
            dist.send(tensor=tensor_contiguous, dst=dst)
    
    # Send last_token_ids
    # Send shape of last_token_ids
    shape_tensor = torch.tensor(last_token_ids.shape).cuda()
    dist.send(tensor=shape_tensor, dst=dst)
    # Send last_token_ids data
    dist.send(tensor=last_token_ids, dst=dst)
    # print("Sender has sent the past_key_values and last_token_ids")


def kv_cache_receiver(num_layers, src):
    # Receive past_key_values
    # First, receive the length of past_key_values
    length_tensor = torch.tensor([0], dtype=torch.long).cuda()
    dist.recv(tensor=length_tensor, src=src)
    length = length_tensor.item()
    
    received_past_key_values = []
    for _ in range(length):
        layer_kv = []
        for _ in range(2):  # Assuming each KV pair consists of 2 tensors: key and value
            # Receive the shape of the tensor
            # TODO: currently we need to manually revise the shape_tensor's dim
            shape_tensor = torch.tensor([0, 0, 0, 0], dtype=torch.long).cuda()  # Adjust based on expected shape dims
            dist.recv(tensor=shape_tensor, src=src)
            shape = tuple(shape_tensor.tolist())
            # Receive the tensor data
            # TODO: currently we need to manually define the receive tensor's type (float 32)
            tensor = torch.zeros(shape, dtype=torch.float32).cuda()  # Adjust dtype as necessary
            dist.recv(tensor=tensor, src=src)
            layer_kv.append(tensor)
        received_past_key_values.append(tuple(layer_kv))
    received_past_key_values = tuple(received_past_key_values)
    
    # Receive last_token_ids
    # Receive the shape of last_token_ids
    shape_tensor = torch.tensor([0], dtype=torch.long).cuda()
    dist.recv(tensor=shape_tensor, src=src)
    shape = tuple(shape_tensor.tolist())
    # Allocate and receive last_token_ids data
    last_token_ids = torch.zeros(shape, dtype=torch.long).cuda()  # Adjust dtype as necessary for input_ids
    dist.recv(tensor=last_token_ids, src=src)
    # Manually add an extra dim to last_token_ids
    last_token_ids = last_token_ids.unsqueeze(0) 
    # print("Receiver has received the past_key_values and last_token_ids")
    return received_past_key_values, last_token_ids


'''
# # Some sender receiver prototypes
def sender(tensor, dst):
    # Send tensor shape
    shape_tensor = torch.tensor(tensor.shape).cuda()
    dist.send(tensor=shape_tensor, dst=dst)

    # Send tensor data
    dist.send(tensor=tensor, dst=dst)
    print("Sender has sent the tensor:", tensor)

def receiver(recv_tensor_dim, src):
    # Receive tensor shape
    ndim = recv_tensor_dim
    shape_tensor = torch.tensor([0] * ndim).cuda()
    dist.recv(tensor=shape_tensor, src=src)
    shape = tuple(shape_tensor.tolist())

    # Allocate tensor with received shape
    tensor = torch.zeros(shape, dtype=torch.float16).cuda()
    dist.recv(tensor=tensor, src=src)
    print("Receiver has received the tensor:", tensor)
'''
