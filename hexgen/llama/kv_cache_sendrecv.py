import torch
import torch.distributed as dist

def quantize_to_int8(tensor):
    """Quantize fp16 tensor values to int8."""
    # Convert tensor from float16 to float32 for more precise computation
    tensor = tensor.float()

    # Find minimum and maximum values to determine scale and offset
    min_val = tensor.min()
    max_val = tensor.max()

    # Normalize tensor to the range 0-255
    # Avoid potential precision issues in float16 by doing calculations in float32
    if max_val != min_val:  # Avoid division by zero if tensor is constant
        normalized_tensor = (tensor - min_val) / (max_val - min_val) * 255
    else:
        normalized_tensor = torch.zeros_like(tensor)  # If constant, all values are the same

    # Convert normalized values to int8
    quantized_tensor = normalized_tensor.clamp(0, 255).char()  # Clamp to ensure values are within the valid range for uint8

    return quantized_tensor

def dequantize_from_int8(quantized_tensor):
    """Dequantize int8 values back to fp16."""
    # Reverse the scaling and normalization
    dequantized_tensor = quantized_tensor.float()
    return dequantized_tensor.to(torch.float16)

def send_data(single_tensor, tensor_dict, dst):
    # Send the single small tensor
    single_tensor_req = dist.isend(tensor=single_tensor, dst=dst)
    
    # Send each tensor in the dictionary
    dict_requests = []
    for tensor in tensor_dict.values():
        tensor = quantize_to_int8(tensor)
        req = dist.isend(tensor=tensor, dst=dst)
        dict_requests.append(req)
    
    # Wait for all sends to complete
    single_tensor_req.wait()
    for req in dict_requests:
        req.wait()

def recv_data(src, num_layers, tensor_shape):
    # Prepare tensor for receiving the single small tensor
    single_tensor_received = torch.zeros(1, dtype=torch.long).cuda()
    single_tensor_req = dist.irecv(tensor=single_tensor_received, src=src)
    
    # Prepare to receive each tensor in the dictionary
    tensor_dict_received = {}
    dict_requests = []
    for i in range(num_layers):
        tensor_received = torch.zeros(*tensor_shape, dtype=torch.int8).cuda()
        req = dist.irecv(tensor=tensor_received, src=src)
        dict_requests.append(req)
        tensor_received = dequantize_from_int8(tensor_received)
        tensor_dict_received[i] = tensor_received
    
    # Wait for all receives to complete
    single_tensor_req.wait()
    for req in dict_requests:
        req.wait()
    
    return single_tensor_received, tensor_dict_received
