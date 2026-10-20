"""Lossless, self-describing tensor transport over a trusted PyTorch process group.

No pickle, quantization, fixed model dimensions, or asynchronous tensor lifetimes.
NCCL uses the current CUDA device; Gloo uses CPU tensors.
"""

import json
import math

import torch
import torch.distributed as dist

DTYPES = {str(dtype): dtype for dtype in (
    torch.float16, torch.bfloat16, torch.float32, torch.float64,
    torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool,
)}
MAX_HEADER = 1024 * 1024
MAX_PAYLOAD = 8 * 1024**3


def wire_device(group=None):
    return torch.device("cuda", torch.cuda.current_device()) if dist.get_backend(group) == "nccl" else torch.device("cpu")


def validate_header(header):
    if header.get("version") != 1 or not isinstance(header.get("tensors"), list):
        raise ValueError("Unsupported tensor packet")
    if len(header["tensors"]) > 1024:
        raise ValueError("Too many tensors")
    size, names = 0, set()
    for item in header["tensors"]:
        name, shape, dtype = item["name"], item["shape"], item["dtype"]
        if not isinstance(name, str) or name in names or dtype not in DTYPES:
            raise ValueError("Invalid tensor name or dtype")
        if not isinstance(shape, list) or len(shape) > 8 or any(type(d) is not int or d < 0 for d in shape):
            raise ValueError("Invalid tensor shape")
        size += math.prod(shape) * torch.empty((), dtype=DTYPES[dtype]).element_size()
        names.add(name)
    if size > MAX_PAYLOAD:
        raise ValueError("Tensor packet exceeds memory limit")
    return size


def send_packet(tensors, metadata, dst, group=None):
    header = {"version": 1, "metadata": metadata, "tensors": [
        {"name": name, "shape": list(t.shape), "dtype": str(t.dtype)}
        for name, t in tensors.items()
    ]}
    payload_bytes = validate_header(header)
    encoded = json.dumps(header, allow_nan=False).encode("utf-8")
    if not 0 < len(encoded) <= MAX_HEADER:
        raise ValueError("Tensor metadata exceeds limit")
    device = wire_device(group)
    dist.send(torch.tensor([len(encoded)], dtype=torch.int64, device=device), dst=dst, group=group)
    dist.send(torch.tensor(list(encoded), dtype=torch.uint8, device=device), dst=dst, group=group)
    for tensor in tensors.values():
        tensor = tensor.detach().to(device).contiguous()
        if tensor.numel():
            dist.send(tensor, dst=dst, group=group)
    return payload_bytes


def recv_packet(src, group=None):
    device = wire_device(group)
    length = torch.empty(1, dtype=torch.int64, device=device)
    dist.recv(length, src=src, group=group)
    size = length.item()
    if not 0 < size <= MAX_HEADER:
        raise ValueError("Invalid tensor header length")
    encoded = torch.empty(size, dtype=torch.uint8, device=device)
    dist.recv(encoded, src=src, group=group)
    header = json.loads(bytes(encoded.cpu().tolist()).decode("utf-8"))
    validate_header(header)
    tensors = {}
    for item in header["tensors"]:
        tensor = torch.empty(item["shape"], dtype=DTYPES[item["dtype"]], device=device)
        if tensor.numel():
            dist.recv(tensor, src=src, group=group)
        tensors[item["name"]] = tensor
    return tensors, header["metadata"]


def send_data(single_tensor, tensor_dict, dst):
    """Compatibility API for the artifact's missing sendrecv module; lossless."""
    keys = list(tensor_dict)
    if any(type(key) is not int for key in keys):
        raise ValueError("KV-cache layer keys must be integers")
    return send_packet({"token": single_tensor, **{
        f"kv/{key}": tensor_dict[key] for key in keys
    }}, {"layers": keys}, dst)


def recv_data(src, num_layers=None, tensor_shape=None):
    tensors, metadata = recv_packet(src)
    cache = {key: tensors[f"kv/{key}"] for key in metadata["layers"]}
    if num_layers is not None and len(cache) != num_layers:
        raise ValueError("Received KV-cache layer count differs from expected count")
    if tensor_shape is not None and any(tuple(t.shape) != tuple(tensor_shape) for t in cache.values()):
        raise ValueError("Received KV-cache shape differs from expected shape")
    return tensors["token"], cache
