from dataclasses import dataclass
from arguments import get_args
from typing import List
from device import Device


@dataclass
class Config:
    args = get_args()

    model_size = args.model_size

    batch_size = args.batch_size
    seq_in = args.seq_in
    seq_out = args.seq_out

    # graph partition config
    niter = args.niter

    # network config
    inter_bw = args.inter_bw
    specs: List[List] = None

    # utils
    device_machine_map = None
    devices : List[Device] = None

    # model config
    S = 4096
    V = 32000
    B_type = 2

    if model_size == 'llama-30b':
        H = 6656
        L = 60
        N_attn_heads = 52
        P = 30
    elif model_size == 'llama-7b':
        H = 4096
        L = 32
        N_attn_heads = 32
        P = 7
    elif model_size == 'llama-13b':
        H = 5120
        L = 40
        N_attn_heads = 40
        P = 13
    elif model_size == 'llama-70b':
        H = 8192
        L = 80
        N_attn_heads = 64
        P = 7

    