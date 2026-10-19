import torch
from torch import nn
import argparse
from tqdm import tqdm
import numpy as np
import random
import h5py
import time
import os
import sys
import threading
import asyncio
from arguments import add_arguments
sys.path.insert(0, '..')
sys.path.insert(0, '../hexgen_core')
sys.path.insert(0, '../../third_party/megatron')
sys.path.insert(0, './modules')
from inference_phases import prefill_only, decode_only
from megatron.initialize import initialize_megatron
from megatron import get_args
from torch.utils.data.distributed import DistributedSampler
from hybrid_parallel_model_dist_dev import get_hybrid_parallel_configs, construct_hybrid_parallel_model, overwrite_megatron_args
from typing import Tuple, List
from llama_config_utils import llama_config_to_gpt2_config, config_from_checkpoint, overwrite_configs_and_args
from transformers import GPT2Config, GPT2Tokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
from hexgen_core.models.gpt import GPTLMHeadModel, shard_state_dict_tp, create_mixer_cls, create_mlp_cls
from hexgen_core import gen_hetero_groups
from load_model_parameters_utils.load_model_parameters import load_model_parameters
from kv_cache_communication import kv_cache_communication_send, kv_cache_communication_recv, send_key_value_memory_dict, recv_key_value_memory_dict, send_logits_with_seqlen_og, recv_logits_with_seqlen_og
from kv_cache_batch import batch_logits_and_key_value_memory_dict 
from kv_cache_management_dev import coordinator_send, coordinator_recv_and_process
from arguments import get_kv_cache, set_kv_cache, clear_kv_cache
from torch.cuda import Stream
import queue
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
import torch.distributed as dist
from tensor_parallel_dim_concat import concat_dicts, tensor_parallel_dim_concat
from gen_hetero_groups_dev_fake import gen_fake_hetero_groups
from kv_cache_sendrecv_int2 import send_data, recv_data
import csv
from collections import deque

def set_seed():
    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def forward_step_func(inputs, inference_params, position_ids, model):
    if isinstance(inputs, (Tuple, List)):
        outputs = model(*inputs, position_ids=position_ids, inference_params=inference_params)
    else:
        outputs = model(inputs, position_ids=position_ids, inference_params=inference_params)
    return outputs

def create_model(args):
    if 'benchmark' in os.path.abspath('..'):
        os.chdir("../../hexgen/llama")    

    local_rank = args.local_rank
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = torch.distributed.get_world_size()

    llama_config = config_from_checkpoint('./llama-config/', args.model_size)
    config = llama_config_to_gpt2_config(llama_config)
    overwrite_configs_and_args(config, args)
    overwrite_megatron_args(config, args)
    
    hybrid_parallel_configs = get_hybrid_parallel_configs(args)
    
    if rank < 2:
        hetero_groups = gen_fake_hetero_groups(hetero_config=[2], pp_partition=[60], layer_num=60, current_tp_rank=0, rank=rank)
    else:
        hetero_groups = gen_fake_hetero_groups(hetero_config=[2], pp_partition=[60], layer_num=60, current_tp_rank=2, rank=rank)

    hetero_groups_0 = gen_hetero_groups(hetero_config=[2], pp_partition=[60], layer_num=60, current_tp_rank=0, rank=0)

    tp_groups = [
            hetero_groups_0['tp_groups'][0],
            ]

    process_groups_whole_model = [
            hetero_groups_0['process_groups_whole_model'],
            ]

    if local_rank == 0:
        print("Creating Model...")

    # Init model on meta device
    mixed_precision = {'fp32': torch.float, 'fp16': torch.float16, 'bf16': torch.bfloat16}[args.mixed_precision]
    gpt_model = GPTLMHeadModel(config, device='meta' if args.initialize_on_meta else 'cpu', dtype=mixed_precision)
    from flash_attn.models.gpt import create_mixer_cls, create_mlp_cls
    factory_kwargs = {'device': 'meta' if args.initialize_on_meta else 'cpu', 'dtype': mixed_precision}
    for i in range(config.num_hidden_layers):
        layer = gpt_model.transformer.layers[i]
        setattr(layer, 'mixer', create_mixer_cls(config, layer_idx=i, process_group=tp_groups[rank//2], **factory_kwargs)(config.hidden_size))
        setattr(layer, 'mlp', create_mlp_cls(config, layer_idx=i, process_group=tp_groups[rank//2], **factory_kwargs)(config.hidden_size))
    
    # Construct hybrid parallel model
    model = construct_hybrid_parallel_model(model=gpt_model, 
                                            model_config=config, 
                                            inference_args=args, 
                                            hybrid_parallel_configs=hybrid_parallel_configs,
                                            pp_partition=args.pp_partition,
                                            device=device,
                                            prefill_size=args.prefill_size,
                                            decode_size=args.decode_size,
                                            hetero_config=args.hetero_config,
                                            hetero_groups=hetero_groups,
                                            tp_groups=tp_groups[rank//2],
                                            process_groups_whole_model=process_groups_whole_model,
                                            )
    
    if rank == 0:
        print('Model configures:')
        print(config)
    time.sleep(rank * 0.1)

    # Initialize the tokenizer for the GPT model.
    tokenizer = LlamaTokenizer.from_pretrained("../../../Llama-2-7b-chat-hf/") 

    return model, tokenizer, hetero_groups['pp_rank_groups']

def inference(model, tokenizer, pp_groups, model_msg, args, queue, req_idx):
    # current rank
    rank = torch.distributed.get_rank()
    
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Tokenize the provided prompt text.
    prompt_text = model_msg['prompt']
    max_length = model_msg['max_new_tokens']
    temperature = model_msg['temperature']
    top_k = model_msg['top_k']
    top_p = model_msg['top_p']
    max_length += len(prompt_text) // 2
    
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").cuda()
    queue.append(input_ids)
    if len(queue) >= 4:
        input_idss = []
        for i in range(4):
            input_idss.append(queue.popleft())
        input_ids = torch.cat(input_idss, dim=0)
        print(input_ids.shape)
        input_ids_shape = [[-1, len(input_ids[0]), args.hidden_size], [-1, len(input_ids[0])], [-1, len(input_ids[0]), args.hidden_size]]
        start_time = time.time()
        next_token, key_value_memory_dict = prefill_only(input_ids, input_ids_shape, model, forward_step_func, max_length, pp_last_stage_rank=pp_groups[0][-1], temperature=temperature, top_k=top_k, top_p=top_p, timing=True)
        if rank == 0:
            csv.writer(open('conversation_workload/prefill_end_time.csv', 'a', newline='')).writerow([time.time()])
        output = decode_only(next_token, 1, len(prompt_text)//2+2, key_value_memory_dict, model, args.hidden_size, forward_step_func, max_length, pp_groups[0][-1], temperature, top_k, top_p, timing=True)
        end_time = time.time()
        if rank == 0:
            csv.writer(open('conversation_workload/decode_end_time.csv', 'a', newline='')).writerow([time.time()])
        infer_time = end_time - start_time
    
    infer_time = 0
    return infer_time


if __name__ == '__main__':
    initialize_megatron(extra_args_provider=add_arguments)
    args = get_args()
    set_seed()

    model, tokenizer, pp_groups = create_model(args)

    model_msg = {
        'prompt': "D " * 1024,
        'max_new_tokens': 150,
        'temperature': 1,
        'top_k': 1, 
        'top_p': 1, 
    }
    
    queue = deque()
    req_idx = 0
    inference(model, tokenizer, pp_groups, model_msg, args, queue, req_idx)
