import torch
import pickle
import torch.distributed as dist
from transformers import LlamaTokenizer
import os
import sys
sys.path.insert(0, '../../hexgen/')
from hexgen_core.models.gpt import GPTLMHeadModel
from llama.llama_config_utils import llama_config_to_gpt2_config, config_from_checkpoint, overwrite_configs_and_args
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
# from kv_cache_communication import kv_cache_sender, kv_cache_receiver
from inference_phases import prefill, decode
from batch_kv_cache_list import batch_past_key_values, batch_past_key_values_with_padding
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from torch.profiler import profile, record_function, ProfilerActivity
from kv_cache_batch_isend_irecv import kv_cache_send, kv_cache_recv
import queue
import threading
from queue_batch import receiver_thread, batch_process_and_decode

def main():
    # Initialize the process group
    dist.init_process_group(backend='nccl')

    # Determine the rank of the current process
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    world_size = dist.get_world_size()
    prefill_size = 1
    decode_size = 1

    # Model and tokenizer init
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-2.7b", use_cache=True).cuda()
    
    model.eval()
    batch_size = 2
    contexts = ["T " * 1000 for _ in range(100)]
    # Create a tensor and operate based on the rank of the process
    if rank < 1:
        for i, context in enumerate(contexts):
            input_ids = tokenizer.encode(context, return_tensors="pt")
            past_key_values, last_input_ids = prefill(model, input_ids.cuda())
            kv_cache_send(past_key_values, last_input_ids, 1)
        dist.barrier()
    else:
        for i in range(100):
            start_time = time.time()
            past_key_values_list, last_input_ids_list = [], []
            for j in range(batch_size):
                past_key_values, last_input_ids = kv_cache_recv(0)
                past_key_values_list.append(past_key_values)
                last_input_ids_list.append(last_input_ids)
            past_key_values = batch_past_key_values_with_padding(*past_key_values_list)
            last_input_ids = torch.cat(last_input_ids_list, dim=0) 
            generated, decode_time = decode(model, past_key_values, last_input_ids, num_tokens=20)
            end_time = time.time()
            print(f'Comm Time Cost:', end_time - start_time - decode_time)
    # Cleanup the process group
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
