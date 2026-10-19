import torch
import random
import queue
import time
import copy
from kv_cache_communication import kv_cache_communication_send, kv_cache_communication_recv
from kv_cache_batch import batch_logits_and_key_value_memory_dict 
import torch.distributed as dist
from hexgen_core import decode_only, prefill_only
import threading
from tensor_parallel_dim_concat import tensor_parallel_dim_concat

def decide_if_sending():
    # Randomly decide to send data or not; 50% chance of sending
    return random.choice([True, True])


def coordinator_send(rank, logits, seqlen_og, key_value_memory_dict, prefill_size, decode_size):
    for i in range(100):
        # Simulate the case that some clients might not have request to send
        will_send = decide_if_sending()
        notification = torch.tensor([rank, int(will_send)], dtype=torch.int).cuda(rank)
        # Notification sent
        dist.send(tensor=notification, dst=2)
        # print(f"rank {rank} send message to rank {rank%2+4}")
        # Send request if will_send=True
        if will_send:
            kv_cache_communication_send(logits, seqlen_og, key_value_memory_dict, 2)
            # print(f"rank {rank} send message to rank {rank%2+4}")
            # print(f"Rank: {rank} send the {i}-th tensor", tensor.tolist())
        # Time span of sending request
        time.sleep(0.1)


def coordinator_recv_and_process(tensor_queue, rank, model, batch_size, prefill_size, decode_size, hidden_size, forward_step_func, max_length, last_pp_stage_id, temperature, top_k, top_p):
    # Receive notification to determine which src to receive
    logits_tp_list = []
    seqlen_og_tp_list = []
    key_value_memory_dict_tp_list = []
    logits_list = []
    seqlen_og_list = []
    key_value_memory_dict_list = []

    notification = torch.empty(2, dtype=torch.int).cuda()
    while True:  # Consider a condition for breaking out of the loop gracefully.
        for src_rank in range(prefill_size//decode_size):
            dist.recv(tensor=notification, src=rank%2)
            # print(f"rank {rank} recv message from rank {src_rank*2+rank%4}")
            if notification[1] == 1:
                logits, seqlen_og, key_value_memory_dict = kv_cache_communication_recv(rank%2)
                # print(f"rank {rank} recv message from rank {src_rank*2+rank%4}")
                key_value_memory_dict_clone = copy.deepcopy(key_value_memory_dict)
                seqlen_og_clone = copy.deepcopy(seqlen_og)
                logits_clone = logits.clone()
                print('Recv message and add to queue ...')
                tensor_queue.put((logits_clone, seqlen_og_clone, key_value_memory_dict_clone))
                if tensor_queue.qsize() >= batch_size:
                    while not tensor_queue.empty():
                        logits, seqlen_og, key_value_memory_dict = tensor_queue.get(timeout=0.1)
                        logits_tp_list.append(logits)
                        seqlen_og_tp_list.append(seqlen_og)
                        key_value_memory_dict_tp_list.append(key_value_memory_dict)
                        if len(logits_tp_list) == 1:
                            logits, seqlen_og, key_value_memory_dict = tensor_parallel_dim_concat(logits_tp_list, seqlen_og_tp_list, key_value_memory_dict_tp_list)
                            logits_list.append(logits)
                            seqlen_og_list.append(seqlen_og)
                            key_value_memory_dict_list.append(key_value_memory_dict)
                            logits_tp_list = []
                            seqlen_og_tp_list = []
                            key_value_memory_dict_tp_list = []
                        if len(logits_list) >= batch_size:
                            logits, key_value_memory_dict = batch_logits_and_key_value_memory_dict(logits_list, key_value_memory_dict_list)
                            seqlen_og = max(seqlen_og_list)
                            print("Process message with batch size:", len(logits_list), '...')
                            output = decode_only(logits, seqlen_og, key_value_memory_dict, model, hidden_size, forward_step_func, max_length, last_pp_stage_id, temperature, top_k, top_p, timing=True)
                            logits_list = []
                            seqlen_og_list = []
                            key_value_memory_dict_list = []
