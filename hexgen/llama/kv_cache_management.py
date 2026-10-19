import torch
import random
import queue
import time
import copy
from kv_cache_communication import kv_cache_communication_send, kv_cache_communication_recv
from kv_cache_batch import batch_logits_and_key_value_memory_dict 
import torch.distributed as dist
from inference_phases import decode_only, prefill_only
import threading

def decide_if_sending():
    # Randomly decide to send data or not; 50% chance of sending
    return random.choice([True, True])


def coordinator_send(rank, decode_index, logits, seqlen_og, key_value_memory_dict):
    # Simulate the case that some clients might not have request to send
    will_send = decide_if_sending()
    notification = torch.tensor([rank, int(will_send)], dtype=torch.int).cuda(rank)
    # Notification sent
    dist.send(tensor=notification, dst=decode_index)

    # Send request if will_send=True
    if will_send:
        kv_cache_communication_send(logits, seqlen_og, key_value_memory_dict, decode_index)
        # print(f"Rank: {rank} send the {i}-th tensor", tensor.tolist())
        # Time span of sending request


def coordinator_recv_and_process(tensor_queue, rank, prefill_size, decode_index, model, batch_size, hidden_size, forward_step_func, max_length, last_pp_stage_id, temperature, top_k, top_p):
    # Receive notification to determine which src to receive
    logits_list = []
    seqlen_og_list = []
    key_value_memory_dict_list = []

    notification = torch.empty(2, dtype=torch.int).cuda(decode_index)
    while True:  # Consider a condition for breaking out of the loop gracefully.
        for src_rank in range(prefill_size):
            dist.recv(tensor=notification, src=src_rank)
            if notification[1] == 1:
                start_time = time.time()
                logits, seqlen_og, key_value_memory_dict = kv_cache_communication_recv(src_rank)
                key_value_memory_dict_clone = copy.deepcopy(key_value_memory_dict)
                seqlen_og_clone = copy.deepcopy(seqlen_og)
                logits_clone = logits.clone()
                print('Recv message and add to queue ...')
                tensor_queue.put((logits_clone, seqlen_og_clone, key_value_memory_dict_clone))
                if tensor_queue.qsize() >= batch_size:
                    while not tensor_queue.empty():
                        logits, seqlen_og, key_value_memory_dict = tensor_queue.get(timeout=0.1)
                        logits_list.append(logits)
                        seqlen_og_list.append(seqlen_og)
                        key_value_memory_dict_list.append(key_value_memory_dict)
                        if len(logits_list) >= batch_size:
                            logits, key_value_memory_dict = batch_logits_and_key_value_memory_dict(logits_list, key_value_memory_dict_list)
                            seqlen_og = max(seqlen_og_list)
                            print("Process message with batch size:", len(logits_list), '...')
                            output = decode_only(logits, seqlen_og, key_value_memory_dict, model, hidden_size, forward_step_func, max_length, last_pp_stage_id, temperature, top_k, top_p, timing=True)
                            end_time = time.time()
                            logits_list = []
                            seqlen_og_list = []
                            key_value_memory_dict_list = []
                            print(f"Recv and Decode Time: {(end_time - start_time) * 1000} ms")
