import torch
import pickle
import torch.distributed as dist
from transformers import LlamaTokenizer
import os
import sys
sys.path.insert(0, '../../hexgen/')
from hexgen_core.models.gpt import GPTLMHeadModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from kv_cache_communication import kv_cache_sender, kv_cache_receiver
from inference_phases import prefill, decode
from batch_kv_cache_list import batch_past_key_values, batch_past_key_values_with_padding
import copy
import torch
import torch.distributed as dist
import time
import queue
import threading
from torch.cuda import Stream
import torch.multiprocessing as mp
import random


def decide_if_sending():
    # Randomly decide to send data or not; 50% chance of sending
    return random.choice([True, False])


def send_tensors(rank, past_key_values, last_input_ids):
    for i in range(100):  # Send tensors at 1-second intervals
        # Simulate the case that some clients might not have request to send
        will_send = decide_if_sending()
        notification = torch.tensor([rank, int(will_send)], dtype=torch.int).cuda(rank)
        # Notification sent
        dist.send(tensor=notification, dst=3)
        
        # Send request if will_send=True
        if will_send:
            # dist.send(tensor=tensor, dst=3)
            kv_cache_sender(past_key_values, last_input_ids, 3) 
            # print(f"Rank: {rank} send the {i}-th tensor", tensor.tolist())
            # Time span of sending request
            time.sleep(1)


def recv_tensors(recv_stream, tensor_queue, rank, num_prefill):
    # Receive notification to determine which src to receive
    notification = torch.empty(2, dtype=torch.int).cuda(3)
    while True:  # Consider a condition for breaking out of the loop gracefully.
        for src_rank in range(num_prefill):
            with torch.cuda.stream(recv_stream):
                dist.recv(tensor=notification, src=src_rank)
                if notification[1] == 1:
                    # If notified, then recv the msg from direct src
                    tensor = torch.empty(3, device=rank, dtype=torch.int)
                    # dist.recv(tensor=tensor, src=src_rank)
                    past_key_values, last_input_ids = kv_cache_receiver(12, src_rank)
                    past_key_values_clone = copy.deepcopy(past_key_values)
                    last_input_ids_clone = last_input_ids.clone()
                    tensor_queue.put((past_key_values_clone, last_input_ids_clone))


def process_tensors(proc_stream, recv_stream, tensor_queue, stop_signal, model, tokenizer, batch_size):
    batch = []
    past_key_values_list = []
    while not stop_signal.is_set() or (stop_signal.is_set() and not tensor_queue.empty()):
        try:
            # Attempt to gather tensors into a batch within the timeout
            while len(batch) < batch_size and not (stop_signal.is_set() and tensor_queue.empty()):
                try:
                    past_key_values, last_input_ids = tensor_queue.get(timeout=0.1)
                    batch.append(last_input_ids)
                    past_key_values_list.append(past_key_values)
                except queue.Empty:
                    if stop_signal.is_set():
                        break  # Break if stop signal is set and queue is empty

            # Check if there are tensors to process
            if batch:
                with torch.cuda.stream(proc_stream):
                    proc_stream.wait_stream(recv_stream)
                    # Stack and process the batch
                    # If batch size is 1, this effectively processes tensors individually
                    past_key_values = batch_past_key_values_with_padding(*past_key_values_list)
                    last_input_ids = torch.cat(batch, dim=0)
                    generated = decode(model, past_key_values, last_input_ids, num_tokens=100)
                    for i in range(batch_size):
                        print(generated[i])
                        generated_text = tokenizer.decode(generated[i], skip_special_tokens=True)
                        print(generated_text)
                batch = []  # Reset the batch after processing
                past_key_values_list = []

        except queue.Empty:
            # This block may not be needed anymore due to inner try-except
            continue


def main():
    # Initialize the process group
    dist.init_process_group(backend='nccl')

    # Determine the rank of the current process
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    batch_size = 4  # Set the batch size for processing
    num_prefill = 3  # Set the prefill number for processing
    
    # Model and tokenizer init
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", use_cache=True).cuda()
    model.eval()

    initial_context_1 = "Explain how to solve a quadratic equation and provide an example problem with a step-by-step solution."
    initial_context_2 = "In a quiet village on the edge of a forgotten forest, a mysterious door appeared to no one's knowledge. Write a story about what lies beyond that door."
    initial_context_3 = "Write a conversation between a time traveler from the future and a famous historical figure about the impacts of technology on society."
    input_ids_1 = tokenizer.encode(initial_context_1, return_tensors='pt').cuda()
    input_ids_2 = tokenizer.encode(initial_context_2, return_tensors='pt').cuda()
    input_ids_3 = tokenizer.encode(initial_context_3, return_tensors='pt').cuda()
    input_ids = [input_ids_1, input_ids_2, input_ids_3]

    if rank < num_prefill:  # Sender side
        past_key_values, last_input_ids = prefill(model, input_ids[rank])
        send_tensors(rank, past_key_values, last_input_ids)
    else:  # Receiver side
        recv_stream = Stream()
        proc_stream = Stream()
        tensor_queue = queue.Queue()
        stop_signal = threading.Event()

        # Start receiving tensors in a separate thread
        recv_thread = threading.Thread(target=recv_tensors, args=(recv_stream, tensor_queue, rank, num_prefill))
        recv_thread.start()

        # Start processing tensors in another thread
        proc_thread = threading.Thread(target=process_tensors, args=(proc_stream, recv_stream, tensor_queue, stop_signal, model, tokenizer, batch_size))
        proc_thread.start()

        # Wait for the receiving thread to finish
        recv_thread.join()
        stop_signal.set()  # Signal processing thread to exit if queue is empty
        proc_thread.join()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()

