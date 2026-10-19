import torch
from kv_cache_batch_isend_irecv import kv_cache_recv
import threading
import queue
import time

def receiver_thread(recv_queue, src_rank):
    while True:
        try:
            # Non-blocking or timeout-aware receiving
            data = kv_cache_recv(src_rank)
            recv_queue.put(data)
        except Exception as e:
            print(f"Error receiving data: {e}")
            break

def batch_process_and_decode(recv_queue, model, batch_size=5):
    while True:
        batch_data = []
        while len(batch_data) < batch_size and not recv_queue.empty():
            data = recv_queue.get()
            batch_data.append(data)

        if batch_data:
            # Assuming you adapt decode to handle batches
            # For simplicity, I'm processing them one by one here
            for past_key_values, last_input_ids in batch_data:
                generated = decode(model, past_key_values, last_input_ids, num_tokens=10)
                print(generated)  # Or handle the generated output as needed

            print(f'Processed a batch of {len(batch_data)} requests.')

        # Break or sleep here based on your application's needs
        # For continuous operation, consider a condition to exit this loop
        time.sleep(1)  # Avoid tight loop if queue is empty
