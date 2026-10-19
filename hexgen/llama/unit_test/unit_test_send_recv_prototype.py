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


def send_tensors(rank):
    for i in range(100):  # Send tensors at 1-second intervals
        # Simulate the case that some clients might not have request to send
        will_send = decide_if_sending()
        notification = torch.tensor([rank, int(will_send)], dtype=torch.int).cuda(rank)
        # Notification sent
        dist.send(tensor=notification, dst=3)
        
        # Send request if will_send=True
        if will_send:
            tensor = torch.tensor([1*rank, 2*rank, 3*rank], dtype=torch.int).cuda(rank)
            dist.send(tensor=tensor, dst=3)
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
                    dist.recv(tensor=tensor, src=src_rank)
                    tensor_clone = tensor.clone()
                    tensor_queue.put(tensor_clone)


def process_tensors(proc_stream, recv_stream, tensor_queue, stop_signal, batch_size):
    batch = []
    while not stop_signal.is_set() or (stop_signal.is_set() and not tensor_queue.empty()):
        try:
            # Attempt to gather tensors into a batch within the timeout
            while len(batch) < batch_size and not (stop_signal.is_set() and tensor_queue.empty()):
                try:
                    tensor = tensor_queue.get(timeout=0.1)
                    batch.append(tensor)
                except queue.Empty:
                    if stop_signal.is_set():
                        break  # Break if stop signal is set and queue is empty

            # Check if there are tensors to process
            if batch:
                with torch.cuda.stream(proc_stream):
                    proc_stream.wait_stream(recv_stream)
                    # Stack and process the batch
                    # If batch size is 1, this effectively processes tensors individually
                    batch_tensor = torch.stack(batch, dim=0)
                    processed_batch = batch_tensor * 2  # Example processing operation
                    print("Receiver: Processed batch", processed_batch.tolist())
                batch = []  # Reset the batch after processing

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
    
    if rank < num_prefill:  # Sender side
        send_tensors(rank)
    else:  # Receiver side
        recv_stream = Stream()
        proc_stream = Stream()
        tensor_queue = queue.Queue()
        stop_signal = threading.Event()

        # Start receiving tensors in a separate thread
        recv_thread = threading.Thread(target=recv_tensors, args=(recv_stream, tensor_queue, rank, num_prefill))
        recv_thread.start()

        # Start processing tensors in another thread
        proc_thread = threading.Thread(target=process_tensors, args=(proc_stream, recv_stream, tensor_queue, stop_signal, batch_size))
        proc_thread.start()

        # Wait for the receiving thread to finish
        recv_thread.join()
        stop_signal.set()  # Signal processing thread to exit if queue is empty
        proc_thread.join()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()

