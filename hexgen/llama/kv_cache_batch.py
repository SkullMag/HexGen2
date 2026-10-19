import torch
import torch.nn.functional as F

def batch_logits_and_key_value_memory_dict(logits_list, kv_memory_dicts_list):
    # Concatenate all logits tensors along the first dimension
    batched_logits = torch.cat(logits_list, dim=0)
    
    # Initialize a new dictionary to store the batched key-value memories
    batched_kv_memory_dict = {}
    
    # Assuming all dictionaries have the same keys
    for key in kv_memory_dicts_list[0].keys():
        # Find the maximum size in the second dimension for current key across all dictionaries
        max_size = max(d[key].size(1) for d in kv_memory_dicts_list)
        # Pad tensors in the second dimension as necessary and concatenate
        tensors_to_batch = []
        for d in kv_memory_dicts_list:
            current_tensor = d[key]
            pad_size = max_size - current_tensor.size(1)
            # Pad if necessary (pad only the second dimension)
            if pad_size > 0:
                padded_tensor = F.pad(current_tensor, (0, 0, 0, 0, 0, 0, 0, pad_size), "constant", 0)
                tensors_to_batch.append(padded_tensor)
            else:
                tensors_to_batch.append(current_tensor)
        
        batched_kv_memory_dict[key] = torch.cat(tensors_to_batch, dim=0)
    
    return batched_logits, batched_kv_memory_dict

def main():
    # Example usage with varying second dimension
    logits_list = [torch.randn(1, 32000, dtype=torch.float16), torch.randn(3, 32000, dtype=torch.float16)]
    kv_memory_dicts_list = [
        {i: torch.randn(1, 150, 2, 32, 128, dtype=torch.float16) for i in range(32)},
        {i: torch.randn(1, 128, 2, 32, 128, dtype=torch.float16) for i in range(32)}  # Note the different second dimension
    ]

    batched_logits, batched_kv_memory_dict = batch_logits_and_key_value_memory_dict(logits_list, kv_memory_dicts_list)

    print(f"Batched logits shape: {batched_logits.shape}")
    # Print the shape of a sample batched key-value memory tensor to check the batching
    print(f"Batched key-value memory sample shape: {batched_kv_memory_dict[0].shape}")

if __name__ == "__main__":
    main()
