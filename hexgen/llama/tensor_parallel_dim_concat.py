import torch

def tensor_parallel_dim_concat(logits_list, seqlen_og_list, key_value_memory_dict_list):
    logits = logits_list[0]
    seqlen_og = seqlen_og_list[0]
    key_value_memory_dict = concat_dicts(key_value_memory_dict_list)
    return logits, seqlen_og, key_value_memory_dict

def concat_dicts(list_of_dicts, dim=3):
    # Assuming all dictionaries in the list have the same keys, get the keys from the first dictionary
    keys = list_of_dicts[0].keys()

    # Initialize an empty dictionary to hold the concatenated tensors
    concatenated_dict = {}

    # For each key, concatenate the tensors from all dictionaries along the specified dimension
    for key in keys:
        # Extract the tensor for the current key from each dictionary and concatenate them
        concatenated_dict[key] = torch.cat([d[key] for d in list_of_dicts], dim=dim)

    return concatenated_dict
