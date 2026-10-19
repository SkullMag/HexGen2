import torch
import torch.nn.functional as F

def batch_past_key_values(*args):
    """
    Batch an arbitrary number of sets of past_key_values together.
    
    Parameters:
    - *args: An arbitrary number of sets of past_key_values.
    
    Returns:
    - batched_past_key_values: The batched past_key_values.
    """
    batched_past_key_values = []
    
    # Assuming all past_key_values have the same number of layers
    num_layers = len(args[0])
    
    # Iterate over each layer
    for layer_index in range(num_layers):
        # Initialize empty lists to collect keys and values for this layer from all sets
        layer_keys = []
        layer_values = []
        
        # Iterate over each set of past_key_values
        for past_key_values in args:
            layer_kv = past_key_values[layer_index]
            key, value = layer_kv
            
            layer_keys.append(key)
            layer_values.append(value)
        
        # Batch the keys and values along the new dimension (dim=0)
        batched_key = torch.cat(layer_keys, dim=0)
        batched_value = torch.cat(layer_values, dim=0)
        
        # Add the batched key-value pairs for this layer to the result
        batched_past_key_values.append((batched_key, batched_value))
    
    return tuple(batched_past_key_values)

def batch_past_key_values_with_padding(*args):
    """
    Batch an arbitrary number of sets of past_key_values together with padding for variable lengths.

    Parameters:
    - *args: An arbitrary number of sets of past_key_values.

    Returns:
    - batched_past_key_values: The batched past_key_values with padding.
    """
    batched_past_key_values = []

    # Assuming all past_key_values have the same number of layers
    num_layers = len(args[0])

    # Iterate over each layer
    for layer_index in range(num_layers):
        # Initialize lists to collect keys and values for this layer from all sets
        layer_keys = []
        layer_values = []

        # Determine the maximum length in this layer for keys and values
        max_key_length = max(key.shape[2] for pv in args for key, _ in [pv[layer_index]])
        max_value_length = max(value.shape[2] for pv in args for _, value in [pv[layer_index]])
        
        # Iterate over each set of past_key_values to pad and collect keys and values
        for past_key_values in args:
            layer_kv = past_key_values[layer_index]
            key, value = layer_kv

            # Pad the key and value tensors to the maximum length
            padded_key = F.pad(key, (0, 0, 0, max_key_length - key.shape[2]))
            padded_value = F.pad(value, (0, 0, 0, max_value_length - value.shape[2]))

            layer_keys.append(padded_key)
            layer_values.append(padded_value)

        # Batch the padded keys and values along the new dimension (dim=0)
        batched_key = torch.cat(layer_keys, dim=0)
        batched_value = torch.cat(layer_values, dim=0)

        # Add the batched key-value pairs for this layer to the result
        batched_past_key_values.append((batched_key, batched_value))

    return tuple(batched_past_key_values)
