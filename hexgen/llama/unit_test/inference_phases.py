import torch
import time

def prefill(model, input_ids):
    start_time = time.time()    
    """
    Process the initial context to obtain past_key_values for caching.
    
    Parameters:
    - context (str): The initial text context.
    
    Returns:
    - past_key_values (tuple): Cached key-value states for the initial context.
    - input_ids (torch.Tensor): The input IDs corresponding to the last token of the context.
    """

    # Forward pass to get past_key_values without generating new tokens
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        past_key_values = outputs.past_key_values
    end_time = time.time()
    print("Prefill_time:", end_time - start_time)
    return past_key_values, input_ids[:, -1:]

def decode(model, past_key_values, input_ids, num_tokens=200):
    """
    Continue generating text from cached past_key_values.
    
    Parameters:
    - past_key_values (tuple): Cached key-value states from the prefill phase.
    - input_ids (torch.Tensor): The input IDs to start generating from.
    - num_tokens (int): Number of tokens to generate.
    
    Returns:
    - generated_text (str): The generated text including the context.
    """
    generated = input_ids
    start_time = time.time()
    for _ in range(num_tokens):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
            logits, past_key_values = outputs.logits, outputs.past_key_values

            # Get the last predicted token
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            generated = torch.cat((generated, next_token), dim=1)

            # Update input_ids for the next iteration
            input_ids = next_token
    end_time = time.time()
    print("Decode_time:", end_time - start_time)
    return generated, end_time - start_time
