import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import time

# Initialize the model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Ensure CUDA is available
if torch.cuda.is_available():
    model.to('cuda')  # Move the model to GPU
    print("Model moved to GPU.")
else:
    print("CUDA is not available. Please check your setup.")
    
model.eval()  # Set the model to evaluation mode

def prefill(model, input_ids):
    """
    Process the initial context to obtain past_key_values for caching,
    ensuring all operations are performed on the GPU.
    """
    with torch.no_grad():
        outputs = model(input_ids=input_ids.to('cuda'), use_cache=True)  # Ensure inputs are on GPU
        past_key_values = outputs.past_key_values
    last_input_ids = input_ids[:, -1:].detach()
    return past_key_values, last_input_ids

# List of different contexts
contexts = [
    "The quick brown fox jumps over the lazy dog",
    "To be or not to be, that is the question",
    "I think, therefore I am",
    "All human beings are born free and equal in dignity and rights",
    "Elementary, my dear Watson",
    "A journey of a thousand miles begins with a single step",
    "The only thing we have to fear is fear itself",
    "That's one small step for man, one giant leap for mankind",
    "In the beginning God created the heavens and the earth",
    "It was the best of times, it was the worst of times"
]

# Dictionaries to store past_key_values and last_input_ids for each context
cached_states = {}

for i, context in enumerate(contexts):
    input_ids = tokenizer.encode(context, return_tensors="pt")
    past_key_values, last_input_ids = prefill(model, input_ids)
    
    # Store in a dictionary with the context (or an index) as the key
    cached_states[i] = {"past_key_values": past_key_values, "last_input_ids": last_input_ids}
    print(cached_states[i]['last_input_ids'])
