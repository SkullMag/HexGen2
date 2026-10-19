import torch
import pickle
import torch.distributed as dist
from transformers import LlamaTokenizer
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from inference_phases import prefill, decode
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from torch.profiler import profile, record_function, ProfilerActivity

def main():
    # Initialize the process group
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-2.7b", use_cache=True).cuda()

    model.eval()
    
    context = "Can you explain what is friend ?"
    input_ids = tokenizer.encode(context, return_tensors="pt")
    past_key_values, last_input_ids = prefill(model, input_ids.cuda())
    print(last_input_ids)
    generated, decode_time = decode(model, past_key_values, last_input_ids, num_tokens=30)
    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    print(generated_text)
    
if __name__ == '__main__':
    main()
