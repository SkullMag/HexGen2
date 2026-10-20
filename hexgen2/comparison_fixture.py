"""Deterministic head-dimension-128 fixture; correctness only, never a benchmark."""
import argparse,json
from pathlib import Path
import torch
from transformers import LlamaConfig,LlamaForCausalLM
from safetensors.torch import save_file
from .prepare_comparison import native_layer
from .record import digest

def create(root):
    root.mkdir(parents=True,exist_ok=False);torch.manual_seed(617)
    config=LlamaConfig(vocab_size=256,hidden_size=512,intermediate_size=768,num_hidden_layers=4,
        num_attention_heads=4,num_key_value_heads=4,max_position_embeddings=2048,bos_token_id=1,eos_token_id=2,pad_token_id=0)
    model=LlamaForCausalLM(config).eval().half();model.save_pretrained(root/'hf')
    native=root/'native';native.mkdir();config.save_pretrained(native);state=model.state_dict()
    for i in range(4):save_file(native_layer(state.__getitem__,i,512,4),str(native/f'layer-{i:02}.safetensors'))
    for key,name in [('model.embed_tokens.weight','embedding'),('model.norm.weight','norm'),('lm_head.weight','head')]:
        save_file({'weight':state[key]},str(native/(name+'.safetensors')))
    identity={'kind':'synthetic-random-weights','purpose':'correctness only','seed':617,'files':{p.name:digest(p) for p in native.iterdir()}}
    (native/'verified-source.json').write_text(json.dumps(identity,indent=2)+'\n')
    cases=[{'id':f'fixture-{j}','input_ids':[1]+[3+(i*7+j)%250 for i in range(n-1)],'max_new_tokens':4} for j,n in enumerate([8,31,65])]
    (root/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    (root/'trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in cases))
    print(json.dumps({'fixture':str(root),'parameters':model.num_parameters(),'head_dimension':128}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();create(a.output)
