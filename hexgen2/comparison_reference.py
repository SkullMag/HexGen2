"""Independent eager Hugging Face FP16 oracle, with optional CPU weight offload."""
import argparse,json,time
from pathlib import Path
import torch
from transformers import LlamaForCausalLM
from .record import digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True);p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--offload',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=LlamaForCausalLM.from_pretrained(a.model_dir,torch_dtype=torch.float16,attn_implementation='eager',low_cpu_mem_usage=True,local_files_only=True).eval()
    if a.offload:
        from accelerate import cpu_offload
        cpu_offload(model,execution_device=torch.device('cuda:0'),offload_buffers=True)
    else:model.cuda()
    cases=json.loads(a.cases.read_text());results=[]
    for case in cases:
        inputs=torch.tensor([case['input_ids']],device='cuda',dtype=torch.long);past=None;tokens=[];logits=[];started=time.time_ns()
        with torch.inference_mode():
            for i in range(case['max_new_tokens']):
                output=model(inputs,past_key_values=past,use_cache=True);last=output.logits[:,-1];token=last.argmax(-1)
                logits.append(last.float().cpu());tokens.extend(token.cpu().tolist());past=output.past_key_values;inputs=token[:,None]
        path=a.output/(case['id']+'.pt');torch.save(torch.cat(logits),path)
        result={'id':case['id'],'input_ids':case['input_ids'],'max_new_tokens':case['max_new_tokens'],'token_ids':tokens,'logits_file':path.name,'logits_sha256':digest(path),'started_utc_ns':started,'ended_utc_ns':time.time_ns()}
        results.append(result);(a.output/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='input_ids'}),flush=True)
    (a.output/'manifest.json').write_text(json.dumps({'kind':'independent-huggingface-eager-fp16','offload':a.offload,'model_dir':a.model_dir,'cases_sha256':digest(a.cases),'source_sha256':digest(__file__),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'tf32':False},indent=2)+'\n')
if __name__=='__main__':main()
