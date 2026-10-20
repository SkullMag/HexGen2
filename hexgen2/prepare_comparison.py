"""Fetch the immutable 13B checkpoint and convert native layers with provenance."""
import argparse,gc,hashlib,json,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
import torch
from transformers import LlamaConfig
from safetensors.torch import save_file
from .record import digest
REPO='openlm-research/open_llama_13b'
REVISION='b6d7fde8392250730d24cc2fcfa3b7e5f9a03ce8'

def download(directory):
    directory.mkdir(parents=True,exist_ok=True)
    result=requests.get(f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true',timeout=90)
    result.raise_for_status();metadata=result.json()
    if metadata['sha']!=REVISION:raise ValueError('Model revision mismatch')
    files=[f for f in metadata['siblings'] if f['rfilename'].endswith(('.bin','.json','.model'))]
    origin={'repository':REPO,'revision':REVISION,'files':[{'name':f['rfilename'],'size':f.get('lfs',{}).get('size',f.get('size')),'upstream_sha256':f.get('lfs',{}).get('sha256')} for f in files]}
    (directory/'download-source.json').write_text(json.dumps(origin,indent=2)+'\n')
    def one(item):
        name=item['rfilename'];target=directory/name;size=item.get('lfs',{}).get('size',item.get('size'));expected=item.get('lfs',{}).get('sha256')
        if not isinstance(size,int):raise ValueError('Missing upstream size: '+name)
        if target.exists() and target.stat().st_size==size and (not expected or digest(target)==expected):return {'name':name,'bytes':size,'sha256':digest(target),'upstream_sha256':expected}
        partial=target.with_suffix(target.suffix+'.partial');offset=partial.stat().st_size if partial.exists() else 0
        if offset>size:raise ValueError('Oversized partial file')
        session=requests.Session();resolved=None
        while offset<size:
            end=min(offset+(16<<20),size)-1
            for attempt in range(6):
                try:
                    url=resolved or f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}?download=true&range_probe={time.time_ns()}'
                    response=session.get(url,headers={'Range':f'bytes={offset}-{end}'},stream=True,timeout=120)
                    if response.status_code not in [200,206]:raise ValueError('HTTP '+str(response.status_code))
                    if response.status_code==200 and (offset!=0 or end!=size-1):raise ValueError('Range response required')
                    if response.status_code==206 and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-{end}/'):raise ValueError('Wrong byte range')
                    resolved=response.url;received=0
                    with partial.open('ab') as stream:
                        for block in response.iter_content(1<<20):stream.write(block);received+=len(block)
                    response.close()
                    if received!=end-offset+1:raise ValueError('Incomplete range')
                    offset=end+1
                    if offset//(512<<20)!=(offset-received)//(512<<20) or offset==size:print('DOWNLOAD_BYTES',name,offset,size,flush=True)
                    break
                except Exception as error:
                    resolved=None
                    if partial.exists():
                        with partial.open('r+b') as stream:stream.truncate(offset)
                    if attempt==5:raise RuntimeError('Checkpoint range transfer failed: '+name+' '+type(error).__name__) from None
                    time.sleep(2)
        actual=digest(partial)
        if expected and actual!=expected:raise ValueError('Upstream SHA mismatch: '+name)
        partial.replace(target);return {'name':name,'bytes':size,'sha256':actual,'upstream_sha256':expected}
    with ThreadPoolExecutor(3) as pool:verified=list(pool.map(one,files))
    origin['verified_files']=verified
    (directory/'verified-source.json').write_text(json.dumps(origin,indent=2)+'\n')
    return origin

def native_layer(get,i,hidden,heads):
    prefix=f'model.layers.{i}.';dimension=hidden//heads
    def inv(w):return w.reshape(heads,2,dimension//2,hidden).transpose(1,2).reshape(-1,hidden)
    return {'mlp.fc1.weight':torch.cat([get(prefix+'mlp.up_proj.weight'),get(prefix+'mlp.gate_proj.weight')]),
            'mlp.fc2.weight':get(prefix+'mlp.down_proj.weight').contiguous(),
            'norm1.weight':get(prefix+'input_layernorm.weight').contiguous(),
            'norm2.weight':get(prefix+'post_attention_layernorm.weight').contiguous(),
            'mixer.Wqkv.weight':torch.cat([inv(get(prefix+'self_attn.q_proj.weight')),inv(get(prefix+'self_attn.k_proj.weight')),get(prefix+'self_attn.v_proj.weight')]),
            'mixer.out_proj.weight':get(prefix+'self_attn.o_proj.weight').contiguous()}

def convert(raw,output,stages):
    config=LlamaConfig.from_pretrained(raw,local_files_only=True)
    if config.num_key_value_heads!=config.num_attention_heads:raise ValueError('This conversion is specifically the pinned MHA 13B model')
    output.mkdir(parents=True,exist_ok=True)
    index=json.loads((raw/'pytorch_model.bin.index.json').read_text())['weight_map']
    shards={name:torch.load(raw/name,map_location='cpu',mmap=True,weights_only=True) for name in sorted(set(index.values()))}
    def get(key):
        tensor=shards[index[key]][key]
        if tensor.dtype!=torch.float16:raise ValueError('Pinned checkpoint is not FP16: '+key)
        return tensor
    for stage in stages:
        half=config.num_hidden_layers//2
        for i in range(stage*half,(stage+1)*half):
            path=output/f'layer-{i:02}.safetensors'
            if not path.exists():save_file(native_layer(get,i,config.hidden_size,config.num_attention_heads),str(path))
            print('NATIVE_LAYER_READY',i,flush=True);gc.collect()
        specials=[('model.embed_tokens.weight','embedding')] if stage==0 else [('model.norm.weight','norm'),('lm_head.weight','head')]
        for key,name in specials:
            target=output/(name+'.safetensors')
            if not target.exists():save_file({'weight':get(key).contiguous()},str(target))
    for name in ['config.json','tokenizer.model','tokenizer_config.json','special_tokens_map.json','generation_config.json','verified-source.json']:
        if (raw/name).exists():(output/name).write_bytes((raw/name).read_bytes())
    identity={'kind':'lossless-native-conversion','repository':REPO,'revision':REVISION,'stages':stages,'converter_sha256':digest(__file__),'files':{p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file() and p.name!='conversion.json'}}
    (output/'conversion.json').write_text(json.dumps(identity,indent=2)+'\n')
    print(json.dumps({'converted':str(output),'stages':stages,'files':len(identity['files'])}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--raw',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--stage',action='append',type=int,choices=[0,1],required=True)
    args=parser.parse_args();torch.set_num_threads(1);download(args.raw);convert(args.raw,args.output,sorted(set(args.stage)))
