"""Two-stage native model adapter for the bounded 13B comparison.

TP=1, PP=2, equal layer counts. Each process loads exactly its owned layer files.
The imported Block/MHA/MLP implementations are selected independently per system.
"""
from dataclasses import dataclass,field
import gc
import importlib
import json
from pathlib import Path
import sys

import torch
from torch import nn
import torch.distributed as dist
from transformers import LlamaConfig
from safetensors.torch import load_file

from . import fa_adapter
from .record import digest

@dataclass
class State:
    max_sequence_len:int
    max_batch_size:int
    sequence_len_offset:int=0
    batch_size_offset:int=0
    key_value_memory_dict:dict=field(default_factory=dict)
    fused_ft_kernel:bool=False
    lengths_per_sample:object=None

class NativeStage(nn.Module):
    def __init__(self,model_dir,stage,system,baseline_root=None,max_batch=1):
        super().__init__()
        self.directory=Path(model_dir);self.stage=stage;self.audit_directory=None;self.audit_context=None;self.audit_sequence=0
        self.hf=LlamaConfig.from_pretrained(model_dir,local_files_only=True)
        if self.hf.model_type!='llama' or self.hf.rope_scaling is not None or self.hf.num_hidden_layers%2:
            raise ValueError('Requires even-layer Llama without RoPE scaling')
        # Both artifacts import an obsolete FSDP module through their package init.
        # Alias its renamed PyTorch 2.5 location; no FSDP training path is executed.
        sys.modules.setdefault('torch.distributed.fsdp.flat_param',importlib.import_module('torch.distributed.fsdp._flat_param'))
        fa_adapter.install() # Before importing either artifact's attention module.
        root=Path(__file__).resolve().parents[1]
        native=(root/'hexgen') if system=='hexgen2' else Path(baseline_root)/'hexgen'
        if not native.is_dir():raise FileNotFoundError(native)
        sys.path[:0]=[str(native),str(native/'llama')]
        from llama_config_utils import llama_config_to_gpt2_config
        import hexgen_core.models.gpt as gpt
        import hexgen_core.modules.block as block_module
        from .native_norm import RMSNorm
        # Both methods use the same explicit unfused RMSNorm/dense/MLP settings.
        gpt.RMSNorm=RMSNorm;block_module.RMSNorm=RMSNorm
        config=llama_config_to_gpt2_config(self.hf)
        for name,value in dict(n_head_kv=self.hf.num_key_value_heads,rotary_emb_base=self.hf.rope_theta,
             use_flash_attn=True,fused_bias_fc=False,fused_mlp=False,fused_dropout_add_ln=False,
             use_torch_rms_norm=True,sequence_parallel=False).items():setattr(config,name,value)
        self.start=stage*(self.hf.num_hidden_layers//2);self.stop=self.start+self.hf.num_hidden_layers//2
        self.layers=nn.ModuleDict()
        for i in range(self.start,self.stop):
            layer=gpt.create_block(config,layer_idx=i,device='cuda',dtype=torch.float16)
            file=self.directory/f'layer-{i:02}.safetensors'
            layer.load_state_dict(load_file(str(file)),strict=True)
            self.layers[str(i)]=layer.eval()
        if stage==0:
            self.embedding=nn.Embedding(self.hf.vocab_size,self.hf.hidden_size,device='cuda',dtype=torch.float16)
            self.embedding.load_state_dict(load_file(str(self.directory/'embedding.safetensors')),strict=True)
        else:
            self.norm=RMSNorm(self.hf.hidden_size,eps=self.hf.rms_norm_eps,device='cuda',dtype=torch.float16)
            self.norm.load_state_dict(load_file(str(self.directory/'norm.safetensors')),strict=True)
            self.head=nn.Linear(self.hf.hidden_size,self.hf.vocab_size,bias=False,device='cuda',dtype=torch.float16)
            self.head.load_state_dict(load_file(str(self.directory/'head.safetensors')),strict=True)
        # Allocate pinned output storage before serving. Per-token pageable
        # token.cpu() copies can synchronize unrelated CUDA communication.
        self.output_buffer=torch.empty(max_batch*self.hf.max_position_embeddings,dtype=torch.int64,pin_memory=True)
        self.eval();gc.collect();torch.cuda.empty_cache()
        self.source={'system':system,'block_module':str(Path(gpt.__file__)),
                     'block_module_sha256':digest(gpt.__file__),'layers':list(range(self.start,self.stop)),
                     'parameter_dtype':str(next(self.parameters()).dtype),'stage_parameter_bytes':sum(p.numel()*p.element_size() for p in self.parameters()),
                     'adaptations':['TP=1 PP=2 equal layers','unfused Torch RMSNorm/dense/MLP','Torch 2.5 FSDP flat_param import alias (unused training dependency)','FP16 native attention with explicit Ampere/Turing dispatch']}

    @torch.inference_mode()
    def step(self,ids,state,pair,head_rank,tail_rank):
        batch,length=ids.shape;shape=(batch,length,self.hf.hidden_size)
        if self.stage==0:
            hidden=self.embedding(ids);residual=None
        else:
            hidden=torch.empty(shape,device='cuda',dtype=torch.float16)
            residual=torch.empty_like(hidden)
            dist.recv(hidden,src=head_rank,group=pair);dist.recv(residual,src=head_rank,group=pair)
        for layer in self.layers.values():
            result=layer(hidden,residual,mixer_kwargs={'inference_params':state})
            hidden,residual=result[:2]
        if self.stage==0:
            dist.send(hidden.contiguous(),dst=tail_rank,group=pair)
            dist.send(residual.contiguous(),dst=tail_rank,group=pair)
            token=torch.empty(batch,device='cuda',dtype=torch.long)
            dist.recv(token,src=tail_rank,group=pair)
            logits=None
        else:
            hidden=self.norm((hidden+residual).to(torch.float16))
            logits=self.head(hidden[:,-1]);token=logits.argmax(-1)
            if self.audit_directory is not None:
                torch.save({'context':self.audit_context,'sequence_offset':state.sequence_len_offset,'logits':logits.float().cpu()},self.audit_directory/f'audit-{self.audit_sequence:06}.pt')
                self.audit_sequence+=1
            dist.send(token.contiguous(),dst=head_rank,group=pair)
        return token,logits

    @torch.inference_mode()
    def prefill(self,ids,count,pair,head_rank,tail_rank,prefill_only=False):
        # A disaggregated prefill worker never writes decode positions. Keeping
        # its live-prefix view must not retain unused output-capacity storage.
        state=State(ids.shape[1]+(0 if prefill_only else count),ids.shape[0])
        token,logits=self.step(ids,state,pair,head_rank,tail_rank)
        state.sequence_len_offset=ids.shape[1]
        return token,state,logits

    @torch.inference_mode()
    def decode(self,first,state,count,pair,head_rank,tail_rank,callback=None):
        token=first;tokens=[token]
        if callback:callback(token)
        for _ in range(count-1):
            token,_=self.step(token[:,None],state,pair,head_rank,tail_rank)
            state.sequence_len_offset+=1
            tokens.append(token)
            if callback:callback(token)
        output=self.output_buffer[:first.numel()*count].view(first.numel(),count)
        output.copy_(torch.stack(tokens,dim=1),non_blocking=True)
        torch.cuda.current_stream().synchronize()
        return output.tolist()

    def restore(self,tensors,prompt_length,count):
        # Received storage belongs to the transport CUDA stream. Its copies and
        # first-token reads on the compute stream must finish before reuse.
        for tensor in tensors.values():
            if tensor.is_cuda:tensor.record_stream(torch.cuda.current_stream())
        first=tensors.pop('first').to(device='cuda');state=State(prompt_length+count,first.numel(),sequence_len_offset=prompt_length)
        expected=set(map(str,range(self.start,self.stop)))
        if set(tensors)!=expected:raise ValueError('Transferred cache layer ownership mismatch')
        for key in sorted(tensors,key=int):
            live=tensors.pop(key)
            shape=(first.numel(),prompt_length,2,self.hf.num_key_value_heads,self.hf.hidden_size//self.hf.num_attention_heads)
            if tuple(live.shape)!=shape or live.dtype!=torch.float16:raise ValueError('Invalid live KV shape or dtype')
            full=torch.empty((first.numel(),state.max_sequence_len,*live.shape[2:]),device='cuda',dtype=torch.float16)
            full[:,:prompt_length].copy_(live);state.key_value_memory_dict[int(key)]=full
            del live
        return first,state
