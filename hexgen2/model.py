"""Native artifact model plus an explicitly labelled CPU correctness reference."""

from dataclasses import dataclass, field
from pathlib import Path
import sys

import torch
from transformers import LlamaConfig, LlamaForCausalLM


@dataclass
class CacheState:
    max_sequence_len: int
    max_batch_size: int = 1
    sequence_len_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    fused_ft_kernel: bool = False
    lengths_per_sample: object = None


class Engine:
    def __init__(self, model_dir, engine="native", device="cuda", dtype="float16"):
        self.device, self.dtype = torch.device(device), getattr(torch, dtype)
        self.config = LlamaConfig.from_pretrained(model_dir, local_files_only=True)
        from .record import model_identity
        self.identity = model_identity(model_dir)
        if self.config.model_type != "llama" or self.config.rope_scaling is not None:
            raise ValueError("This baseline supports Llama architecture without RoPE scaling")
        self.kind = engine
        if engine == "hf-reference":
            self.model = LlamaForCausalLM.from_pretrained(
                model_dir, local_files_only=True, torch_dtype=self.dtype,
                use_safetensors=True, attn_implementation="eager",
            ).to(self.device).eval()
            return
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("Native HexGen-2 requires CUDA; use hf-reference only for CPU validation")
        if torch.cuda.get_device_capability(self.device)[0] < 8:
            raise ValueError("This native FlashAttention baseline requires an Ampere-or-newer GPU")
        root = Path(__file__).resolve().parents[1]
        sys.path[:0] = [str(root / "hexgen"), str(root / "hexgen/llama")]
        from hexgen_core.models.gpt import GPTLMHeadModel
        from llama_config_utils import llama_config_to_gpt2_config
        from load_model_parameters_utils.remap_state_dict import remap_state_dict_hf_llama
        from safetensors.torch import load_file
        config = llama_config_to_gpt2_config(self.config)
        config.n_head_kv = self.config.num_key_value_heads
        config.rotary_emb_base = self.config.rope_theta
        config.use_flash_attn = True
        config.fused_bias_fc = False
        config.fused_mlp = False
        config.fused_dropout_add_ln = False
        config.use_torch_rms_norm = True
        config.sequence_parallel = False
        self.model = GPTLMHeadModel(config, device=self.device, dtype=self.dtype).eval()
        shards = sorted(Path(model_dir).glob("*.safetensors"))
        if not shards:
            raise ValueError("A local safetensors checkpoint is required")
        state = {}
        for shard in shards:
            values = load_file(str(shard), device="cpu")
            if set(state).intersection(values):
                raise ValueError("Duplicate checkpoint keys")
            state.update(values)
        # Strict loading prevents accidental use of randomly initialized weights.
        self.model.load_state_dict(remap_state_dict_hf_llama(state, config), strict=True)

    @torch.inference_mode()
    def prefill(self, input_ids, max_new_tokens):
        ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        if self.kind == "hf-reference":
            result = self.model(ids, use_cache=True)
            cache = {i: torch.stack((k.transpose(1, 2), v.transpose(1, 2)), dim=2).contiguous()
                     for i, (k, v) in enumerate(result.past_key_values)}
            logits = result.logits[:, -1]
        else:
            state = CacheState(len(input_ids) + max_new_tokens)
            logits = self.model(ids, inference_params=state, last_token_only=True).logits
            # Send only initialized prompt positions, never the unused cache capacity.
            cache = {i: t[:, :len(input_ids)].contiguous() for i, t in state.key_value_memory_dict.items()}
        return logits.argmax(-1), cache, logits

    @torch.inference_mode()
    def decode(self, first_token, cache, prompt_length, max_new_tokens, on_token=None):
        token = first_token.to(self.device).reshape(1)
        generated = [int(token.item())]
        if on_token:
            on_token(generated[-1])
        if set(cache) != set(range(self.config.num_hidden_layers)):
            raise ValueError("KV cache does not match the configured layer count")
        for value in cache.values():
            expected = (1, prompt_length, 2, self.config.num_key_value_heads,
                        self.config.hidden_size // self.config.num_attention_heads)
            if tuple(value.shape) != expected or value.dtype != self.dtype:
                raise ValueError("KV cache shape/dtype does not match checkpoint and prompt")
        if self.kind == "hf-reference":
            past = tuple((cache[i][:, :, 0].transpose(1, 2).contiguous().to(self.device),
                          cache[i][:, :, 1].transpose(1, 2).contiguous().to(self.device))
                         for i in range(len(cache)))
        else:
            state = CacheState(prompt_length + max_new_tokens)
            state.sequence_len_offset = prompt_length
            for i, live in cache.items():
                full = torch.empty((1, state.max_sequence_len, *live.shape[2:]),
                                   device=self.device, dtype=self.dtype)
                full[:, :prompt_length].copy_(live.to(self.device))
                state.key_value_memory_dict[i] = full
        for _ in range(max_new_tokens - 1):
            if self.kind == "hf-reference":
                result = self.model(token[:, None], past_key_values=past, use_cache=True)
                past, logits = result.past_key_values, result.logits[:, -1]
            else:
                logits = self.model(token[:, None], inference_params=state, last_token_only=True).logits
                state.sequence_len_offset += 1
            token = logits.argmax(-1)
            generated.append(int(token.item()))
            if on_token:
                on_token(generated[-1])
        return generated

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
